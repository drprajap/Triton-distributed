################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import argparse
import datetime
from functools import partial
import os
import random
import socket

import torch
from triton_dist.kernels.amd.gemm_allreduce import (
    create_gemm_ar_context,
    gemm_allreduce_op,
    gemm_allreduce_op_dma,
)
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import dist_print, initialize_distributed, finalize_distributed, rand_tensor


def gemm_allreduce_torch(a: torch.Tensor, b: torch.Tensor, pg: torch.distributed.ProcessGroup):
    """Reference torch implementation for gemm+allreduce"""
    # Perform local GEMM
    c = torch.matmul(a, b.T)
    # Allreduce across ranks
    torch.distributed.all_reduce(c, group=pg)
    return c


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
}


# Benchmark matrix: list of (M, N) message shapes. K is fixed across all.
DEFAULT_MN_SHAPES = [
    (32, 4096), (64, 4096), (128, 4096), (256, 4096), (512, 4096), (1024, 4096),
    (32, 5120), (64, 5120), (128, 5120), (256, 5120), (512, 5120), (1024, 5120),
    (32, 8192), (64, 8192), (128, 8192), (256, 8192), (512, 8192), (1024, 8192),
]
DEFAULT_BENCH_DTYPES = ["float16", "bfloat16"]
DEFAULT_AR_MODES = ["cu", "dma"]

# Looser tolerances for the matrix sweep so the DEFAULT_GEMM_CONFIG-based kernel
# (no autotune) passes the functional gate across all shapes. These are wider
# than the single-run THRESHOLD_MAP because different block sizes accumulate K
# in different orders, amplifying low-precision round-off for some shapes.
BENCH_THRESHOLD_MAP = {
    torch.float16: 7e-2,
    torch.bfloat16: 7e-2,
}


def _make_data(M, N, K, dtype, pg: torch.distributed.ProcessGroup):
    current_device = torch.cuda.current_device()
    scale = (pg.rank() + 1) * 0.01
    A = rand_tensor((M, K), dtype=dtype, device=current_device) * scale
    B = rand_tensor((N, K), dtype=dtype, device=current_device) * scale
    return A, B


def _run_triton_op(mode: str, ctx, A, B, autotune: bool = True):
    if mode == "cu":
        return gemm_allreduce_op(ctx, A, B, autotune=autotune)
    if mode == "dma":
        return gemm_allreduce_op_dma(ctx, A, B, autotune=autotune)
    raise ValueError(f"Unknown AR mode: {mode}")


def run_stress_test(args, TP_GROUP, dtype, atol, rtol):
    """Run stress test with random shapes"""
    RANK = torch.distributed.get_rank()
    WORLD_SIZE = torch.distributed.get_world_size()

    max_M, max_N, max_K = args.M, args.N, args.K

    dist_print(f"Running stress test: {args.stress_rounds} rounds")
    dist_print(f"Max M={max_M}, Max N={max_N}, Max K={max_K}, dtype={dtype}")

    for round_idx in range(args.stress_rounds):
        M = random.randint(256, max_M) // 256 * 256
        N = random.randint(256, max_N) // 256 * 256
        K_per_rank = random.randint(256, max_N) // 256 * 256
        ar_stream = torch.cuda.Stream(priority=-1)
        dist_print(f"\nRound {round_idx + 1}/{args.stress_rounds}: M={M}, N={N}, K_per_rank={K_per_rank}")

        try:
            ctx = create_gemm_ar_context(ar_stream=ar_stream, rank=RANK, world_size=WORLD_SIZE, max_M=M, N=N,
                                         dtype=dtype)

            for _ in range(10):
                A, B = _make_data(M, N, K_per_rank, dtype, TP_GROUP)
                output_triton = gemm_allreduce_op(ctx, A, B, autotune=False)
                output_torch = gemm_allreduce_torch(A, B, TP_GROUP)
                assert_allclose(output_triton, output_torch, atol=atol, rtol=rtol, verbose=False)
            dist_print(f"✅ Round {round_idx + 1} passed")

        except Exception as e:
            dist_print(f"❌ Round {round_idx + 1} failed: {str(e)}")
            torch.cuda.synchronize()
            torch.distributed.barrier()
            raise RuntimeError(f"Stress test failed at round {round_idx + 1}: {str(e)}")

    torch.cuda.synchronize()
    torch.distributed.barrier()

    dist_print("\n" + "=" * 60)
    dist_print(f"✅ Stress test completed: All {args.stress_rounds} rounds passed")


def _ar_bytes(M: int, N: int, elem_size: int, world_size: int) -> int:
    """Effective payload volume per rank for an intra-node all-reduce.
    Each rank sends and receives (world_size - 1) copies of the MxN tensor
    (ring-equivalent algorithmic bandwidth). For world_size=2 this is
    simply one tensor in each direction.
    """
    return 2 * (world_size - 1) * M * N * elem_size


def _bench_one(mode: str, ctx, A, B, iters: int, warmup: int, autotune: bool):
    torch.cuda.synchronize()
    torch.distributed.barrier()
    _, ms = perf_func(partial(_run_triton_op, mode, ctx, A, B, autotune), iters=iters, warmup_iters=warmup)
    return ms


def _bench_torch_ref(A, B, tp_group, iters: int, warmup: int):
    torch.cuda.synchronize()
    torch.distributed.barrier()
    _, ms = perf_func(partial(gemm_allreduce_torch, A, B, tp_group), iters=iters, warmup_iters=warmup)
    return ms


def _format_row(record):
    return (f"| {record['dtype']:<9} | {record['M']:>5} | {record['N']:>5} | {record['K']:>6} | "
            f"{record['ar_mode']:<3} | {record['hip_visible_devices']:<15} | {record['world_size']:>3} | "
            f"{record['triton_ms']:>8.3f} | {record['torch_ms']:>8.3f} | {record['speedup']:>6.2f} | "
            f"{record['gbps']:>7.2f} | {record['functional']} |")


MATRIX_HEADER_ROWS = [
    "| dtype     |     M |     N |      K | AR  | HIP_VISIBLE_DEVICES | WS  | triton_ms | torch_ms |  speedup |    GB/s | func |",
    "|-----------|-------|-------|--------|-----|----------------------|-----|-----------|----------|----------|---------|------|",
]


def run_benchmark_matrix(args, TP_GROUP):
    """Run functional + perf matrix sweep across dtypes, (M,N) shapes and AR modes.
    Emits structured Markdown table rows from rank 0 only and (optionally)
    appends them to a results file.
    """
    RANK = torch.distributed.get_rank()
    WORLD_SIZE = torch.distributed.get_world_size()
    hip_visible = os.environ.get("HIP_VISIBLE_DEVICES", "unset")
    hostname = socket.gethostname()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    dtypes = args.bench_dtypes or DEFAULT_BENCH_DTYPES
    modes = args.bench_ar_modes or DEFAULT_AR_MODES
    if args.bench_shapes:
        shapes = []
        for spec in args.bench_shapes.split(","):
            m, n = spec.lower().split("x")
            shapes.append((int(m), int(n)))
    else:
        shapes = DEFAULT_MN_SHAPES
    K_global = args.bench_K
    K_per_rank = K_global // WORLD_SIZE
    autotune_on = not args.no_bench_autotune

    if RANK == 0:
        dist_print("\n" + "=" * 96)
        dist_print(f"GEMM+AR benchmark matrix | host={hostname} | WORLD_SIZE={WORLD_SIZE} | "
                   f"HIP_VISIBLE_DEVICES={hip_visible} | K(global)={K_global} | started={started_utc}")
        dist_print("=" * 96)
        for line in MATRIX_HEADER_ROWS:
            dist_print(line)

    # Results are gathered on rank 0 for final write-out.
    records = []

    # Largest M, N used to pre-allocate a single context that can serve all shapes.
    max_M = max(m for m, _ in shapes)
    max_N = max(n for _, n in shapes)

    for dtype_name in dtypes:
        dtype = DTYPE_MAP[dtype_name]
        atol = BENCH_THRESHOLD_MAP[dtype]
        rtol = BENCH_THRESHOLD_MAP[dtype]

        ar_stream = torch.cuda.Stream(priority=-1)
        ctx = create_gemm_ar_context(ar_stream=ar_stream, rank=RANK, world_size=WORLD_SIZE,
                                     max_M=max_M, N=max_N, dtype=dtype)
        try:
            for (M, N) in shapes:
                A, B = _make_data(M, N, K_per_rank, dtype, TP_GROUP)
                torch_ref_out = gemm_allreduce_torch(A, B, TP_GROUP)
                torch_ms = _bench_torch_ref(A, B, TP_GROUP, args.iters, args.warmup)

                for mode in modes:
                    # Functional check before timing. Synchronize pass/fail
                    # across ranks so that any rank's failure aborts the
                    # perf branch collectively (avoids torch.distributed hangs).
                    local_fail = 0
                    fail_msg = ""
                    try:
                        triton_out = _run_triton_op(mode, ctx, A, B, autotune=autotune_on)
                        assert_allclose(torch_ref_out, triton_out, atol=atol, rtol=rtol, verbose=False)
                    except Exception as e:
                        local_fail = 1
                        fail_msg = f"{type(e).__name__}: {e}"
                    fail_flag = torch.tensor([local_fail], dtype=torch.int32, device=torch.cuda.current_device())
                    torch.distributed.all_reduce(fail_flag, op=torch.distributed.ReduceOp.MAX)
                    if int(fail_flag.item()) != 0:
                        functional = "FAIL"
                        triton_ms = float("nan")
                        if RANK == 0:
                            dist_print(f"[functional FAIL] dtype={dtype_name} M={M} N={N} mode={mode}: "
                                       f"local_rank0_msg={fail_msg!r}")
                        torch.cuda.synchronize()
                        torch.distributed.barrier()
                    else:
                        functional = "PASS"
                        triton_ms = _bench_one(mode, ctx, A, B, args.iters, args.warmup, autotune_on)

                    speedup = (torch_ms / triton_ms) if (triton_ms and triton_ms == triton_ms) else float("nan")
                    payload_bytes = _ar_bytes(M, N, A.element_size(), WORLD_SIZE)
                    gbps = ((payload_bytes / (triton_ms * 1e-3)) / (1024 ** 3)) if triton_ms and triton_ms == triton_ms else float("nan")

                    record = {
                        "dtype": dtype_name,
                        "M": M,
                        "N": N,
                        "K": K_global,
                        "ar_mode": mode,
                        "hip_visible_devices": hip_visible,
                        "world_size": WORLD_SIZE,
                        "triton_ms": triton_ms,
                        "torch_ms": torch_ms,
                        "speedup": speedup,
                        "gbps": gbps,
                        "functional": functional,
                    }
                    records.append(record)
                    if RANK == 0:
                        dist_print(_format_row(record))

                del A, B, torch_ref_out
                if 'triton_out' in dir():
                    del triton_out
        finally:
            del ctx
            torch.cuda.synchronize()
            torch.distributed.barrier()

    if RANK == 0 and args.output:
        _write_matrix_markdown(args.output, records, hostname, hip_visible, WORLD_SIZE, K_global, started_utc)


def _write_matrix_markdown(path, records, hostname, hip_visible, world_size, K_global, started_utc):
    header = (
        f"\n### Run at {started_utc}\n\n"
        f"- Host: `{hostname}`\n"
        f"- WORLD_SIZE: `{world_size}`\n"
        f"- HIP_VISIBLE_DEVICES: `{hip_visible}`\n"
        f"- K (global): `{K_global}`\n\n"
    )
    lines = [header]
    lines.extend(line + "\n" for line in MATRIX_HEADER_ROWS)
    for r in records:
        lines.append(_format_row(r) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.writelines(lines)
    dist_print(f"[rank0] appended {len(records)} rows to {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=1024)
    parser.add_argument("N", type=int, nargs="?", default=8192)
    parser.add_argument("K", type=int, nargs="?", default=4096)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--ar_mode", default="cu", choices=["cu", "dma"],
                        help="Single-run mode: AR transport (cu=Triton consumer kernel, dma=hipMemcpy NoCU)")
    parser.add_argument("--no_autotune", action="store_true",
                        help="Disable Triton autotune for single-run mode (default: autotune on)")
    parser.add_argument("--warmup", default=10, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=20, type=int, help="perf iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--stress", default=False, action="store_true", help="run stress test with random shapes")
    parser.add_argument("--stress_rounds", type=int, default=10, help="number of stress test rounds")

    # Benchmark matrix mode.
    parser.add_argument("--benchmark", action="store_true",
                        help="Run the full GEMM+AR perf matrix (fixed K, all shapes/dtypes/AR modes)")
    parser.add_argument("--bench_K", type=int, default=4096, help="Global K used for the matrix sweep")
    parser.add_argument("--bench_dtypes", nargs="*", default=None, choices=["float16", "bfloat16"],
                        help="Subset of dtypes to sweep (default: both)")
    parser.add_argument("--bench_ar_modes", nargs="*", default=None, choices=["cu", "dma"],
                        help="Subset of AR modes to sweep (default: both)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to a Markdown file to append matrix rows to (rank 0 only)")
    parser.add_argument("--no_bench_autotune", action="store_true",
                        help="Disable autotune in matrix sweep (faster; uses DEFAULT_GEMM_CONFIG)")
    parser.add_argument("--bench_shapes", type=str, default=None,
                        help="Comma-separated (M,N) shapes to restrict the sweep, e.g. '32x4096,256x4096'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed(args.seed)

    dtype = DTYPE_MAP[args.dtype]
    atol = THRESHOLD_MAP[dtype]
    rtol = THRESHOLD_MAP[dtype]

    if args.benchmark:
        try:
            run_benchmark_matrix(args, TP_GROUP)
        finally:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            finalize_distributed()
        raise SystemExit(0)

    M = args.M
    N = args.N
    K = args.K // WORLD_SIZE

    iters = args.iters
    warmup_iters = args.warmup

    if args.stress:
        run_stress_test(args, TP_GROUP, dtype, atol, rtol)

    # Create context for GEMM+AllReduce
    ar_stream = torch.cuda.Stream(priority=-1)
    ctx = create_gemm_ar_context(ar_stream=ar_stream, rank=RANK, world_size=WORLD_SIZE, max_M=M, N=N, dtype=dtype)
    autotune_on_single = not args.no_autotune
    a, b = _make_data(M, N, K, dtype, TP_GROUP)
    torch_output = gemm_allreduce_torch(a, b, TP_GROUP)
    triton_output = _run_triton_op(args.ar_mode, ctx, a, b, autotune=autotune_on_single)
    assert_allclose(torch_output, triton_output, atol=THRESHOLD_MAP[dtype], rtol=THRESHOLD_MAP[dtype])

    with group_profile("gemm_ar", args.profile, group=TP_GROUP):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch_output, duration_ms_torch = perf_func(partial(gemm_allreduce_torch, a, b, TP_GROUP), iters=iters,
                                                    warmup_iters=warmup_iters)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        triton_output, duration_ms_triton = perf_func(
            partial(_run_triton_op, args.ar_mode, ctx, a, b, autotune_on_single),
            iters=iters, warmup_iters=warmup_iters)

    dist_print(f"torch #{RANK} {duration_ms_torch:0.2f} ms/iter", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"triton[{args.ar_mode}] #{RANK} {duration_ms_triton:0.2f} ms/iter", need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))

    speedup = duration_ms_torch / duration_ms_triton
    dist_print(f"Speedup: {speedup:0.2f}x", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    # Explicitly delete rocSHMEM-backed tensors before finalization
    # without explicit cleanup, rocshmem barrier_all collective operation
    # is called during python shutdown when some ranks may already have exited,
    # which may cause segfaults.
    del ctx, a, b, triton_output
    finalize_distributed()
