"""Minimal pure AR bandwidth sweep on ROCm/MI355.

Compares two apples-to-apples AR transports at the same world-size /
payload grid, using the standard nccl-tests bandwidth convention::

    bytes   = count * sizeof(dtype)
    algbw   = bytes / time_s / 1e9                     # algorithmic GB/s
    busbw   = algbw * 2 * (world - 1) / world          # ring bus GB/s

Supported transports:
  - torch       : torch.distributed.all_reduce  (RCCL under the hood).
  - triton_dma  : Triton-distributed pure-AR DMA kernel (the DMA transport
                  from gemm_allreduce_op_dma, run without any fused GEMM).

Usage (launched via scripts/launch_amd.sh)::

    ARNOLD_WORKER_GPU=8 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        bash scripts/launch_amd.sh python/triton_dist/test/amd/bench_pure_ar.py \
        --transport torch --output /workspace/gemm_ar_perf/mi355_pure_ar_ws8.md

Emits a markdown table (rank-0 only) that fits directly into the report.
"""
from __future__ import annotations

import argparse
import os
from typing import List

import torch
import torch.distributed as dist

# Lazy: only imported when --transport is triton_*
_TRITON_DIST_INITIALIZED = False


DEFAULT_SIZES_BYTES = [
    1 << 10,        # 1 KB
    4 << 10,        # 4 KB
    16 << 10,       # 16 KB
    64 << 10,       # 64 KB
    256 << 10,      # 256 KB
    1 << 20,        # 1 MB
    4 << 20,        # 4 MB
    16 << 20,       # 16 MB
    64 << 20,       # 64 MB
    256 << 20,      # 256 MB
    1 << 30,        # 1 GB
]


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

TRANSPORTS = ("torch", "triton_dma", "triton_one_shot")


def human_bytes(n: int) -> str:
    for unit, thresh in [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
        if n >= thresh:
            v = n / thresh
            return f"{v:.1f}{unit}" if v < 10 else f"{int(v)}{unit}"
    return f"{n}B"


def bench_one_torch(tensor: torch.Tensor, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dist.all_reduce(tensor)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def bench_one_triton_dma(ctx, input_tensor, ar_op, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        ar_op(ctx, input_tensor)
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        ar_op(ctx, input_tensor)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", default="torch", choices=list(TRANSPORTS))
    ap.add_argument("--dtype", default="float16", choices=list(DTYPES.keys()))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sizes_bytes", type=str, default="",
                    help="Comma-separated list of payload sizes in bytes. "
                         "Empty = use built-in log-scale sweep.")
    ap.add_argument("--max_bytes", type=int, default=256 << 20,
                    help="Cap payload size for the sweep. Default 256 MB "
                         "which fits a reasonable rocSHMEM heap.")
    ap.add_argument("--ctx_N", type=int, default=8192,
                    help="N column count for the triton symmetric buffer "
                         "(only used for --transport triton_dma).")
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    if args.transport in ("triton_dma", "triton_one_shot"):
        # triton_dist.initialize_distributed handles both torch.dist and
        # rocshmem init in the correct order for AMD.
        from triton_dist.utils import initialize_distributed, finalize_distributed  # noqa: F401
        initialize_distributed()
    else:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dtype = DTYPES[args.dtype]
    elem = torch.tensor([], dtype=dtype).element_size()

    sizes: List[int]
    if args.sizes_bytes:
        sizes = [int(s) for s in args.sizes_bytes.split(",") if s.strip()]
    else:
        sizes = [s for s in DEFAULT_SIZES_BYTES if s <= args.max_bytes]

    hvd = os.environ.get("HIP_VISIBLE_DEVICES", "<unset>")

    # Transport setup.
    ctx = None
    ar_op = None
    if args.transport in ("triton_dma", "triton_one_shot"):
        # Lazy import so --transport torch does not require triton_dist build.
        from triton_dist.kernels.amd.gemm_allreduce import (
            create_gemm_ar_context,
            pure_allreduce_dma_op,
            pure_allreduce_one_shot_op,
        )
        N = args.ctx_N
        max_bytes = max(sizes)
        max_numel = max_bytes // elem
        # Round up max_M so max_M * N >= max_numel.
        max_M = (max_numel + N - 1) // N
        if rank == 0:
            print(f"# {args.transport} context: max_M={max_M}  N={N}  "
                  f"symm_bytes≈{max_M * N * elem / (1<<20):.1f} MB / rank",
                  flush=True)
        ar_stream = torch.cuda.Stream()
        ctx = create_gemm_ar_context(ar_stream, rank, world, max_M=max_M, N=N,
                                     dtype=dtype)
        if args.transport == "triton_dma":
            ar_op = pure_allreduce_dma_op
        else:
            ar_op = pure_allreduce_one_shot_op

    if rank == 0:
        print(f"# Pure all_reduce bandwidth sweep  (transport={args.transport})",
              flush=True)
        print(f"# dtype={args.dtype}  world={world}  "
              f"HIP_VISIBLE_DEVICES={hvd}  iters={args.iters} warmup={args.warmup}",
              flush=True)
        print("", flush=True)
        print("| transport | dtype | bytes | human | count | world | "
              "HIP_VISIBLE_DEVICES | time_ms | algbw_GBps | busbw_GBps |", flush=True)
        print("|---|---|---:|---|---:|---:|---|---:|---:|---:|", flush=True)

    rows = []
    # Correctness sanity check for triton transports: run once on a 16KB payload
    # with per-rank-distinct values and compare against torch.distributed.
    if args.transport in ("triton_dma", "triton_one_shot"):
        vsize = 8192
        v = torch.full((vsize, ), float(rank + 1), dtype=dtype, device="cuda")
        ref = v.clone()
        # Initialize ref via torch's RCCL backend (already initialized as part
        # of triton_dist.initialize_distributed).
        dist.all_reduce(ref)
        got = ar_op(ctx, v)
        got_cpu = got.detach().float().cpu()
        ref_cpu = ref.detach().float().cpu()
        ok = torch.allclose(got_cpu, ref_cpu, rtol=1e-2, atol=1e-2)
        if rank == 0:
            expected_scalar = sum(r + 1 for r in range(world))
            print(f"# correctness: transport={args.transport} world={world} "
                  f"expected={expected_scalar} got[0]={got_cpu[0].item():.3f} "
                  f"ok={ok}", flush=True)
        if not ok:
            if rank == 0:
                print(f"# FAIL: first mismatch @0: got={got_cpu[0].item()} "
                      f"ref={ref_cpu[0].item()}", flush=True)
            torch.cuda.synchronize()
            os._exit(2)
        del v, ref, got
        torch.cuda.synchronize()
        dist.barrier()

    for nbytes in sizes:
        count = nbytes // elem
        if count <= 0:
            continue
        try:
            t = torch.empty(count, dtype=dtype, device="cuda")
            t.fill_(1.0)
        except RuntimeError as e:
            if rank == 0:
                print(f"# skip {human_bytes(nbytes)}: {e}", flush=True)
            continue

        try:
            if args.transport == "torch":
                ms = bench_one_torch(t, args.iters, args.warmup)
            elif args.transport in ("triton_dma", "triton_one_shot"):
                ms = bench_one_triton_dma(ctx, t, ar_op, args.iters, args.warmup)
            else:
                raise ValueError(f"unknown transport {args.transport}")
        except RuntimeError as e:
            if rank == 0:
                print(f"# failed {human_bytes(nbytes)}: {e}", flush=True)
            del t
            torch.cuda.empty_cache()
            continue

        time_s = ms * 1e-3
        algbw = nbytes / time_s / 1e9
        busbw = algbw * 2.0 * (world - 1) / world
        if rank == 0:
            row = (f"| {args.transport} | {args.dtype} | {nbytes} | {human_bytes(nbytes)} "
                   f"| {count} | {world} | {hvd} | {ms:.4f} | {algbw:.2f} | {busbw:.2f} |")
            print(row, flush=True)
            rows.append(row)
        del t
        torch.cuda.empty_cache()

    if rank == 0 and args.output:
        header = [
            f"# Pure all_reduce bandwidth sweep  (transport={args.transport})",
            f"# dtype={args.dtype}  world={world}  HIP_VISIBLE_DEVICES={hvd}  "
            f"iters={args.iters} warmup={args.warmup}",
            "",
            "| transport | dtype | bytes | human | count | world | "
            "HIP_VISIBLE_DEVICES | time_ms | algbw_GBps | busbw_GBps |",
            "|---|---|---:|---|---:|---:|---|---:|---:|---:|",
        ]
        with open(args.output, "a") as f:
            for line in header:
                f.write(line + "\n")
            for row in rows:
                f.write(row + "\n")
            f.write("\n")

    dist.barrier()
    if args.output and rank == 0:
        try:
            os.fsync(os.open(args.output, os.O_RDONLY))
        except OSError:
            pass
    # rocSHMEM teardown can segfault on some stacks after we've released
    # tensor lists. All measurement data is already persisted at this point,
    # so force-exit cleanly to avoid masking real failures with dealloc noise.
    if args.transport in ("triton_dma", "triton_one_shot"):
        torch.cuda.synchronize()
        os._exit(0)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
