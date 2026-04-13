"""
Tutorial 13: Work-Stealing GEMM + Communication Overlap

Implements a simplified work-stealing persistent GEMM kernel that uses
atomic counters to dynamically assign tiles to CUs. This makes the GEMM
kernel adaptive to the number of available CUs, contrasting with the
static grid approach in tutorials 09/12.

Compares three modes:
1. static:  standard persistent GEMM with fixed grid = total_tiles
2. ws-full: work-stealing GEMM with grid = total CUs (all CUs)
3. ws-half: work-stealing GEMM with grid = half CUs (simulates CU pressure)

Each mode is run with and without concurrent communication (Triton copy
kernel on a separate stream) to measure overlap tolerance.
"""

import argparse
import datetime
import statistics
from dataclasses import dataclass
from typing import List, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def static_matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def ws_matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    tile_counter,
    total_tiles,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    xcd_id = pid % NUM_XCDS
    tiles_per_xcd = tl.cdiv(total_tiles, NUM_XCDS)
    xcd_base = xcd_id * tiles_per_xcd
    xcd_end = min(xcd_base + tiles_per_xcd, total_tiles)
    tiles_this_xcd = xcd_end - xcd_base

    counter_ptr = tile_counter + xcd_id
    local_tile_idx = tl.atomic_add(counter_ptr, 1, scope="gpu")

    while local_tile_idx < tiles_this_xcd:
        tile_id = xcd_base + local_tile_idx

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = acc.to(C.dtype.element_ty)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

        local_tile_idx = tl.atomic_add(counter_ptr, 1, scope="gpu")


@triton.jit
def comm_copy_kernel(src, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(src + offsets, mask=mask)
    tl.store(dst + offsets, data, mask=mask)


@dataclass
class RunResult:
    mode: str
    gemm_only_ms: float
    overlap_ms: float
    comm_overhead_ms: float
    grid_size: int
    num_cus_used: int


def run_static_gemm(A, B, C, gemm_iters, warmup, repeats, total_sms) -> RunResult:
    M, K = A.shape
    _, N = B.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    GROUP_SIZE_M = 8
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (total_tiles,)

    for _ in range(warmup):
        for _ in range(gemm_iters):
            static_matmul_kernel[grid](
                A, B, C, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=GROUP_SIZE_M,
            )
        torch.cuda.synchronize()

    latencies = []
    for _ in range(repeats):
        start = datetime.datetime.now()
        for _ in range(gemm_iters):
            static_matmul_kernel[grid](
                A, B, C, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=GROUP_SIZE_M,
            )
        torch.cuda.synchronize()
        latencies.append((datetime.datetime.now() - start).total_seconds() * 1000.0)

    return RunResult("static", statistics.median(latencies), 0.0, 0.0, grid[0], total_sms)


def run_ws_gemm(A, B, C, gemm_iters, warmup, repeats, num_cus, label) -> RunResult:
    M, K = A.shape
    _, N = B.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    GROUP_SIZE_M = 8
    NUM_XCDS = 4
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (num_cus,)

    tile_counter = torch.zeros(NUM_XCDS, dtype=torch.int32, device=A.device)

    for _ in range(warmup):
        tile_counter.zero_()
        for _ in range(gemm_iters):
            tile_counter.zero_()
            ws_matmul_kernel[grid](
                A, B, C, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                tile_counter, total_tiles,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                GROUP_SIZE_M=GROUP_SIZE_M, NUM_XCDS=NUM_XCDS,
            )
        torch.cuda.synchronize()

    latencies = []
    for _ in range(repeats):
        start = datetime.datetime.now()
        for _ in range(gemm_iters):
            tile_counter.zero_()
            ws_matmul_kernel[grid](
                A, B, C, M, N, K,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                tile_counter, total_tiles,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                GROUP_SIZE_M=GROUP_SIZE_M, NUM_XCDS=NUM_XCDS,
            )
        torch.cuda.synchronize()
        latencies.append((datetime.datetime.now() - start).total_seconds() * 1000.0)

    return RunResult(label, statistics.median(latencies), 0.0, 0.0, grid[0], num_cus)


def run_with_concurrent_comm(
    run_gemm_fn, A, B, C, comm_src, comm_dst,
    gemm_iters, warmup, repeats, comm_iters,
) -> float:
    n_elements = comm_src.numel()
    BLOCK_SIZE = 1024
    comm_grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    gemm_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    for _ in range(warmup):
        with torch.cuda.stream(gemm_stream):
            run_gemm_fn()
        with torch.cuda.stream(comm_stream):
            for _ in range(comm_iters):
                comm_copy_kernel[comm_grid](comm_src, comm_dst, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()

    latencies = []
    for _ in range(repeats):
        start = datetime.datetime.now()
        with torch.cuda.stream(gemm_stream):
            run_gemm_fn()
        with torch.cuda.stream(comm_stream):
            for _ in range(comm_iters):
                comm_copy_kernel[comm_grid](comm_src, comm_dst, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()
        latencies.append((datetime.datetime.now() - start).total_seconds() * 1000.0)

    return statistics.median(latencies)


def parse_args():
    p = argparse.ArgumentParser(description="Work-stealing GEMM + communication overlap benchmark")
    p.add_argument("--M", type=int, default=8192)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--K", type=int, default=8192)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--gemm-iters", type=int, default=4, help="GEMM iterations per timed step")
    p.add_argument("--comm-size-mb", type=float, default=32.0, help="Communication buffer size in MB")
    p.add_argument("--comm-iters", type=int, default=8, help="Communication kernel iterations per step")
    p.add_argument("--validate", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    total_sms = torch.cuda.get_device_properties(0).multi_processor_count

    A = torch.randn(args.M, args.K, dtype=dtype, device=device) * 0.01
    B = torch.randn(args.K, args.N, dtype=dtype, device=device) * 0.01
    C = torch.empty(args.M, args.N, dtype=dtype, device=device)

    comm_elements = int(args.comm_size_mb * 1024 * 1024 / 2)
    comm_src = torch.randn(comm_elements, dtype=dtype, device=device)
    comm_dst = torch.empty_like(comm_src)

    if args.validate:
        ref = torch.matmul(A, B)
        run_static_gemm(A, B, C, 1, 1, 1, total_sms)
        static_ok = torch.allclose(C, ref, atol=1e-1, rtol=1e-1)
        tile_counter = torch.zeros(4, dtype=torch.int32, device=device)
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        total_tiles = triton.cdiv(args.M, BLOCK_M) * triton.cdiv(args.N, BLOCK_N)
        ws_matmul_kernel[(total_sms,)](
            A, B, C, args.M, args.N, args.K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            tile_counter, total_tiles,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=8, NUM_XCDS=4,
        )
        torch.cuda.synchronize()
        ws_ok = torch.allclose(C, ref, atol=1e-1, rtol=1e-1)
        print(f"Validation: static={'PASS' if static_ok else 'FAIL'}, ws={'PASS' if ws_ok else 'FAIL'}")
        if not (static_ok and ws_ok):
            return

    half_sms = total_sms // 2
    quarter_sms = total_sms // 4
    three_quarter_sms = (total_sms * 3) // 4

    print(f"\n=== Work-Stealing GEMM Overlap Experiment ===")
    print(f"Shape: M={args.M}, N={args.N}, K={args.K}, dtype={args.dtype}")
    print(f"total_sms={total_sms}, gemm_iters={args.gemm_iters}, comm_size={args.comm_size_mb}MB, "
          f"comm_iters={args.comm_iters}, warmup={args.warmup}, repeats={args.repeats}")

    results: List[RunResult] = []

    r_static = run_static_gemm(A, B, C, args.gemm_iters, args.warmup, args.repeats, total_sms)
    results.append(r_static)

    r_ws_full = run_ws_gemm(A, B, C, args.gemm_iters, args.warmup, args.repeats, total_sms, "ws-full")
    results.append(r_ws_full)

    r_ws_3q = run_ws_gemm(A, B, C, args.gemm_iters, args.warmup, args.repeats, three_quarter_sms, "ws-3/4")
    results.append(r_ws_3q)

    r_ws_half = run_ws_gemm(A, B, C, args.gemm_iters, args.warmup, args.repeats, half_sms, "ws-half")
    results.append(r_ws_half)

    r_ws_quarter = run_ws_gemm(A, B, C, args.gemm_iters, args.warmup, args.repeats, quarter_sms, "ws-quarter")
    results.append(r_ws_quarter)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    total_tiles = triton.cdiv(args.M, BLOCK_M) * triton.cdiv(args.N, BLOCK_N)
    tile_counter = torch.zeros(4, dtype=torch.int32, device=A.device)

    def make_static_fn():
        grid = (total_tiles,)
        def fn():
            for _ in range(args.gemm_iters):
                static_matmul_kernel[grid](
                    A, B, C, args.M, args.N, args.K,
                    A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                    C.stride(0), C.stride(1),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=8,
                )
        return fn

    def make_ws_fn(num_cus):
        grid = (num_cus,)
        def fn():
            for _ in range(args.gemm_iters):
                tile_counter.zero_()
                ws_matmul_kernel[grid](
                    A, B, C, args.M, args.N, args.K,
                    A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                    C.stride(0), C.stride(1),
                    tile_counter, total_tiles,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                    GROUP_SIZE_M=8, NUM_XCDS=4,
                )
        return fn

    cus_map = {"ws-full": total_sms, "ws-3/4": three_quarter_sms,
               "ws-half": half_sms, "ws-quarter": quarter_sms}
    for r in results:
        if r.mode == "static":
            gemm_fn = make_static_fn()
        else:
            gemm_fn = make_ws_fn(cus_map[r.mode])

        overlap_ms = run_with_concurrent_comm(
            gemm_fn, A, B, C, comm_src, comm_dst,
            args.gemm_iters, args.warmup, args.repeats, args.comm_iters,
        )
        r.overlap_ms = overlap_ms
        r.comm_overhead_ms = overlap_ms - r.gemm_only_ms

    print(f"\n{'Mode':<12} {'Grid':>6} {'CUs':>5} {'GEMM-only(ms)':>14} {'w/Comm(ms)':>12} {'Overhead(ms)':>13} {'Overhead%':>10}")
    for r in results:
        pct = (r.comm_overhead_ms / r.gemm_only_ms * 100) if r.gemm_only_ms > 0 else 0
        print(f"{r.mode:<12} {r.grid_size:>6} {r.num_cus_used:>5} {r.gemm_only_ms:>14.3f} {r.overlap_ms:>12.3f} {r.comm_overhead_ms:>13.3f} {pct:>9.1f}%")

    print(f"\nKey insight: work-stealing should show lower overhead% under CU pressure")
    print(f"because it adapts to whatever CUs are available, while static grid stalls.")


if __name__ == "__main__":
    main()
