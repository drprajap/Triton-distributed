#!/usr/bin/env python3
"""
Minimal GEMM-only CU Masking Benchmark
========================================

Same tile config as the AG+GEMM kernel (256x256x64, 8 warps, 2 stages)
but no AG, no distributed, no barriers. Single GPU.

Isolates whether hipExtStreamCreateWithCUMask adds overhead
to kernel dispatch/execution for a persistent GEMM.

Usage:
    export PYTHONPATH=$PYTHONPATH:$(pwd)/python
    python tutorials/16-gemm-only-cu-mask.py --num-sms 213 --gemm-iters 8
    python tutorials/16-gemm-only-cu-mask.py --num-sms 213 --gemm-iters 8 --profile
"""
import argparse
import statistics
import time

import torch
import triton
import triton.language as tl

from triton_dist.cu_masking import (
    CUMaskedStreamWrapper,
    create_cu_mask,
    create_stream_with_cu_mask,
    get_device_info,
    partition_cus,
)


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 1,
                'waves_per_eu': 2,
                'kpack': 1,
                'matrix_instr_nonkdim': 16,
            },
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0})
@triton.jit
def persistent_gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        tl.assume(pid_m > 0)
        tl.assume(pid_n > 0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)


def run_gemm(A, B, C, M, N, K, num_sms, stream, gemm_iters):
    grid = lambda META: (
        min(num_sms, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),
    )
    with torch.cuda.stream(stream):
        for _ in range(gemm_iters):
            persistent_gemm_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(1), B.stride(0),
                C.stride(0), C.stride(1),
                NUM_SMS=num_sms,
                NUM_XCDS=4,
            )


def time_fn(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def main():
    parser = argparse.ArgumentParser(description="GEMM-only CU mask benchmark")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=5504, help="N_per_rank for 2-rank TP with N=11008")
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--num-sms", type=int, default=213)
    parser.add_argument("--gemm-iters", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--comm-ratio", type=float, default=0.3)
    parser.add_argument("--profile", action="store_true",
                        help="Reduce to 1 repeat for profiling with rocprofv3")
    args = parser.parse_args()

    if args.profile:
        args.warmup = 1
        args.repeats = 1

    device_info = get_device_info()
    total_cus = device_info.total_cus
    comm_cus, compute_cus = partition_cus(
        total_cus, strategy="interleaved", comm_ratio=args.comm_ratio
    )

    dtype = torch.float16
    A = torch.randn((args.M, args.K), dtype=dtype, device="cuda")
    B = torch.randn((args.N, args.K), dtype=dtype, device="cuda")
    C = torch.empty((args.M, args.N), dtype=dtype, device="cuda")

    num_tiles_m = (args.M + 255) // 256
    num_tiles_n = (args.N + 255) // 256
    total_tiles = num_tiles_m * num_tiles_n

    print(f"Device: {torch.cuda.get_device_name(0)}  CUs: {total_cus}")
    print(f"GEMM: M={args.M}, N={args.N}, K={args.K}, tiles={num_tiles_m}x{num_tiles_n}={total_tiles}")
    print(f"Grid (NUM_SMS): {args.num_sms}, gemm_iters: {args.gemm_iters}")
    print(f"Tile executions per call: {total_tiles}, waves: {(total_tiles + args.num_sms - 1) // args.num_sms}")
    print(f"CU mask: {len(compute_cus)} compute CUs ({100*len(compute_cus)/total_cus:.0f}%)")
    print(f"warmup: {args.warmup}, repeats: {args.repeats}")

    regular_stream = torch.cuda.Stream()
    compute_mask = create_cu_mask(compute_cus, total_cus)
    masked_stream = CUMaskedStreamWrapper(create_stream_with_cu_mask(compute_mask))

    modes = [
        ("Regular stream", regular_stream),
        (f"CU-masked stream ({len(compute_cus)} CUs)", masked_stream),
    ]

    print(f"\n{'Mode':<45s} {'Median(ms)':>10s} {'Mean(ms)':>10s} {'Std(ms)':>10s}")
    results = {}
    for label, stream in modes:
        times = time_fn(
            lambda s=stream: run_gemm(A, B, C, args.M, args.N, args.K, args.num_sms, s, args.gemm_iters),
            args.warmup, args.repeats,
        )
        med = statistics.median(times)
        mn = statistics.mean(times)
        sd = statistics.pstdev(times)
        print(f"{label:<45s} {med:>10.3f} {mn:>10.3f} {sd:>10.3f}")
        results[label] = med

    baseline = list(results.values())[0]
    print("\nSpeedups (vs regular stream):")
    for label, med in results.items():
        print(f"  {label:<45s}: {baseline / med:.3f}x")

    masked_stream.destroy()


if __name__ == "__main__":
    import os

    if os.environ.get("CU_MASK_SUITE_CALLER") != "1":
        print(
            "[DEPRECATED] Direct execution is deprecated. "
            "Use tutorials/cu_mask_suite.py compute-only ... or scripts/cu_experiments.py."
        )
    main()
