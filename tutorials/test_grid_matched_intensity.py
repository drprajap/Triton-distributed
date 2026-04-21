#!/usr/bin/env python3
"""
Grid-Matched Compute Intensity vs CU Masking
=============================================

Apples-to-apples comparison: grid=213, CU mask=213 compute CUs.
Both regular and CU-masked streams launch the same number of WGs.
The only difference is whether hipExtStreamCreateWithCUMask is used.

Sweeps compute intensity from pure BW (load/store) to heavy FMA.
Also includes GEMM-only comparison from Tutorial 16.

Usage:
    HIP_VISIBLE_DEVICES=4 python tutorials/test_grid_matched_intensity.py
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


@triton.jit
def bw_copy_kernel(
    src_ptr, dst_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_WGS: tl.constexpr,
):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, n_blocks, NUM_WGS):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, data, mask=mask)


@triton.jit
def compute_copy_kernel(
    src_ptr, dst_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_WGS: tl.constexpr,
    COMPUTE_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, n_blocks, NUM_WGS):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(src_ptr + offsets, mask=mask).to(tl.float32)
        acc = data
        for _ in range(COMPUTE_ITERS):
            acc = acc * 0.999 + data * 0.001
        tl.store(dst_ptr + offsets, acc.to(tl.float16), mask=mask)


def time_fn(fn, warmup=3, repeats=10):
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


def run_kernel(kernel_fn, src, dst, n_elements, grid, stream, n_iters, **extra_kwargs):
    with torch.cuda.stream(stream):
        for _ in range(n_iters):
            kernel_fn[(grid,)](
                src, dst, n_elements,
                BLOCK_SIZE=8192, NUM_WGS=grid,
                num_warps=8, num_stages=1,
                **extra_kwargs,
            )


def main():
    parser = argparse.ArgumentParser(description="Grid-matched compute intensity vs CU masking")
    parser.add_argument("--grid", type=int, default=None, help="Grid (WGs). Default = mask CU count")
    parser.add_argument("--mask-cus", type=int, default=None, help="CU mask count. Default = compute CUs from comm-ratio")
    parser.add_argument("--comm-ratio", type=float, default=0.3)
    parser.add_argument("--size-mb", type=int, default=32)
    parser.add_argument("--iters", type=int, default=16)
    args = parser.parse_args()

    device_info = get_device_info()
    total_cus = device_info.total_cus

    comm_cus, compute_cus = partition_cus(total_cus, strategy="interleaved", comm_ratio=args.comm_ratio)

    if args.mask_cus is not None and args.mask_cus > len(compute_cus):
        mask_cu_list = list(range(args.mask_cus))
    elif args.mask_cus is not None:
        mask_cu_list = compute_cus[:args.mask_cus]
    else:
        mask_cu_list = compute_cus
    mask = create_cu_mask(mask_cu_list, total_cus)
    mask_count = len(mask_cu_list)

    grid = args.grid if args.grid is not None else mask_count

    size_mb = args.size_mb
    n_iters = args.iters
    dtype = torch.float16
    n_elements = (size_mb * 1024 * 1024) // 2
    src = torch.randn(n_elements, dtype=dtype, device="cuda")
    dst = torch.empty_like(src)

    regular_stream = torch.cuda.Stream()
    masked_stream = CUMaskedStreamWrapper(create_stream_with_cu_mask(mask))

    print(f"Device: {torch.cuda.get_device_name(0)}  CUs: {total_cus}")
    print(f"Grid: {grid} WGs")
    print(f"CU mask: {mask_count} CUs ({100*mask_count/total_cus:.0f}%)")
    print(f"Data: {size_mb} MB x {n_iters} iters, BLOCK_SIZE=8192, num_warps=8")
    print()

    configs = [
        ("BW-only (load/store)", bw_copy_kernel, {}),
        ("Light compute (64 FMA)", compute_copy_kernel, {"COMPUTE_ITERS": 64}),
        ("Medium compute (256 FMA)", compute_copy_kernel, {"COMPUTE_ITERS": 256}),
        ("Heavy compute (1024 FMA)", compute_copy_kernel, {"COMPUTE_ITERS": 1024}),
    ]

    print(f"{'Kernel Type':<30s} {'Regular (ms)':>12s} {'Masked (ms)':>12s} {'Slowdown':>10s}")
    print("-" * 66)

    for label, kernel_fn, extra in configs:
        times_reg = time_fn(
            lambda k=kernel_fn, e=extra: run_kernel(k, src, dst, n_elements, grid, regular_stream, n_iters, **e),
        )
        times_masked = time_fn(
            lambda k=kernel_fn, e=extra: run_kernel(k, src, dst, n_elements, grid, masked_stream, n_iters, **e),
        )
        med_reg = statistics.median(times_reg)
        med_masked = statistics.median(times_masked)
        slowdown = med_masked / med_reg if med_reg > 0 else 0
        print(f"{label:<30s} {med_reg:>12.3f} {med_masked:>12.3f} {slowdown:>9.2f}x")

    masked_stream.destroy()
    print("\nDone.")


if __name__ == "__main__":
    import os

    if os.environ.get("CU_MASK_SUITE_CALLER") != "1":
        print(
            "[DEPRECATED] Direct execution is deprecated. "
            "Use tutorials/cu_mask_suite.py intensity-sweep ... or scripts/cu_experiments.py."
        )
    main()
