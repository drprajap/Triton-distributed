#!/usr/bin/env python3
"""
Multi-GPU Comm-Only Benchmark: DMA vs CU-Kernel, with/without CU Masking
==========================================================================

P2P transfers between 2 GPUs via rocshmem IPC memory (XGMI).
Isolates CU masking effect on pure inter-GPU communication.

Tests a matrix of modes:
  - DMA (hipMemcpyDeviceToDeviceNoCU) on regular/CU-masked streams
  - Triton copy kernel on regular/CU-masked streams
  - Grid-limited copy kernel (persistent style, 70% WGs) on regular/CU-masked streams

Usage:
    HIP_VISIBLE_DEVICES=4,5 ARNOLD_WORKER_GPU=2 \\
      ./scripts/launch_amd.sh tutorials/15-comm-only-benchmark.py \\
      --size-mb 32 --copy-iters 16
"""
import argparse
import importlib.util
import statistics
import time
import gc
from pathlib import Path

import torch
import triton
import triton.language as tl

import pyrocshmem
from hip import hip
from triton_dist.utils import HIP_CHECK
from triton_dist.cu_masking import (
    CUMaskedStreamWrapper,
    create_cu_mask,
    create_stream_with_cu_mask,
    get_device_info,
    partition_cus,
)


def load_tutorial09():
    path = Path(__file__).with_name("09-AMD-overlapping-allgather-gemm.py")
    spec = importlib.util.spec_from_file_location("t09", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t09 = load_tutorial09()


@triton.jit
def copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, data, mask=mask)


@triton.jit
def persistent_copy_kernel(
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
def heavy_copy_kernel(
    src_ptr, dst_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_WGS: tl.constexpr,
):
    """Persistent copy kernel with large BLOCK_SIZE and num_warps to inflate
    register pressure and force ~1 WG/CU occupancy.  This lets us confirm
    that the WG dispatch counter bug triggers for *any* kernel, not just GEMM,
    when CU masking is applied."""
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, n_blocks, NUM_WGS):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, data, mask=mask)


def do_heavy_cu_copy(src_flat, dst_flat, n_elements, stream, n_iters, num_wgs):
    BLOCK_SIZE = 8192
    with torch.cuda.stream(stream):
        for _ in range(n_iters):
            heavy_copy_kernel[(num_wgs,)](
                src_flat, dst_flat, n_elements,
                BLOCK_SIZE=BLOCK_SIZE, NUM_WGS=num_wgs,
                num_warps=8, num_stages=1,
            )


def time_fn(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
        torch.distributed.barrier()
    times = []
    for _ in range(repeats):
        torch.distributed.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def do_dma_copy(src_ptr, dst_ptr, nbytes, stream, n_iters):
    if isinstance(stream, CUMaskedStreamWrapper):
        stream_handle = stream._hip_stream
    else:
        stream_handle = stream.cuda_stream
    for _ in range(n_iters):
        cp = hip.hipMemcpyAsync(
            dst_ptr, src_ptr, nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
            stream_handle,
        )
        HIP_CHECK(cp)


def do_cu_copy(src_flat, dst_flat, n_elements, stream, n_iters, max_grid=None):
    BLOCK_SIZE = 1024
    if max_grid is not None:
        num_wgs = max_grid
        with torch.cuda.stream(stream):
            for _ in range(n_iters):
                persistent_copy_kernel[(num_wgs,)](
                    src_flat, dst_flat, n_elements,
                    BLOCK_SIZE=BLOCK_SIZE, NUM_WGS=num_wgs,
                )
    else:
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        with torch.cuda.stream(stream):
            for _ in range(n_iters):
                copy_kernel[grid](src_flat, dst_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE)


def do_chunked_dma_copy(src_flat, dst_flat, n_elements, elem_size,
                        chunk_elems, stream, n_iters):
    """DMA copy in chunks matching tutorial 12 pattern: many small hipMemcpyAsync calls."""
    if isinstance(stream, CUMaskedStreamWrapper):
        stream_handle = stream._hip_stream
    else:
        stream_handle = stream.cuda_stream
    n_chunks = triton.cdiv(n_elements, chunk_elems)
    chunk_bytes = chunk_elems * elem_size
    for _ in range(n_iters):
        for c in range(n_chunks):
            offset = c * chunk_elems
            actual = min(chunk_elems, n_elements - offset)
            actual_bytes = actual * elem_size
            cp = hip.hipMemcpyAsync(
                dst_flat.data_ptr() + offset * elem_size,
                src_flat.data_ptr() + offset * elem_size,
                actual_bytes,
                hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                stream_handle,
            )
            HIP_CHECK(cp)


def do_chunked_cu_copy(src_flat, dst_flat, n_elements, chunk_elems,
                       stream, n_iters, max_grid=None):
    """CU-kernel copy in chunks matching tutorial 12 pattern: many small kernel launches."""
    BLOCK_SIZE = 1024
    n_chunks = triton.cdiv(n_elements, chunk_elems)
    with torch.cuda.stream(stream):
        for _ in range(n_iters):
            for c in range(n_chunks):
                offset = c * chunk_elems
                actual = min(chunk_elems, n_elements - offset)
                src_chunk = src_flat[offset: offset + actual]
                dst_chunk = dst_flat[offset: offset + actual]
                if max_grid is not None:
                    persistent_copy_kernel[(max_grid,)](
                        src_chunk, dst_chunk, actual,
                        BLOCK_SIZE=BLOCK_SIZE, NUM_WGS=max_grid,
                    )
                else:
                    grid = (triton.cdiv(actual, BLOCK_SIZE),)
                    copy_kernel[grid](src_chunk, dst_chunk, actual, BLOCK_SIZE=BLOCK_SIZE)


def run_benchmark(rank, num_ranks, workspace_tensors, size_mb,
                  warmup, repeats, copy_iters, device_info,
                  chunk_size_kb=None):
    total_cus = device_info.total_cus
    remote_rank = 1 - rank

    elem_size = workspace_tensors[rank].element_size()
    n_bytes = size_mb * 1024 * 1024
    n_elements = n_bytes // elem_size

    if n_elements > workspace_tensors[rank].numel():
        if rank == 0:
            max_mb = workspace_tensors[rank].numel() * elem_size // (1024 * 1024)
            print(f"  SKIP {size_mb}MB — exceeds workspace ({max_mb}MB)")
        return None

    src_flat = workspace_tensors[rank].reshape(-1)[:n_elements]
    dst_flat = workspace_tensors[remote_rank].reshape(-1)[:n_elements]

    regular_stream = torch.cuda.Stream()

    cus_03, _ = partition_cus(total_cus, strategy="interleaved", comm_ratio=0.3)
    cus_01, _ = partition_cus(total_cus, strategy="interleaved", comm_ratio=0.1)
    mask_03 = create_cu_mask(cus_03, total_cus)
    mask_01 = create_cu_mask(cus_01, total_cus)
    masked_stream_03 = CUMaskedStreamWrapper(create_stream_with_cu_mask(mask_03))
    masked_stream_01 = CUMaskedStreamWrapper(create_stream_with_cu_mask(mask_01))

    grid_70pct = int(total_cus * 0.7)  # 213
    grid_03 = len(cus_03)  # 91 — matches CU-masked 0.3 exactly
    grid_01 = len(cus_01)  # 30 — matches CU-masked 0.1 exactly
    total_mb = size_mb * copy_iters
    full_grid_wgs = triton.cdiv(n_elements, 1024)

    if chunk_size_kb is not None:
        chunk_elems = (chunk_size_kb * 1024) // elem_size
        n_chunks = triton.cdiv(n_elements, chunk_elems)
    else:
        chunk_elems = None
        n_chunks = 1

    modes = []
    if chunk_elems is not None:
        # Chunked modes matching tutorial 12 pattern
        modes = [
            (f"DMA  chunked | regular stream",
             lambda: do_chunked_dma_copy(src_flat, dst_flat, n_elements, elem_size,
                                         chunk_elems, regular_stream, copy_iters)),
            (f"DMA  chunked | CU-masked 0.3 (91 CUs)",
             lambda: do_chunked_dma_copy(src_flat, dst_flat, n_elements, elem_size,
                                         chunk_elems, masked_stream_03, copy_iters)),
            (f"DMA  chunked | CU-masked 0.1 (30 CUs)",
             lambda: do_chunked_dma_copy(src_flat, dst_flat, n_elements, elem_size,
                                         chunk_elems, masked_stream_01, copy_iters)),

            (f"CU-k chunked | regular, full grid",
             lambda: do_chunked_cu_copy(src_flat, dst_flat, n_elements, chunk_elems,
                                        regular_stream, copy_iters)),
            (f"CU-k chunked | CU-masked 0.3, full grid",
             lambda: do_chunked_cu_copy(src_flat, dst_flat, n_elements, chunk_elems,
                                        masked_stream_03, copy_iters)),
            (f"CU-k chunked | CU-masked 0.1, full grid",
             lambda: do_chunked_cu_copy(src_flat, dst_flat, n_elements, chunk_elems,
                                        masked_stream_01, copy_iters)),

            (f"CU-k chunked | regular, grid 70%",
             lambda g=grid_70pct: do_chunked_cu_copy(src_flat, dst_flat, n_elements,
                                                     chunk_elems, regular_stream,
                                                     copy_iters, max_grid=g)),
            (f"CU-k chunked | CU-masked 0.3, grid 70%",
             lambda g=grid_70pct: do_chunked_cu_copy(src_flat, dst_flat, n_elements,
                                                     chunk_elems, masked_stream_03,
                                                     copy_iters, max_grid=g)),
            (f"CU-k chunked | CU-masked 0.1, grid 70%",
             lambda g=grid_70pct: do_chunked_cu_copy(src_flat, dst_flat, n_elements,
                                                     chunk_elems, masked_stream_01,
                                                     copy_iters, max_grid=g)),
        ]
    else:
        modes = [
        # --- DMA modes ---
        ("DMA  | regular stream",
         lambda: do_dma_copy(src_flat.data_ptr(), dst_flat.data_ptr(), n_bytes,
                             regular_stream, copy_iters)),
        ("DMA  | CU-masked 0.3 (91 CUs)",
         lambda: do_dma_copy(src_flat.data_ptr(), dst_flat.data_ptr(), n_bytes,
                             masked_stream_03, copy_iters)),
        ("DMA  | CU-masked 0.1 (30 CUs)",
         lambda: do_dma_copy(src_flat.data_ptr(), dst_flat.data_ptr(), n_bytes,
                             masked_stream_01, copy_iters)),

        # --- CU-kernel full grid ---
        (f"CU-k | regular, full grid ({full_grid_wgs} WGs)",
         lambda: do_cu_copy(src_flat, dst_flat, n_elements,
                            regular_stream, copy_iters)),
        (f"CU-k | CU-masked 0.3, full grid",
         lambda: do_cu_copy(src_flat, dst_flat, n_elements,
                            masked_stream_03, copy_iters)),
        (f"CU-k | CU-masked 0.1, full grid",
         lambda: do_cu_copy(src_flat, dst_flat, n_elements,
                            masked_stream_01, copy_iters)),

        # --- CU-kernel grid=70% (persistent) ---
        (f"CU-k | regular, grid 70% ({grid_70pct} WGs)",
         lambda g=grid_70pct: do_cu_copy(src_flat, dst_flat, n_elements,
                                         regular_stream, copy_iters, max_grid=g)),
        (f"CU-k | CU-masked 0.3, grid 70%",
         lambda g=grid_70pct: do_cu_copy(src_flat, dst_flat, n_elements,
                                         masked_stream_03, copy_iters, max_grid=g)),
        (f"CU-k | CU-masked 0.1, grid 70%",
         lambda g=grid_70pct: do_cu_copy(src_flat, dst_flat, n_elements,
                                         masked_stream_01, copy_iters, max_grid=g)),

        # --- CU-kernel grid=CU mask count (persistent, fair comparison) ---
        (f"CU-k | regular, grid={grid_03} (0.3 CUs)",
         lambda g=grid_03: do_cu_copy(src_flat, dst_flat, n_elements,
                                       regular_stream, copy_iters, max_grid=g)),
        (f"CU-k | CU-masked 0.3, grid={grid_03}",
         lambda g=grid_03: do_cu_copy(src_flat, dst_flat, n_elements,
                                       masked_stream_03, copy_iters, max_grid=g)),
        (f"CU-k | regular, grid={grid_01} (0.1 CUs)",
         lambda g=grid_01: do_cu_copy(src_flat, dst_flat, n_elements,
                                       regular_stream, copy_iters, max_grid=g)),
        (f"CU-k | CU-masked 0.1, grid={grid_01}",
         lambda g=grid_01: do_cu_copy(src_flat, dst_flat, n_elements,
                                       masked_stream_01, copy_iters, max_grid=g)),

        # --- Heavy CU-kernel (high register pressure, ~1 WG/CU) ---
        # BLOCK_SIZE=8192, num_warps=8 → inflated register usage to trigger
        # WG dispatch counter bug with CU masking
        (f"HEAVY| regular, grid={total_cus} WGs",
         lambda g=total_cus: do_heavy_cu_copy(src_flat, dst_flat, n_elements,
                                              regular_stream, copy_iters, num_wgs=g)),
        (f"HEAVY| CU-masked 0.3, grid={total_cus} WGs",
         lambda g=total_cus: do_heavy_cu_copy(src_flat, dst_flat, n_elements,
                                              masked_stream_03, copy_iters, num_wgs=g)),
        (f"HEAVY| CU-masked 0.1, grid={total_cus} WGs",
         lambda g=total_cus: do_heavy_cu_copy(src_flat, dst_flat, n_elements,
                                              masked_stream_01, copy_iters, num_wgs=g)),
        (f"HEAVY| regular, grid={grid_70pct} WGs",
         lambda g=grid_70pct: do_heavy_cu_copy(src_flat, dst_flat, n_elements,
                                               regular_stream, copy_iters, num_wgs=g)),
        (f"HEAVY| CU-masked 0.3, grid={grid_70pct} WGs",
         lambda g=grid_70pct: do_heavy_cu_copy(src_flat, dst_flat, n_elements,
                                               masked_stream_03, copy_iters, num_wgs=g)),
        (f"HEAVY| CU-masked 0.1, grid={grid_70pct} WGs",
         lambda g=grid_70pct: do_heavy_cu_copy(src_flat, dst_flat, n_elements,
                                               masked_stream_01, copy_iters, num_wgs=g)),
    ]

    try:
        if rank == 0:
            chunk_desc = f", chunked={n_chunks}x{chunk_size_kb}KB" if chunk_size_kb else ""
            print(f"\n{'='*90}")
            print(f" {size_mb} MB x {copy_iters} iters = {total_mb} MB total p2p via XGMI{chunk_desc}")
            print(f"{'='*90}")
            print(f"{'Mode':<45s} {'Median':>9s} {'Mean':>9s} {'Std':>7s} {'BW':>9s}")
            print(f"{'':45s} {'(ms)':>9s} {'(ms)':>9s} {'(ms)':>7s} {'(GB/s)':>9s}")
            print(f"{'-'*90}")

        results = {}
        for label, fn in modes:
            times = time_fn(fn, warmup, repeats)
            med = statistics.median(times)
            mn = statistics.mean(times)
            sd = statistics.pstdev(times)
            bw = (total_mb / 1024) / (med / 1000)
            if rank == 0:
                print(f"  {label:<43s} {med:>9.3f} {mn:>9.3f} {sd:>7.3f} {bw:>9.1f}")
            results[label] = med

        if rank == 0:
            baseline = list(results.values())[0]
            print(f"\n  Speedups (vs DMA regular):")
            for label, med in results.items():
                print(f"    {label:<43s}: {baseline / med:.3f}x")

        return results
    finally:
        # Always tear down masked streams even if one mode errors.
        masked_stream_03.destroy()
        masked_stream_01.destroy()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU comm-only: DMA vs CU-kernel, with/without CU masking"
    )
    parser.add_argument("--size-mb", type=int, nargs="+", default=[32])
    parser.add_argument("--copy-iters", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--chunk-size-kb", type=int, default=None,
                        help="Chunk size in KB for chunked copy mode (e.g. 2048 for 2MB chunks "
                             "matching tutorial 12). If unset, copies are single large transfers.")
    parser.add_argument("--workspace-m", type=int, default=65536,
                        help="Workspace M dimension (controls max buffer size)")
    parser.add_argument("--workspace-k", type=int, default=4096,
                        help="Workspace K dimension")
    args = parser.parse_args()

    rank, _, num_ranks, tp_group = t09.init()
    pyrocshmem.init_rocshmem_by_uniqueid(tp_group)

    device_info = get_device_info()
    dtype = torch.float16

    workspace_tensors = pyrocshmem.rocshmem_create_tensor_list_intra_node(
        [args.workspace_m, args.workspace_k], dtype
    )
    torch.cuda.synchronize()
    torch.distributed.barrier()

    max_workspace_bytes = workspace_tensors[rank].numel() * workspace_tensors[rank].element_size()
    cus_03, _ = partition_cus(device_info.total_cus, strategy="interleaved", comm_ratio=0.3)
    cus_01, _ = partition_cus(device_info.total_cus, strategy="interleaved", comm_ratio=0.1)

    if rank == 0:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Total CUs: {device_info.total_cus}")
        print(f"CU-masked 0.3: {len(cus_03)} comm CUs ({100*len(cus_03)/device_info.total_cus:.0f}%)")
        print(f"CU-masked 0.1: {len(cus_01)} comm CUs ({100*len(cus_01)/device_info.total_cus:.0f}%)")
        print(f"Grid 70%: {int(device_info.total_cus * 0.7)} WGs (persistent copy)")
        print(f"Workspace: {args.workspace_m}x{args.workspace_k} fp16 = {max_workspace_bytes//(1024*1024)} MB")
        print(f"Copy iters: {args.copy_iters}, warmup: {args.warmup}, repeats: {args.repeats}")
        if args.chunk_size_kb:
            print(f"Chunk size: {args.chunk_size_kb} KB (chunked copy matching tutorial 12)")
        else:
            print(f"Chunk size: none (single contiguous copy)")
        print(f"Ranks: {num_ranks}")

    for size_mb in args.size_mb:
        run_benchmark(
            rank, num_ranks, workspace_tensors, size_mb,
            args.warmup, args.repeats, args.copy_iters, device_info,
            chunk_size_kb=args.chunk_size_kb,
        )

    if rank == 0:
        print("\nDone.")

    # Release rocSHMEM-backed tensors before finalize to avoid freeing
    # symmetric heap allocations during Python object teardown after finalize.
    del workspace_tensors
    gc.collect()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    pyrocshmem.rocshmem_finalize()
    t09.destroy()


if __name__ == "__main__":
    import os

    if os.environ.get("CU_MASK_SUITE_CALLER") != "1":
        print(
            "[DEPRECATED] Direct execution is deprecated. "
            "Use tutorials/cu_mask_suite.py comm-only ... or scripts/cu_experiments.py."
        )
    main()
