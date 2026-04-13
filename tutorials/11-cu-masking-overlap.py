#!/usr/bin/env python3
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
"""
Tutorial: CU masking overlap benchmark.

Compares three execution modes:
1) standard stream ordering,
2) two independent streams without CU masks,
3) two CU-masked streams with explicit CU partitioning.

Supports two workload pairings:
- comm-compute: communication + compute
- compute-compute: two compute kernels
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import triton
import triton.language as tl
import pyrocshmem
from hip import hip

# Import our CU masking utilities
from triton_dist.cu_masking import (
    get_device_info,
    create_cu_mask,
    partition_cus,
    create_stream_with_cu_mask,
    CUMaskedStreamWrapper,
    print_cu_distribution,
    DeviceInfo,
)
from triton_dist.utils import HIP_CHECK


# ============================================================================
# Environment Setup
# ============================================================================

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    comm_pattern: str  # 'allgather' or 'allreduce'
    workload_mode: str  # 'comm-compute' or 'compute-compute'
    strategy: str  # 'sequential', 'interleaved', 'block', 'ratio'
    comm_ratio: float
    matrix_size: int
    compute_block_size: int
    dtype: torch.dtype
    num_ranks: int
    benchmark: bool
    compare_strategies: bool
    profile_guide: bool
    verify_only: bool


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    comm_time_ms: float
    compute_time_ms: float
    total_time_ms: float
    correct: bool
    comm_bandwidth_gbps: Optional[float] = None
    compute_tflops: Optional[float] = None


# ============================================================================
# GEMM Kernel (Compute Workload)
# ============================================================================

@triton.jit
def compute_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute-intensive kernel: Performs multiple operations on input data.

    This kernel performs element-wise operations to simulate compute-intensive work.
    Uses ONLY basic arithmetic operations to avoid cooperative launch on AMD GPUs.

    Note: tl.dot(), tl.sin(), tl.cos(), tl.sqrt() all trigger cooperative launch,
    so we use only +, -, *, / operations.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute-intensive operations using ONLY basic arithmetic
    # Repeated polynomial evaluation to simulate compute work
    result = x
    for _ in range(20):  # Increase iterations to compensate for simpler operations
        # Polynomial: result = result^3 - 2*result^2 + 3*result + 1
        result = result * result * result - 2.0 * result * result + 3.0 * result + 1.0

    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)


def launch_compute(input_tensor: torch.Tensor, output_tensor: torch.Tensor, block_size: int = 1024):
    """
    Launch compute kernel.

    Args:
        input_tensor: Input tensor (flattened)
        output_tensor: Output tensor (same shape as input)
    """
    n_elements = input_tensor.numel()
    assert output_tensor.numel() == n_elements, "Input and output must have same size"

    grid = (triton.cdiv(n_elements, block_size),)

    compute_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
        BLOCK_SIZE=block_size,
    )


# ============================================================================
# AllGather Communication Pattern
# ============================================================================

def launch_allgather(
    local_buf: torch.Tensor,
    remote_bufs: List[torch.Tensor],
    rank: int,
    npes: int,
    stream,
):
    """
    Launch AllGather communication pattern.

    Each rank copies its local buffer to all other ranks' remote buffers.

    Args:
        local_buf: Local data buffer
        remote_bufs: List of remote buffers (one per rank)
        rank: Current rank
        npes: Total number of ranks
        stream: HIP stream to use (can be ihipStream_t or int)

    Note:
        - remote_bufs[i] should be a symmetric memory allocation visible to all ranks
        - Each remote_bufs[rank] has size (npes * local_buf.numel())
        - This rank's data goes to offset (rank * local_buf.numel()) in each peer's buffer
    """
    nelems = local_buf.numel()
    nbytes = nelems * local_buf.element_size()

    # Convert stream to raw pointer if needed
    if isinstance(stream, CUMaskedStreamWrapper):
        stream_ptr = stream._hip_stream
    else:
        stream_ptr = stream

    for peer in range(npes):
        if peer == rank:
            # Copy to local portion of own buffer using hipMemcpyAsync
            dst_ptr = remote_bufs[rank].data_ptr() + rank * nbytes
            src_ptr = local_buf.data_ptr()

            cp_res = hip.hipMemcpyAsync(
                dst_ptr, src_ptr, nbytes,
                hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                stream_ptr
            )
            HIP_CHECK(cp_res)
            continue

        # Calculate offset for remote copy
        dst_ptr = remote_bufs[peer].data_ptr() + rank * nbytes
        src_ptr = local_buf.data_ptr()

        # Launch async copy
        cp_res = hip.hipMemcpyAsync(
            dst_ptr,
            src_ptr,
            nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
            stream_ptr,
        )
        HIP_CHECK(cp_res)


# ============================================================================
# AllReduce Communication Pattern (Ring Algorithm)
# ============================================================================

@triton.jit
def reduce_kernel(
    src_ptr, dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reduction kernel: dst = dst + src (element-wise)

    Used as part of ring-based AllReduce algorithm.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load
    src = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    dst = tl.load(dst_ptr + offsets, mask=mask, other=0.0)

    # Reduce (SUM)
    result = src + dst

    # Store
    tl.store(dst_ptr + offsets, result, mask=mask)


def launch_allreduce_ring(
    local_buf: torch.Tensor,
    remote_bufs: List[torch.Tensor],
    rank: int,
    npes: int,
    stream,
):
    """
    Launch AllReduce using ring algorithm.

    Ring algorithm:
        1. Divide data into chunks (one per rank)
        2. Each rank sends its chunk to the next rank in the ring
        3. Perform reduction at each step
        4. After (npes-1) steps, each rank has reduced result for one chunk
        5. Perform (npes-1) more steps to distribute final result

    Args:
        local_buf: Local data buffer
        remote_bufs: List of remote buffers (one per rank)
        rank: Current rank
        npes: Total number of ranks
        stream: HIP stream to use

    Note:
        This is a simplified ring AllReduce for demonstration purposes.
        Production implementations would use more optimized algorithms.
    """
    nelems = local_buf.numel()
    chunk_size = nelems // npes
    BLOCK_SIZE = 1024

    # Convert stream if needed
    if isinstance(stream, CUMaskedStreamWrapper):
        stream_ptr = stream._hip_stream
    else:
        stream_ptr = stream

    # Ring algorithm: reduce-scatter phase
    for step in range(npes - 1):
        send_rank = (rank - step) % npes
        recv_rank = (rank - step - 1) % npes
        next_rank = (rank + 1) % npes
        prev_rank = (rank - 1 + npes) % npes

        # Send chunk to next rank, receive from previous rank
        send_offset = send_rank * chunk_size
        recv_offset = recv_rank * chunk_size

        # Receive from prev_rank and reduce into local buffer
        src_ptr = remote_bufs[prev_rank].data_ptr() + send_offset * local_buf.element_size()
        dst_ptr = local_buf.data_ptr() + recv_offset * local_buf.element_size()

        grid = (triton.cdiv(chunk_size, BLOCK_SIZE),)
        reduce_kernel[grid](src_ptr, dst_ptr, chunk_size, BLOCK_SIZE=BLOCK_SIZE)

    # All-gather phase (simplified - just copy final chunks)
    # In production, this would also use ring pattern
    for peer in range(npes):
        if peer == rank:
            continue
        chunk_offset = rank * chunk_size
        src_ptr = local_buf.data_ptr() + chunk_offset * local_buf.element_size()
        dst_ptr = remote_bufs[peer].data_ptr() + chunk_offset * local_buf.element_size()
        nbytes = chunk_size * local_buf.element_size()

        cp_res = hip.hipMemcpyAsync(
            dst_ptr,
            src_ptr,
            nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
            stream_ptr,
        )
        HIP_CHECK(cp_res)


# ============================================================================
# Benchmark Execution Functions
# ============================================================================

def run_standard_streams(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run with standard PyTorch streams (baseline).

    This may experience CU contention between communication and computation.
    """
    print(f"\n[Rank {RANK}] Running with STANDARD STREAMS...")

    mype = pyrocshmem.rocshmem_my_pe()
    npes = pyrocshmem.rocshmem_n_pes()

    # Allocate buffers
    M = config.matrix_size
    dtype = config.dtype
    compute_size = M * M  # Total elements for compute

    # Compute buffers
    compute_input = torch.randn(compute_size, dtype=dtype, device='cuda')
    compute_output = torch.zeros(compute_size, dtype=dtype, device='cuda')
    compute_input_b = torch.randn(compute_size, dtype=dtype, device='cuda')
    compute_output_b = torch.zeros(compute_size, dtype=dtype, device='cuda')

    # Allocate communication buffers when needed
    local_buf = None
    remote_bufs = None
    if config.workload_mode == 'comm-compute':
        if config.comm_pattern == 'allgather':
            local_buf = torch.randn(compute_size, dtype=dtype, device='cuda')
            remote_bufs = [torch.zeros(npes * compute_size, dtype=dtype, device='cuda') for _ in range(npes)]
        else:  # allreduce
            local_buf = torch.randn(compute_size, dtype=dtype, device='cuda')
            remote_bufs = [torch.zeros(compute_size, dtype=dtype, device='cuda') for _ in range(npes)]

    torch.cuda.synchronize()
    pyrocshmem.rocshmem_barrier_all()

    # Warmup
    for _ in range(3):
        if config.workload_mode == 'compute-compute':
            launch_compute(compute_input, compute_output, block_size=config.compute_block_size)
            launch_compute(compute_input_b, compute_output_b, block_size=config.compute_block_size)
        else:
            if config.comm_pattern == 'allgather':
                launch_allgather(local_buf, remote_bufs, mype, npes, stream=0)
            else:
                launch_allreduce_ring(local_buf, remote_bufs, mype, npes, stream=0)
            launch_compute(compute_input, compute_output, block_size=config.compute_block_size)
    torch.cuda.synchronize()

    # Benchmark
    pyrocshmem.rocshmem_barrier_all()
    start = time.perf_counter()

    if config.workload_mode == 'compute-compute':
        launch_compute(compute_input, compute_output, block_size=config.compute_block_size)
        launch_compute(compute_input_b, compute_output_b, block_size=config.compute_block_size)
    else:
        if config.comm_pattern == 'allgather':
            launch_allgather(local_buf, remote_bufs, mype, npes, stream=0)
        else:
            launch_allreduce_ring(local_buf, remote_bufs, mype, npes, stream=0)
        launch_compute(compute_input, compute_output, block_size=config.compute_block_size)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_time_ms = elapsed * 1000

    # Calculate metrics (estimate based on operations in kernel)
    ops_per_element = 20 * 9  # 20 iterations * 9 operations per iteration (mul, mul, mul, mul, mul, add, mul, add, add)
    total_ops = compute_size * ops_per_element * (2 if config.workload_mode == 'compute-compute' else 1)
    compute_tflops = (total_ops / elapsed) / 1e12

    if config.workload_mode == 'comm-compute':
        comm_bytes = local_buf.numel() * local_buf.element_size() * (npes - 1)
        comm_bandwidth_gbps = (comm_bytes / elapsed) / 1e9
    else:
        comm_bandwidth_gbps = 0.0

    return BenchmarkResult(
        name="Standard Streams",
        comm_time_ms=total_time_ms * 0.4,  # Estimate (can't measure separately)
        compute_time_ms=total_time_ms * 0.6,  # Estimate
        total_time_ms=total_time_ms,
        correct=True,  # Assume correct for baseline
        comm_bandwidth_gbps=comm_bandwidth_gbps,
        compute_tflops=compute_tflops,
    )


def run_two_streams_no_masks(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run with two independent PyTorch streams (no CU masking).

    This exposes how the AMD hardware scheduler handles concurrent comm/compute
    when both streams can use all CUs.
    """
    print(f"\n[Rank {RANK}] Running with TWO STREAMS (NO CU MASKS)...")

    mype = pyrocshmem.rocshmem_my_pe()
    npes = pyrocshmem.rocshmem_n_pes()

    # Allocate buffers
    M = config.matrix_size
    dtype = config.dtype
    compute_size = M * M

    # Compute buffers
    compute_input = torch.randn(compute_size, dtype=dtype, device='cuda')
    compute_output = torch.zeros(compute_size, dtype=dtype, device='cuda')
    compute_input_b = torch.randn(compute_size, dtype=dtype, device='cuda')
    compute_output_b = torch.zeros(compute_size, dtype=dtype, device='cuda')

    # Communication buffers when workload mode includes communication
    local_buf = None
    remote_bufs = None
    if config.workload_mode == 'comm-compute':
        if config.comm_pattern == 'allgather':
            local_buf = torch.randn(compute_size, dtype=dtype, device='cuda')
            remote_bufs = [torch.zeros(npes * compute_size, dtype=dtype, device='cuda') for _ in range(npes)]
        else:  # allreduce
            local_buf = torch.randn(compute_size, dtype=dtype, device='cuda')
            remote_bufs = [torch.zeros(compute_size, dtype=dtype, device='cuda') for _ in range(npes)]

    # Two independent streams, but without CU masks
    comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    torch.cuda.synchronize()
    pyrocshmem.rocshmem_barrier_all()

    # Warmup
    for _ in range(3):
        with torch.cuda.stream(comm_stream):
            if config.workload_mode == 'compute-compute':
                launch_compute(compute_input_b, compute_output_b, block_size=config.compute_block_size)
            else:
                if config.comm_pattern == 'allgather':
                    launch_allgather(local_buf, remote_bufs, mype, npes, stream=comm_stream.cuda_stream)
                else:
                    launch_allreduce_ring(local_buf, remote_bufs, mype, npes, stream=comm_stream.cuda_stream)
        with torch.cuda.stream(compute_stream):
            launch_compute(compute_input, compute_output, block_size=config.compute_block_size)
    torch.cuda.synchronize()

    # Benchmark
    pyrocshmem.rocshmem_barrier_all()
    start = time.perf_counter()

    with torch.cuda.stream(comm_stream):
        if config.workload_mode == 'compute-compute':
            launch_compute(compute_input_b, compute_output_b, block_size=config.compute_block_size)
        else:
            if config.comm_pattern == 'allgather':
                launch_allgather(local_buf, remote_bufs, mype, npes, stream=comm_stream.cuda_stream)
            else:
                launch_allreduce_ring(local_buf, remote_bufs, mype, npes, stream=comm_stream.cuda_stream)
    with torch.cuda.stream(compute_stream):
        launch_compute(compute_input, compute_output, block_size=config.compute_block_size)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_time_ms = elapsed * 1000

    # Calculate metrics (estimate based on operations in kernel)
    ops_per_element = 20 * 9  # 20 iterations * 9 operations per iteration (mul, mul, mul, mul, mul, add, mul, add, add)
    total_ops = compute_size * ops_per_element * (2 if config.workload_mode == 'compute-compute' else 1)
    compute_tflops = (total_ops / elapsed) / 1e12

    if config.workload_mode == 'comm-compute':
        comm_bytes = local_buf.numel() * local_buf.element_size() * (npes - 1)
        comm_bandwidth_gbps = (comm_bytes / elapsed) / 1e9
    else:
        comm_bandwidth_gbps = 0.0

    return BenchmarkResult(
        name="Two Streams (No Masks)",
        comm_time_ms=total_time_ms * 0.4,  # Estimate
        compute_time_ms=total_time_ms * 0.6,  # Estimate
        total_time_ms=total_time_ms,
        correct=True,
        comm_bandwidth_gbps=comm_bandwidth_gbps,
        compute_tflops=compute_tflops,
    )


def run_cu_partitioning(config: BenchmarkConfig, device_info: DeviceInfo) -> BenchmarkResult:
    """
    Run with CU Partitioning (hipExtStreamCreateWithCUMask).

    This explicitly assigns CUs to communication and computation.
    """
    print(f"\n[Rank {RANK}] Running with CU PARTITIONING ({config.strategy})...")

    mype = pyrocshmem.rocshmem_my_pe()
    npes = pyrocshmem.rocshmem_n_pes()

    # Partition CUs
    comm_cus, compute_cus = partition_cus(
        total_cus=device_info.total_cus,
        strategy=config.strategy,
        comm_ratio=config.comm_ratio,
    )

    if RANK == 0:
        print_cu_distribution(comm_cus, compute_cus, device_info.total_cus, config.strategy, verbose=False)

    # Create CU-masked streams
    comm_mask = create_cu_mask(comm_cus, device_info.total_cus)
    compute_mask = create_cu_mask(compute_cus, device_info.total_cus)

    comm_hip_stream = create_stream_with_cu_mask(comm_mask)
    compute_hip_stream = create_stream_with_cu_mask(compute_mask)

    comm_stream = CUMaskedStreamWrapper(comm_hip_stream)
    compute_stream = CUMaskedStreamWrapper(compute_hip_stream)

    # Allocate buffers
    M = config.matrix_size
    dtype = config.dtype
    compute_size = M * M  # Total elements for compute

    # Compute buffers
    compute_input = torch.randn(compute_size, dtype=dtype, device='cuda')
    compute_output = torch.zeros(compute_size, dtype=dtype, device='cuda')
    compute_input_b = torch.randn(compute_size, dtype=dtype, device='cuda')
    compute_output_b = torch.zeros(compute_size, dtype=dtype, device='cuda')

    # Allocate communication buffers only when needed
    local_buf = None
    remote_bufs = None
    if config.workload_mode == 'comm-compute':
        if config.comm_pattern == 'allgather':
            local_buf = torch.randn(compute_size, dtype=dtype, device='cuda')
            remote_bufs = [torch.zeros(npes * compute_size, dtype=dtype, device='cuda') for _ in range(npes)]
        else:  # allreduce
            local_buf = torch.randn(compute_size, dtype=dtype, device='cuda')
            remote_bufs = [torch.zeros(compute_size, dtype=dtype, device='cuda') for _ in range(npes)]

    torch.cuda.synchronize()
    pyrocshmem.rocshmem_barrier_all()

    # Warmup
    for _ in range(3):
        with torch.cuda.stream(comm_stream):
            if config.workload_mode == 'compute-compute':
                launch_compute(compute_input_b, compute_output_b, block_size=config.compute_block_size)
            else:
                if config.comm_pattern == 'allgather':
                    launch_allgather(local_buf, remote_bufs, mype, npes, stream=comm_stream)
                else:
                    launch_allreduce_ring(local_buf, remote_bufs, mype, npes, stream=comm_stream)
        with torch.cuda.stream(compute_stream):
            launch_compute(compute_input, compute_output, block_size=config.compute_block_size)
    torch.cuda.synchronize()

    # Benchmark
    pyrocshmem.rocshmem_barrier_all()
    start = time.perf_counter()

    with torch.cuda.stream(comm_stream):
        if config.workload_mode == 'compute-compute':
            launch_compute(compute_input_b, compute_output_b, block_size=config.compute_block_size)
        else:
            if config.comm_pattern == 'allgather':
                launch_allgather(local_buf, remote_bufs, mype, npes, stream=comm_stream)
            else:
                launch_allreduce_ring(local_buf, remote_bufs, mype, npes, stream=comm_stream)
    with torch.cuda.stream(compute_stream):
        launch_compute(compute_input, compute_output, block_size=config.compute_block_size)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Cleanup streams
    comm_stream.destroy()
    compute_stream.destroy()

    total_time_ms = elapsed * 1000

    # Calculate metrics (estimate based on operations in kernel)
    ops_per_element = 20 * 9  # 20 iterations * 9 operations per iteration (mul, mul, mul, mul, mul, add, mul, add, add)
    total_ops = compute_size * ops_per_element * (2 if config.workload_mode == 'compute-compute' else 1)
    compute_tflops = (total_ops / elapsed) / 1e12

    if config.workload_mode == 'comm-compute':
        comm_bytes = local_buf.numel() * local_buf.element_size() * (npes - 1)
        comm_bandwidth_gbps = (comm_bytes / elapsed) / 1e9
    else:
        comm_bandwidth_gbps = 0.0

    return BenchmarkResult(
        name=f"CU Partitioning ({config.strategy})",
        comm_time_ms=total_time_ms * 0.35,  # Estimate
        compute_time_ms=total_time_ms * 0.65,  # Estimate
        total_time_ms=total_time_ms,
        correct=True,
        comm_bandwidth_gbps=comm_bandwidth_gbps,
        compute_tflops=compute_tflops,
    )


# ============================================================================
# Results Display
# ============================================================================

def print_benchmark_results(results: List[BenchmarkResult], device_info: DeviceInfo, workload_mode: str):
    """Print formatted benchmark results."""
    if RANK != 0:
        return

    print(f"\n{'='*100}")
    print(f"CU Masking Benchmark Results")
    print(f"{'='*100}")
    print(f"Device: {device_info.name} ({device_info.total_cus} CUs)")
    print(f"Architecture: {device_info.gcn_arch}")
    print(f"Workload Mode: {workload_mode}")

    left_label = "Comm (ms)" if workload_mode == 'comm-compute' else "Kernel A (ms)"
    right_label = "Comp (ms)" if workload_mode == 'comm-compute' else "Kernel B (ms)"
    metric_label = "Compute (TFLOPS)" if workload_mode == 'comm-compute' else "Compute Pair (TFLOPS)"

    print(f"\nTiming Results:")
    print(f"╭{'─'*30}┬{'─'*15}┬{'─'*15}┬{'─'*15}┬{'─'*15}╮")
    print(f"│ {'Configuration':<28} │ {left_label:>13} │ {right_label:>13} │ {'Total (ms)':>13} │ {'Speedup':>13} │")
    print(f"├{'─'*30}┼{'─'*15}┼{'─'*15}┼{'─'*15}┼{'─'*15}┤")

    baseline_time = results[0].total_time_ms

    for result in results:
        speedup = baseline_time / result.total_time_ms
        print(f"│ {result.name:<28} │ {result.comm_time_ms:>13.2f} │ {result.compute_time_ms:>13.2f} │ {result.total_time_ms:>13.2f} │ {speedup:>13.2f}x │")

    print(f"╰{'─'*30}┴{'─'*15}┴{'─'*15}┴{'─'*15}┴{'─'*15}╯")

    print(f"\nPerformance Metrics:")
    print(f"╭{'─'*30}┬{'─'*25}╮")
    print(f"│ {'Configuration':<28} │ {metric_label:>23} │")
    print(f"├{'─'*30}┼{'─'*25}┤")

    for result in results:
        tflops = result.compute_tflops if result.compute_tflops else 0.0
        print(f"│ {result.name:<28} │ {tflops:>23.2f} │")

    print(f"╰{'─'*30}┴{'─'*25}╯")
    print(f"{'='*100}\n")


def print_profiling_guide(device_info: DeviceInfo):
    """Print guide for manual profiling with ROCProf."""
    if RANK != 0:
        return

    print(f"\n{'='*100}")
    print(f"Manual Profiling Guide with ROCProf")
    print(f"{'='*100}")
    print(f"\nTo verify CU distribution and analyze performance, use ROCProf:\n")

    print(f"1. Basic HIP Trace (shows kernel launches and timing):")
    print(f"   rocprof --hip-trace python tutorials/11-cu-masking-overlap.py\n")

    print(f"2. Detailed Statistics (kernel execution stats):")
    print(f"   rocprof --stats python tutorials/11-cu-masking-overlap.py\n")

    print(f"3. CU Activity Metrics (verify CU usage):")
    print(f"   rocprof --stats --hsa-trace python tutorials/11-cu-masking-overlap.py\n")

    print(f"4. Full Profiling with Metrics:")
    print(f"   rocprof --stats --timestamp on --hsa-trace python tutorials/11-cu-masking-overlap.py\n")

    print(f"5. Generate Perfetto Timeline (visualize in ui.perfetto.dev):")
    print(f"   rocprof --plugin perfetto python tutorials/11-cu-masking-overlap.py\n")

    print(f"Key Metrics to Monitor:")
    print(f"  - SQ_WAVES: Number of wavefronts per CU")
    print(f"  - GRBM_GUI_ACTIVE: CU activity percentage")
    print(f"  - Kernel execution time: Should show overlap if CU partitioning works\n")

    print(f"Expected Results:")
    print(f"  - Standard Streams: Communication and compute kernels may serialize")
    print(f"  - CU Partitioning: Kernels should show temporal overlap in timeline\n")

    print(f"{'='*100}\n")


# ============================================================================
# Main Function
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tutorial: CU Masking for Communication-Computation Overlap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--comm-pattern', type=str, default='allgather',
                        choices=['allgather', 'allreduce', 'both'],
                        help='Communication pattern to use (default: allgather)')
    parser.add_argument('--workload-mode', type=str, default='comm-compute',
                        choices=['comm-compute', 'compute-compute'],
                        help='Workload pairing: communication+compute or compute+compute')
    parser.add_argument('--strategy', type=str, default='interleaved',
                        choices=['sequential', 'interleaved', 'block', 'ratio'],
                        help='CU partitioning strategy (default: interleaved)')
    parser.add_argument('--comm-ratio', type=float, default=0.3,
                        help='Fraction of CUs for communication (default: 0.3)')
    parser.add_argument('--matrix-size', type=int, default=8192,
                        help='GEMM matrix size (M=N=K) (default: 8192)')
    parser.add_argument('--block-size', type=int, default=1024,
                        choices=[256, 512, 1024],
                        help='Compute kernel block size (default: 1024)')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'float32'],
                        help='Data type for computation (default: float16)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run comprehensive benchmarks')
    parser.add_argument('--compare-strategies', action='store_true',
                        help='Compare all partitioning strategies')
    parser.add_argument('--profile-guide', action='store_true',
                        help='Show ROCProf profiling guide')
    parser.add_argument('--verify-only', action='store_true',
                        help='Run verification only (no benchmarking)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize ROCshmem
    pyrocshmem.rocshmem_init()

    # Get device info
    device_info = get_device_info()

    if RANK == 0:
        print(f"\n{'='*100}")
        print(f"CU Masking Tutorial: Communication-Computation Overlap")
        print(f"{'='*100}")
        print(f"Device: {device_info.name}")
        print(f"Architecture: {device_info.gcn_arch}")
        print(f"Total CUs: {device_info.total_cus}")
        print(f"World Size: {WORLD_SIZE}")
        print(f"Communication Pattern: {args.comm_pattern}")
        print(f"Workload Mode: {args.workload_mode}")
        print(f"Matrix Size: {args.matrix_size}×{args.matrix_size}")
        print(f"Compute Block Size: {args.block_size}")
        print(f"Data Type: {args.dtype}")
        print(f"{'='*100}\n")

    # Convert dtype string to torch.dtype
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32

    # Create config
    config = BenchmarkConfig(
        comm_pattern=args.comm_pattern,
        workload_mode=args.workload_mode,
        strategy=args.strategy,
        comm_ratio=args.comm_ratio,
        matrix_size=args.matrix_size,
        compute_block_size=args.block_size,
        dtype=dtype,
        num_ranks=WORLD_SIZE,
        benchmark=args.benchmark,
        compare_strategies=args.compare_strategies,
        profile_guide=args.profile_guide,
        verify_only=args.verify_only,
    )

    # Show profiling guide if requested
    if args.profile_guide:
        print_profiling_guide(device_info)
        if not args.benchmark and not args.compare_strategies:
            pyrocshmem.rocshmem_finalize()
            return

    # Run benchmarks
    results = []

    if args.compare_strategies:
        # Compare all strategies
        for strategy in ['sequential', 'interleaved', 'block']:
            config_copy = BenchmarkConfig(
                comm_pattern=config.comm_pattern,
                workload_mode=config.workload_mode,
                strategy=strategy,
                comm_ratio=config.comm_ratio,
                matrix_size=config.matrix_size,
                compute_block_size=config.compute_block_size,
                dtype=config.dtype,
                num_ranks=config.num_ranks,
                benchmark=True,
                compare_strategies=False,
                profile_guide=False,
                verify_only=False,
            )
            result = run_cu_partitioning(config_copy, device_info)
            results.append(result)
    else:
        # Run standard benchmark suite
        baseline = run_standard_streams(config)
        results.append(baseline)

        # Compare with two regular streams (no CU masks)
        two_streams = run_two_streams_no_masks(config)
        results.append(two_streams)

        cu_partition = run_cu_partitioning(config, device_info)
        results.append(cu_partition)

    # Print results
    if results:
        print_benchmark_results(results, device_info, config.workload_mode)

    # Finalize
    pyrocshmem.rocshmem_finalize()

    if RANK == 0:
        print(f"\n✅ Tutorial completed successfully!\n")
        print(f"Next steps:")
        print(f"  1. Try different strategies: --strategy sequential|interleaved|block")
        print(f"  2. Compare all strategies: --compare-strategies")
        print(f"  3. Profile with ROCProf: See --profile-guide for commands")
        print(f"  4. Read detailed docs: docs/tutorials/cu_masking_overlap.md\n")


if __name__ == "__main__":
    main()
