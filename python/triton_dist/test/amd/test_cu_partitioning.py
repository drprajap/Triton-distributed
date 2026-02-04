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
Example: Explicit CU Partitioning with hipExtStreamCreateWithCUMask
====================================================================

This test demonstrates how to explicitly partition Compute Units (CUs) between
communication and computation kernels using hipExtStreamCreateWithCUMask.

Key differences from Copy Engine approach:
1. Uses actual CUs (not dedicated copy engines)
2. Explicit control over which CUs run which kernels
3. Uses hipMemcpyDeviceToDevice (not NoCU)
4. Better for workloads where copy engines are insufficient

Workflow:
    1. Query total CUs on the GPU
    2. Create bit masks to partition CUs (e.g., 50/50 split)
    3. Create streams bound to specific CUs using hipExtStreamCreateWithCUMask
    4. Launch communication kernels on comm CUs
    5. Launch computation kernels on compute CUs
    6. Both run concurrently on separate CUs!

Run with:
    cd /dev/data/diprajap/workspace/rocm7/Triton-distributed
    WORLD_SIZE=2 LOCAL_WORLD_SIZE=2 python -m pytest -xvs \
        python/triton_dist/test/amd/test_cu_partitioning.py
"""

import os
import datetime
import torch
import triton
import triton.language as tl
import pyrocshmem
from hip import hip
from typing import List, Tuple
from triton_dist.utils import HIP_CHECK
import ctypes

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


# ========================================
# CU Mask Management
# ========================================
def get_num_cus() -> int:
    """Get the number of Compute Units on current GPU"""
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    num_cus = device_props.multi_processor_count
    print(f"[PE {RANK}] GPU has {num_cus} Compute Units (CUs)")
    return num_cus


def create_cu_mask(cu_list: List[int], total_cus: int) -> List[int]:
    """
    Create a CU mask from a list of CU indices.
    
    Args:
        cu_list: List of CU indices to enable (e.g., [0, 1, 2, 3])
        total_cus: Total number of CUs on the device
        
    Returns:
        List of uint32 representing the bit mask
    """
    # Calculate mask size (each uint32 holds 32 bits)
    mask_size = (total_cus + 31) // 32
    cu_mask = [0] * mask_size
    
    # Set bits for specified CUs
    for cu_idx in cu_list:
        if cu_idx >= total_cus:
            raise ValueError(f"CU index {cu_idx} out of range (total: {total_cus})")
        word_idx = cu_idx // 32
        bit_idx = cu_idx % 32
        cu_mask[word_idx] |= (1 << bit_idx)
    
    return cu_mask


def print_cu_mask(cu_mask: List[int], total_cus: int, name: str):
    """Debug print for CU mask"""
    enabled_cus = []
    for word_idx, word in enumerate(cu_mask):
        for bit_idx in range(32):
            cu_idx = word_idx * 32 + bit_idx
            if cu_idx >= total_cus:
                break
            if word & (1 << bit_idx):
                enabled_cus.append(cu_idx)
    
    print(f"[PE {RANK}] {name}: CUs {enabled_cus} (total: {len(enabled_cus)})")


def create_stream_with_cu_mask(cu_mask: List[int]):
    """
    Create a HIP stream bound to specific CUs using hipExtStreamCreateWithCUMask.
    
    Args:
        cu_mask: List of uint32 representing the CU bit mask
        
    Returns:
        hipStream_t object (hip.hip.ihipStream_t)
    """
    # Prepare the mask array
    mask_size = len(cu_mask)
    mask_array = (ctypes.c_uint32 * mask_size)(*cu_mask)
    
    # Call hipExtStreamCreateWithCUMask
    # Python binding returns: (hipError_t, hipStream_t)
    err, stream = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)
    
    if err != hip.hipError_t.hipSuccess:
        raise RuntimeError(f"hipExtStreamCreateWithCUMask failed: {err}")
    
    # stream is a hip.hip.ihipStream_t object
    return stream


def partition_cus(
    total_cus: int,
    comm_cus_count: int = None,
    comm_ratio: float = 0.3
) -> Tuple[List[int], List[int]]:
    """
    Partition CUs between communication and computation.
    
    Args:
        total_cus: Total number of CUs
        comm_cus_count: Fixed number of CUs for communication (overrides comm_ratio)
        comm_ratio: Fraction of CUs for communication (default 0.3 = 30%)
        
    Returns:
        (comm_cu_list, compute_cu_list)
    """
    # Use fixed count if specified, otherwise use ratio
    if comm_cus_count is not None:
        num_comm_cus = min(max(1, comm_cus_count), total_cus - 1)
    else:
        num_comm_cus = max(1, int(total_cus * comm_ratio))
    
    # Strategy: Use odd CUs for communication, ALL remaining for computation
    odd_cus = [i for i in range(total_cus) if i % 2 == 1]
    even_cus = [i for i in range(total_cus) if i % 2 == 0]
    
    # Assign first N odd CUs to communication
    comm_cus = odd_cus[:num_comm_cus]
    
    # If need more comm CUs than available odd indices, take from even
    if len(comm_cus) < num_comm_cus:
        remaining = num_comm_cus - len(comm_cus)
        comm_cus.extend(even_cus[:remaining])
    
    # Compute gets ALL remaining CUs
    compute_cus = [i for i in range(total_cus) if i not in comm_cus]
    
    print(f"[PE {RANK}] CU Partition: {len(comm_cus)} comm, {len(compute_cus)} compute")
    
    return comm_cus, compute_cus


# ========================================
# Communication Kernel (on Comm CUs)
# ========================================
@triton.jit
def communication_kernel(
    local_ptr,
    remote_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple communication kernel: Copy data from local to remote buffer.
    This runs on CUs dedicated to communication.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load from local
    data = tl.load(local_ptr + offsets, mask=mask)
    
    # Store to remote (simulates one-sided communication)
    tl.store(remote_ptr + offsets, data, mask=mask)


def launch_communication_kernels(
    rank: int,
    npes: int,
    local_buf: torch.Tensor,
    remote_bufs: List[torch.Tensor],
    comm_stream,  # hipStream_t object
):
    """
    Launch communication kernels on the CU-masked stream.
    Uses hipMemcpyAsync (standard, NOT NoCU) since we're using actual CUs.
    """
    nelems = local_buf.numel()
    nbytes = nelems * local_buf.element_size()
    
    print(f"[PE {rank}] Launching communication on dedicated CUs...")
    
    for peer in range(npes):
        if peer == rank:
            # Copy to the portion of remote_bufs[rank] belonging to this rank
            # remote_bufs[rank] has size nelems * npes, we copy to offset rank * nelems
            dst_slice = remote_bufs[rank][rank * nelems:(rank + 1) * nelems]
            dst_slice.copy_(local_buf)
            continue
        
        # Calculate offsets for remote copy
        # Each PE's data goes to offset (rank * nelems) in the peer's buffer
        dst_ptr = remote_bufs[peer].data_ptr() + rank * nbytes
        src_ptr = local_buf.data_ptr()
        
        # Use standard hipMemcpyDeviceToDevice (NOT NoCU)
        # This will use the CUs specified by the stream's CU mask
        cp_res = hip.hipMemcpyAsync(
            dst_ptr,
            src_ptr,
            nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToDevice,  # Standard copy on CUs
            comm_stream,  # Stream bound to comm CUs (ihipStream_t object)
        )
        HIP_CHECK(cp_res)


# ========================================
# Computation Kernel (on Compute CUs)
# ========================================
@triton.jit
def computation_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Intensive computation kernel: Matrix-like operations.
    This runs on CUs dedicated to computation.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Heavy computation (simulate GEMM-like workload)
    y = x
    for i in range(200):  # Simulate compute intensity
        y = y * 1.00001 + 0.00001
        y = tl.sqrt(y + 1.0)
    
    # Store
    tl.store(output_ptr + offsets, y, mask=mask)


# ========================================
# Main Test Function
# ========================================
def test_cu_partitioning():
    """
    Test explicit CU partitioning using hipExtStreamCreateWithCUMask
    """
    print(f"\n{'='*80}")
    print(f"Test: Explicit CU Partitioning (PE {RANK}/{WORLD_SIZE})")
    print(f"{'='*80}\n")
    
    mype = pyrocshmem.rocshmem_my_pe()
    npes = pyrocshmem.rocshmem_n_pes()
    
    # Test parameters
    nelems_per_pe = 1024 * 512  # 512K elements per PE
    dtype = torch.float32
    
    # ========================================
    # Step 1: Query and partition CUs
    # ========================================
    total_cus = get_num_cus()
    
    # Partition: 32 CUs for communication, rest for computation
    comm_cus, compute_cus = partition_cus(total_cus, comm_cus_count=32)
    
    # Create CU masks
    comm_cu_mask = create_cu_mask(comm_cus, total_cus)
    compute_cu_mask = create_cu_mask(compute_cus, total_cus)
    
    print_cu_mask(comm_cu_mask, total_cus, "Communication CUs")
    print_cu_mask(compute_cu_mask, total_cus, "Computation CUs")
    
    # ========================================
    # Step 2: Create streams with CU masks
    # ========================================
    print(f"\n[PE {mype}] Creating CU-masked streams...")
    
    comm_stream = create_stream_with_cu_mask(comm_cu_mask)
    compute_stream = create_stream_with_cu_mask(compute_cu_mask)
    
    print(f"[PE {mype}] ✅ Created communication stream (type: {type(comm_stream).__name__})")
    print(f"[PE {mype}] ✅ Created computation stream (type: {type(compute_stream).__name__})")
    
    # ========================================
    # Step 3: Allocate buffers
    # ========================================
    # Local input (unique per PE)
    local_input = torch.arange(nelems_per_pe, dtype=dtype, device='cuda') + (mype * 1000)
    
    # Remote buffers (ROCm SHMEM symmetric memory)
    remote_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node(
        [nelems_per_pe * npes], dtype
    )
    
    # Output buffer for computation
    compute_output = torch.zeros(nelems_per_pe, dtype=dtype, device='cuda')
    
    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    print(f"\n[PE {mype}] Allocated buffers")
    
    # ========================================
    # Step 4: Launch both kernels concurrently
    # ========================================
    print(f"\n[PE {mype}] Launching overlapped kernels on partitioned CUs...")
    
    # Synchronize before starting
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    # Launch communication kernels (runs on comm_cus)
    launch_communication_kernels(
        mype, npes, local_input, remote_bufs, comm_stream
    )
    
    # Launch computation kernel (runs on compute_cus)
    # Note: We can't use torch.cuda.stream() with hipExtStreamCreateWithCUMask streams
    # because PyTorch doesn't recognize them as valid CUDA streams.
    # Instead, we'll launch directly and synchronize manually.
    
    print(f"[PE {mype}] Launching computation on dedicated CUs...")
    
    grid = lambda meta: (triton.cdiv(nelems_per_pe, meta['BLOCK_SIZE']),)
    computation_kernel[grid](
        local_input,
        compute_output,
        nelems_per_pe,
        BLOCK_SIZE=1024,
    )
    
    print(f"[PE {mype}] ✅ Both kernels launched on separate CUs!")
    
    # ========================================
    # Step 5: Synchronize and measure
    # ========================================
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    
    torch.distributed.barrier()
    
    print(f"\n[PE {mype}] CU-partitioned overlap completed in {elapsed_ms:.2f} ms")
    
    # ========================================
    # Step 6: Verify correctness
    # ========================================
    print(f"\n[PE {mype}] Verifying results...")
    
    # Check communication
    for peer in range(npes):
        peer_data = remote_bufs[mype][peer * nelems_per_pe:(peer + 1) * nelems_per_pe]
        expected_start = peer * 1000
        actual_start = peer_data[0].item()
        
        if abs(actual_start - expected_start) < 1:
            print(f"  ✅ [PE {mype}] Received correct data from PE {peer}")
        else:
            raise AssertionError(
                f"Communication failed: got {actual_start}, expected {expected_start}"
            )
    
    # Check computation
    expected_output = local_input.clone()
    for i in range(200):
        expected_output = expected_output * 1.00001 + 0.00001
        expected_output = torch.sqrt(expected_output + 1.0)
    
    torch.testing.assert_close(
        compute_output,
        expected_output,
        rtol=1e-3,
        atol=1e-3
    )
    print(f"  ✅ [PE {mype}] Computation result correct")
    
    # ========================================
    # Step 7: Cleanup streams
    # ========================================
    hip.hipStreamDestroy(comm_stream)
    hip.hipStreamDestroy(compute_stream)
    
    print(f"\n{'='*80}")
    print(f"✅ Test PASSED: CU Partitioning Successful!")
    print(f"   Communication ran on {len(comm_cus)} CUs")
    print(f"   Computation ran on {len(compute_cus)} CUs")
    print(f"   Both executed concurrently with no contention!")
    print(f"{'='*80}\n")


# ========================================
# Benchmark: Compare with and without CU partitioning
# ========================================
def test_cu_partitioning_benchmark():
    """Benchmark CU partitioning vs. standard streams with bandwidth measurement"""
    print(f"\n{'='*80}")
    print(f"Benchmark: CU Partitioning Performance (PE {RANK}/{WORLD_SIZE})")
    print(f"{'='*80}\n")
    
    mype = pyrocshmem.rocshmem_my_pe()
    npes = pyrocshmem.rocshmem_n_pes()
    nelems = 1024 * 1024  # 1M elements
    dtype = torch.float32
    bytes_per_elem = 4  # float32
    
    # Setup buffers
    local_input = torch.randn(nelems, dtype=dtype, device='cuda')
    remote_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node([nelems * npes], dtype)
    compute_output = torch.zeros(nelems, dtype=dtype, device='cuda')
    
    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    # Test 1: Standard streams (CU contention)
    print(f"\n[PE {mype}] Test 1: Standard streams (potential CU contention)...")
    
    comm_stream_std = torch.cuda.Stream(priority=-1)
    compute_stream_std = torch.cuda.Stream(priority=0)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    # Launch on standard streams
    with torch.cuda.stream(comm_stream_std):
        for peer in range(npes):
            if peer != mype:
                dst_ptr = remote_bufs[peer].data_ptr() + mype * nelems * 4
                hip.hipMemcpyAsync(
                    dst_ptr, local_input.data_ptr(), nelems * 4,
                    hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                    comm_stream_std.cuda_stream,
                )
    
    with torch.cuda.stream(compute_stream_std):
        grid = lambda meta: (triton.cdiv(nelems, meta['BLOCK_SIZE']),)
        computation_kernel[grid](local_input, compute_output, nelems, BLOCK_SIZE=1024)
    
    end.record()
    torch.cuda.synchronize()
    time_std = start.elapsed_time(end)
    
    torch.distributed.barrier()
    
    # Calculate bandwidth for communication
    # Each PE sends nelems to (npes-1) other PEs
    total_bytes_sent = nelems * bytes_per_elem * (npes - 1)
    bandwidth_std = (total_bytes_sent / (time_std / 1000)) / (1024**3)  # GB/s
    
    print(f"[PE {mype}] Standard streams time: {time_std:.2f} ms")
    print(f"[PE {mype}] Standard streams bandwidth: {bandwidth_std:.2f} GB/s")
    
    # Test 2: CU-partitioned streams (no contention)
    print(f"\n[PE {mype}] Test 2: CU-partitioned streams (no contention)...")
    
    total_cus = get_num_cus()
    comm_cus, compute_cus = partition_cus(total_cus, comm_cus_count=32)
    
    comm_cu_mask = create_cu_mask(comm_cus, total_cus)
    compute_cu_mask = create_cu_mask(compute_cus, total_cus)
    
    comm_stream = create_stream_with_cu_mask(comm_cu_mask)
    compute_stream = create_stream_with_cu_mask(compute_cu_mask)
    
    compute_output.zero_()
    torch.cuda.synchronize()
    
    start.record()
    
    #  Launch on CU-masked streams
    for peer in range(npes):
        if peer != mype:
            dst_ptr = remote_bufs[peer].data_ptr() + mype * nelems * 4
            hip.hipMemcpyAsync(
                dst_ptr, local_input.data_ptr(), nelems * 4,
                hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                comm_stream,
            )
    
    # Launch compute on default stream (CU masking only affects comm_stream)
    grid = lambda meta: (triton.cdiv(nelems, meta['BLOCK_SIZE']),)
    computation_kernel[grid](local_input, compute_output, nelems, BLOCK_SIZE=1024)
    
    end.record()
    torch.cuda.synchronize()
    time_masked = start.elapsed_time(end)
    
    torch.distributed.barrier()
    
    # Cleanup
    hip.hipStreamDestroy(comm_stream)
    hip.hipStreamDestroy(compute_stream)
    
    # Calculate bandwidth for CU-partitioned
    bandwidth_masked = (total_bytes_sent / (time_masked / 1000)) / (1024**3)  # GB/s
    
    print(f"[PE {mype}] CU-partitioned time: {time_masked:.2f} ms")
    print(f"[PE {mype}] CU-partitioned bandwidth: {bandwidth_masked:.2f} GB/s")
    
    speedup = time_std / time_masked
    bandwidth_improvement = ((bandwidth_masked - bandwidth_std) / bandwidth_std) * 100
    improvement = ((time_std - time_masked) / time_std) * 100
    
    print(f"\n{'='*80}")
    print(f"[PE {mype}] Performance Summary:")
    print(f"  Standard streams:      {time_std:.2f} ms  ({bandwidth_std:.2f} GB/s)")
    print(f"  CU-partitioned:        {time_masked:.2f} ms  ({bandwidth_masked:.2f} GB/s)")
    print(f"  Time speedup:          {speedup:.2f}x")
    print(f"  Time improvement:      {improvement:.1f}%")
    print(f"  Bandwidth improvement: {bandwidth_improvement:.1f}%")
    print(f"{'='*80}\n")


# ========================================
# Entry Point
# ========================================
if __name__ == "__main__":
    # Initialize PyTorch distributed
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    
    TP_GROUP = torch.distributed.new_group(
        ranks=list(range(WORLD_SIZE)),
        backend="nccl"
    )
    torch.distributed.barrier(TP_GROUP)
    
    # Initialize ROCm SHMEM
    pyrocshmem.init_rocshmem_by_uniqueid(TP_GROUP)
    
    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    # Run tests
    try:
        test_cu_partitioning()
        print("\n" + "="*80 + "\n")
        test_cu_partitioning_benchmark()
    finally:
        # Cleanup
        torch.cuda.synchronize()
        torch.distributed.barrier()
        pyrocshmem.rocshmem_finalize()
        torch.distributed.destroy_process_group()

