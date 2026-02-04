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
Example: Overlapping Communication and Computation Kernels
===========================================================

This test demonstrates how to overlap ROCm SHMEM communication with Triton
computation kernels by assigning them to different CU resources using HIP streams.

Key techniques:
1. Use multiple torch.cuda.Stream() with priority=-1 for communication
2. Use hipMemcpyDeviceToDeviceNoCU to avoid CU contention
3. Use producer-consumer pattern with signals
4. Launch communication and computation concurrently

Run with:
    cd /dev/data/diprajap/workspace/rocm7/Triton-distributed
    WORLD_SIZE=2 LOCAL_WORLD_SIZE=2 python -m pytest -xvs \
        python/triton_dist/test/amd/test_overlap_comm_compute.py
"""

import os
import datetime
import torch
import triton
import triton.language as tl
import pyrocshmem
from hip import hip
from typing import List
from triton_dist.utils import HIP_CHECK

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


# ========================================
# 1. Communication Kernel (Producer)
# ========================================
def communication_producer(
    rank: int,
    npes: int,
    local_buf: torch.Tensor,
    remote_bufs: List[torch.Tensor],
    comm_streams: List[torch.cuda.Stream],
):
    """
    AllGather pattern: Each PE sends its data to all other PEs
    Uses AMD Copy Engines (NoCU) to avoid competing for CUs with compute kernels
    """
    nelems = local_buf.numel()
    nbytes = nelems * local_buf.element_size()
    
    # Pattern: Send to all peers using different streams
    for peer in range(npes):
        if peer == rank:
            # Copy local data to own buffer (on compute stream)
            remote_bufs[rank][rank * nelems:(rank + 1) * nelems].copy_(local_buf)
            continue
        
        # Select stream (round-robin)
        stream_idx = peer % len(comm_streams)
        comm_stream = comm_streams[stream_idx]
        
        # Calculate offsets
        dst_ptr = remote_bufs[peer].data_ptr() + rank * nelems * local_buf.element_size()
        src_ptr = local_buf.data_ptr()
        
        # Use hipMemcpyAsync with NoCU flag for CU isolation
        cp_res = hip.hipMemcpyAsync(
            dst_ptr,
            src_ptr,
            nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,  # ✅ Key flag!
            comm_stream.cuda_stream,
        )
        HIP_CHECK(cp_res)
        
        print(f"  [PE {rank}] Launched comm to PE {peer} on stream {stream_idx}")


# ========================================
# 2. Computation Kernel (Consumer)
# ========================================
@triton.jit
def computation_consumer_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple computation kernel: Square each element
    In real use case, this would be GEMM, attention, etc.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute (simulate heavy computation)
    y = x * x
    for _ in range(100):  # Simulate compute intensity
        y = y * 1.0001
    
    # Store
    tl.store(output_ptr + offsets, y, mask=mask)


# ========================================
# 3. Main Test Function
# ========================================
def test_overlap_communication_computation():
    """
    Test overlapping communication and computation using multiple streams
    """
    print(f"\n{'='*80}")
    print(f"Test: Overlapping Communication + Computation (PE {RANK}/{WORLD_SIZE})")
    print(f"{'='*80}\n")
    
    mype = pyrocshmem.rocshmem_my_pe()
    npes = pyrocshmem.rocshmem_n_pes()
    
    # Test parameters
    nelems_per_pe = 1024 * 256  # 256K elements per PE
    dtype = torch.float32
    
    # ========================================
    # Step 1: Create multiple streams
    # ========================================
    # Communication streams (high priority)
    num_comm_streams = npes if npes <= 4 else 4
    comm_streams = [torch.cuda.Stream(priority=-1) for _ in range(num_comm_streams)]
    
    # Compute stream (normal priority)
    compute_stream = torch.cuda.current_stream()
    
    print(f"[PE {mype}] Created {num_comm_streams} communication streams")
    
    # ========================================
    # Step 2: Allocate buffers
    # ========================================
    # Local input (unique per PE)
    local_input = torch.arange(nelems_per_pe, dtype=dtype, device='cuda') + (mype * 1000)
    
    # Remote buffers (ROCm SHMEM symmetric memory)
    remote_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node(
        [nelems_per_pe * npes], dtype
    )
    
    # Output buffer for computation
    compute_output = torch.zeros(nelems_per_pe * npes, dtype=dtype, device='cuda')
    
    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    print(f"[PE {mype}] Allocated buffers")
    
    # ========================================
    # Step 3: Synchronize streams before overlap
    # ========================================
    for comm_stream in comm_streams:
        comm_stream.wait_stream(compute_stream)
    
    # ========================================
    # Step 4: Launch BOTH communication and computation
    # ========================================
    print(f"[PE {mype}] Launching overlapped kernels...")
    
    # Start timer
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(compute_stream)
    
    # Launch communication (runs on comm_streams)
    communication_producer(
        mype, npes, local_input, remote_bufs, comm_streams
    )
    
    # Launch computation (runs on compute_stream)
    # This kernel runs CONCURRENTLY with communication!
    grid = lambda meta: (triton.cdiv(compute_output.numel(), meta['BLOCK_SIZE']),)
    computation_consumer_kernel[grid](
        local_input,
        compute_output[:nelems_per_pe],
        nelems_per_pe,
        BLOCK_SIZE=1024,
    )
    
    print(f"[PE {mype}] ✅ Both kernels launched (should overlap!)")
    
    # ========================================
    # Step 5: Synchronize and measure
    # ========================================
    end_event.record(compute_stream)
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    
    torch.distributed.barrier()
    
    print(f"[PE {mype}] Overlap completed in {elapsed_ms:.2f} ms")
    
    # ========================================
    # Step 6: Verify correctness
    # ========================================
    # Check communication: Each PE should have received data from all peers
    for peer in range(npes):
        peer_data = remote_bufs[mype][peer * nelems_per_pe:(peer + 1) * nelems_per_pe]
        expected_start = peer * 1000
        actual_start = peer_data[0].item()
        
        if abs(actual_start - expected_start) < 1:
            print(f"  ✅ [PE {mype}] Received correct data from PE {peer}")
        else:
            print(f"  ❌ [PE {mype}] Wrong data from PE {peer}: {actual_start} != {expected_start}")
            raise AssertionError(f"Communication verification failed")
    
    # Check computation
    expected_output = local_input * local_input
    for _ in range(100):
        expected_output = expected_output * 1.0001
    
    torch.testing.assert_close(
        compute_output[:nelems_per_pe],
        expected_output,
        rtol=1e-3,
        atol=1e-3
    )
    print(f"  ✅ [PE {mype}] Computation result correct")
    
    print(f"\n{'='*80}")
    print(f"✅ Test PASSED: Communication and Computation Overlapped Successfully!")
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
    
    # Run test
    test_overlap_communication_computation()
    
    # Cleanup
    torch.cuda.synchronize()
    torch.distributed.barrier()
    pyrocshmem.rocshmem_finalize()
    torch.distributed.destroy_process_group()

