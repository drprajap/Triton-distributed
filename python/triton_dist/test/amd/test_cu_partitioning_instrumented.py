"""
CU Partitioning Test with Manual Instrumentation
=================================================

This version adds detailed timing using HIP events, which works without profilers.
It provides stream-level timing for communication and computation operations.
"""

import os
import sys
import torch
import torch.distributed as dist
from hip import hip

# Import from the original test
sys.path.insert(0, os.path.dirname(__file__))
from test_cu_partitioning import (
    launch_communication_kernels,
    computation_kernel,
    partition_cus,
    create_stream_with_cu_mask
)

RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", str(RANK)))


def hip_check(result):
    """Check HIP API call result"""
    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result
    if err != hip.hipSuccess:
        raise RuntimeError(f"HIP Error: {err}")


class HIPEventTimer:
    """Context manager for timing HIP operations with events"""
    def __init__(self, name, stream=None):
        self.name = name
        self.stream = stream
        err, self.start_event = hip.hipEventCreate()
        hip_check(err)
        err, self.end_event = hip.hipEventCreate()
        hip_check(err)
        self.elapsed_ms = None
    
    def __enter__(self):
        if self.stream is not None:
            hip_check(hip.hipEventRecord(self.start_event, self.stream))
        else:
            hip_check(hip.hipEventRecord(self.start_event, None))
        return self
    
    def __exit__(self, *args):
        if self.stream is not None:
            hip_check(hip.hipEventRecord(self.end_event, self.stream))
        else:
            hip_check(hip.hipEventRecord(self.end_event, None))
        hip_check(hip.hipEventSynchronize(self.end_event))
        err, elapsed = hip.hipEventElapsedTime(self.start_event, self.end_event)
        hip_check(err)
        self.elapsed_ms = elapsed
        print(f"[PE {RANK}] ⏱️  {self.name}: {self.elapsed_ms:.3f} ms")
    
    def __del__(self):
        try:
            hip.hipEventDestroy(self.start_event)
            hip.hipEventDestroy(self.end_event)
        except:
            pass


def test_cu_partitioning_instrumented():
    """Test with detailed timing instrumentation"""
    
    # Initialize distributed
    if WORLD_SIZE > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(LOCAL_RANK)
    
    # Import rocSHMEM
    import pyrocshmem
    pyrocshmem.rocshmem_init()
    mype = pyrocshmem.my_pe()
    npes = pyrocshmem.n_pes()
    
    print(f"\n{'='*80}")
    print(f"[PE {mype}] CU Partitioning Test - INSTRUMENTED VERSION")
    print(f"{'='*80}\n")
    
    # Get buffer configuration
    rows = int(os.environ.get("CU_PART_BUFFER_ROWS", "4096"))
    cols = int(os.environ.get("CU_PART_BUFFER_COLS", "4096"))
    use_fp16 = int(os.environ.get("CU_PART_USE_FP16", "0"))
    dtype = torch.float16 if use_fp16 else torch.float32
    
    print(f"[PE {mype}] Configuration:")
    print(f"  Buffer shape: {rows}×{cols}")
    print(f"  Dtype: {dtype}")
    print(f"  PEs: {npes}")
    
    # Partition CUs
    err, props = hip.hipGetDeviceProperties(LOCAL_RANK)
    hip_check(err)
    total_cus = props.multiProcessorCount
    
    comm_cus, compute_cus = partition_cus(total_cus, comm_cus_count=32)
    
    # Create CU-masked streams
    with HIPEventTimer(f"Stream creation", None):
        comm_stream = create_stream_with_cu_mask(comm_cus)
        compute_stream = create_stream_with_cu_mask(compute_cus)
    
    # Allocate buffers
    nelems = rows * cols
    nelems_per_pe = nelems
    bytes_per_elem = 2 if use_fp16 else 4
    
    with HIPEventTimer(f"Buffer allocation", None):
        # rocSHMEM symmetric buffers
        local_buf_ptr = pyrocshmem.rocshmem_malloc(nelems * bytes_per_elem)
        remote_buf_ptr = pyrocshmem.rocshmem_malloc(nelems * npes * bytes_per_elem)
        
        # Torch tensors
        local_bufs = []
        remote_bufs = []
        for _ in range(npes):
            local_bufs.append(torch.zeros(nelems, dtype=dtype, device='cuda'))
            remote_bufs.append(torch.zeros(nelems, dtype=dtype, device='cuda'))
        
        local_input = torch.arange(nelems, dtype=dtype, device='cuda') + mype * 1000
        compute_output = torch.zeros(nelems, dtype=dtype, device='cuda')
    
    print(f"[PE {mype}] Starting timed execution...")
    print()
    
    # ========================================
    # TIMED EXECUTION
    # ========================================
    
    # Warmup iteration (not timed)
    launch_communication_kernels(
        local_bufs, remote_bufs, local_buf_ptr, remote_buf_ptr,
        nelems, nelems_per_pe, npes, mype, bytes_per_elem, comm_stream
    )
    computation_kernel[(200,)](local_input, compute_output, nelems, num_warps=8, stream=compute_stream)
    hip_check(hip.hipStreamSynchronize(comm_stream))
    hip_check(hip.hipStreamSynchronize(compute_stream))
    
    print(f"[PE {mype}] Warmup complete, starting benchmark...\n")
    
    # Timed iteration
    timer_comm = HIPEventTimer("Communication kernels", comm_stream)
    timer_comp = HIPEventTimer("Computation kernel", compute_stream)
    timer_total = HIPEventTimer("Total (overlapped)", None)
    
    with timer_total:
        with timer_comm:
            launch_communication_kernels(
                local_bufs, remote_bufs, local_buf_ptr, remote_buf_ptr,
                nelems, nelems_per_pe, npes, mype, bytes_per_elem, comm_stream
            )
        
        with timer_comp:
            computation_kernel[(200,)](local_input, compute_output, nelems, num_warps=8, stream=compute_stream)
    
    # Calculate overlap metrics
    comm_time = timer_comm.elapsed_ms
    comp_time = timer_comp.elapsed_ms
    total_time = timer_total.elapsed_ms
    
    # Theoretical total if sequential
    sequential_time = comm_time + comp_time
    overlap_time = sequential_time - total_time
    overlap_pct = (overlap_time / sequential_time) * 100 if sequential_time > 0 else 0
    
    print(f"\n[PE {mype}] {'='*80}")
    print(f"[PE {mype}] TIMING ANALYSIS")
    print(f"[PE {mype}] {'='*80}")
    print(f"[PE {mype}] Communication time:    {comm_time:8.3f} ms")
    print(f"[PE {mype}] Computation time:      {comp_time:8.3f} ms")
    print(f"[PE {mype}] Total (overlapped):    {total_time:8.3f} ms")
    print(f"[PE {mype}] Sequential (no overlap):{sequential_time:8.3f} ms")
    print(f"[PE {mype}] Overlap savings:       {overlap_time:8.3f} ms ({overlap_pct:.1f}%)")
    
    # Calculate bandwidth
    total_data_gb = (nelems * bytes_per_elem * npes) / (1024**3)
    bandwidth_gbps = (total_data_gb / (comm_time / 1000)) if comm_time > 0 else 0
    
    print(f"[PE {mype}] Communication bandwidth: {bandwidth_gbps:.2f} GB/s")
    print(f"[PE {mype}] {'='*80}\n")
    
    # Gather timing stats across all PEs
    if WORLD_SIZE > 1:
        # Convert to tensors for all_reduce
        comm_time_tensor = torch.tensor([comm_time], device='cuda')
        comp_time_tensor = torch.tensor([comp_time], device='cuda')
        total_time_tensor = torch.tensor([total_time], device='cuda')
        overlap_pct_tensor = torch.tensor([overlap_pct], device='cuda')
        bandwidth_tensor = torch.tensor([bandwidth_gbps], device='cuda')
        
        # Get max times (slowest PE)
        dist.all_reduce(comm_time_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(comp_time_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(total_time_tensor, op=dist.ReduceOp.MAX)
        
        # Get average overlap and bandwidth
        dist.all_reduce(overlap_pct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(bandwidth_tensor, op=dist.ReduceOp.SUM)
        overlap_pct_avg = overlap_pct_tensor.item() / npes
        bandwidth_avg = bandwidth_tensor.item() / npes
        
        if mype == 0:
            print(f"\n{'='*80}")
            print(f"AGGREGATE STATISTICS (across {npes} PEs)")
            print(f"{'='*80}")
            print(f"Max communication time:    {comm_time_tensor.item():8.3f} ms")
            print(f"Max computation time:      {comp_time_tensor.item():8.3f} ms")
            print(f"Max total time:            {total_time_tensor.item():8.3f} ms")
            print(f"Average overlap:           {overlap_pct_avg:8.1f}%")
            print(f"Average per-PE bandwidth:  {bandwidth_avg:8.2f} GB/s")
            print(f"Aggregate bandwidth:       {bandwidth_avg * npes:8.2f} GB/s")
            print(f"{'='*80}\n")
    
    # Cleanup
    hip.hipStreamDestroy(comm_stream)
    hip.hipStreamDestroy(compute_stream)
    pyrocshmem.rocshmem_finalize()
    
    if WORLD_SIZE > 1:
        dist.destroy_process_group()
    
    print(f"[PE {mype}] ✅ Test complete!\n")


if __name__ == "__main__":
    test_cu_partitioning_instrumented()




