# Overlapping Communication and Computation in Triton-Distributed

This guide explains how to overlap ROCm SHMEM communication and Triton computation kernels by assigning them to different GPU resources using HIP streams.

## 🎯 Core Concept: CU Partitioning

AMD GPUs have two types of execution resources:
1. **Compute Units (CUs)**: Execute compute kernels (GEMM, attention, etc.)
2. **Copy Engines (DMA)**: Dedicated hardware for memory copies

By using **multiple HIP streams** and the **`hipMemcpyDeviceToDeviceNoCU`** flag, you can run communication and computation **concurrently without CU contention**.

## 📋 Quick Reference

### 1. Create Multiple Streams

```python
import torch
from typing import List

# Communication streams (high priority = -1 for lower latency)
comm_streams: List[torch.cuda.Stream] = [
    torch.cuda.Stream(priority=-1) 
    for _ in range(num_ranks)
]

# Computation stream (use current or create new)
compute_stream = torch.cuda.current_stream()
# OR
compute_stream = torch.cuda.Stream(priority=0)  # normal priority
```

**Key Stream APIs:**
```python
# Get HIP stream handle for HIP APIs
hip_stream = torch_stream.cuda_stream

# Wait for another stream before continuing
stream_a.wait_stream(stream_b)  # stream_a waits for stream_b

# Synchronize specific stream
stream.synchronize()

# Execute code in specific stream context
with torch.cuda.stream(my_stream):
    # Operations here use my_stream
    kernel[grid](...)
```

### 2. Use Copy Engines (NO CU Usage!)

```python
from hip import hip
from triton_dist.utils import HIP_CHECK

# ❌ Standard copy - uses CUs, competes with compute kernels
hip.hipMemcpyKind.hipMemcpyDeviceToDevice

# ✅ AMD Copy Engine - isolated hardware, NO CU contention!
hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU
```

**Example:**
```python
cp_res = hip.hipMemcpyAsync(
    dst_ptr,              # Destination device pointer
    src_ptr,              # Source device pointer
    nbytes,               # Number of bytes
    hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,  # Use Copy Engine!
    comm_stream.cuda_stream,  # Execute on communication stream
)
HIP_CHECK(cp_res)  # Check for errors
```

### 3. Pattern: AllGather with Stream Pool

```python
def allgather_multi_stream(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_buffers: List[torch.Tensor],  # ROCm SHMEM symmetric buffers
    comm_streams: List[torch.cuda.Stream],
):
    """Copy local data to all remote PEs using multiple streams"""
    M, N = local_tensor.shape
    elem_size = local_tensor.element_size()
    nbytes = M * N * elem_size
    
    for peer in range(num_ranks):
        if peer == rank:
            continue  # Skip self
        
        # Round-robin stream selection
        stream_idx = peer % len(comm_streams)
        comm_stream = comm_streams[stream_idx]
        
        # Calculate destination offset in remote buffer
        dst_offset = rank * M * N * elem_size
        dst_ptr = remote_buffers[peer].data_ptr() + dst_offset
        src_ptr = local_tensor.data_ptr()
        
        # Launch copy on communication stream (uses Copy Engine)
        cp_res = hip.hipMemcpyAsync(
            dst_ptr,
            src_ptr,
            nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
            comm_stream.cuda_stream,
        )
        HIP_CHECK(cp_res)
```

### 4. Pattern: Chunked Communication with Producer-Consumer

For fine-grained overlap, split communication into chunks and use signals:

```python
def chunked_allgather_producer(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_buffers: List[torch.Tensor],
    signal_buffers: List[torch.Tensor],  # Sync signals
    comm_streams: List[torch.cuda.Stream],
    chunk_size: int,
):
    """
    AllGather with chunking: Allows computation to start before 
    all communication completes
    """
    M, N = local_tensor.shape
    num_chunks = M // chunk_size
    elem_size = local_tensor.element_size()
    
    # Helper: one tensor for setting signals via memcpy
    one = torch.ones(1, dtype=torch.int32, device='cuda')
    
    for peer in range(num_ranks):
        if peer == rank:
            continue
        
        for chunk_idx in range(num_chunks):
            # Select stream
            stream_idx = peer % len(comm_streams)
            comm_stream = comm_streams[stream_idx]
            
            # Calculate chunk offsets
            chunk_offset = chunk_idx * chunk_size * N * elem_size
            dst_offset = (rank * M + chunk_idx * chunk_size) * N * elem_size
            chunk_bytes = chunk_size * N * elem_size
            
            # Copy chunk data
            cp_res = hip.hipMemcpyAsync(
                remote_buffers[peer].data_ptr() + dst_offset,
                local_tensor.data_ptr() + chunk_offset,
                chunk_bytes,
                hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                comm_stream.cuda_stream,
            )
            HIP_CHECK(cp_res)
            
            # Set signal: this chunk is ready
            signal_idx = rank * num_chunks + chunk_idx
            cp_res = hip.hipMemcpyAsync(
                signal_buffers[peer].data_ptr() + signal_idx * 4,
                one.data_ptr(),
                4,  # int32 size
                hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                comm_stream.cuda_stream,
            )
            HIP_CHECK(cp_res)
```

**Consumer Triton Kernel:**
```python
@triton.jit
def gemm_consumer_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    signal_ptr,  # Points to signal buffer
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """GEMM that waits for communication chunks before computing"""
    pid_m = tl.program_id(0)
    
    # Calculate which chunks this tile depends on
    m_start = pid_m * BLOCK_M
    m_end = m_start + BLOCK_M
    chunk_start = m_start // chunk_size
    chunk_end = (m_end - 1) // chunk_size
    
    # Wait for all dependent chunks
    for chunk_id in range(chunk_start, chunk_end + 1):
        signal_addr = signal_ptr + chunk_id * 4
        # Poll until signal == 1 (chunk ready)
        tl.wait_value_eq(signal_addr, 1)
    
    # Now safe to compute!
    # ... (standard GEMM computation) ...
```

### 5. Complete Workflow

```python
def overlap_allgather_gemm(
    local_input: torch.Tensor,  # [M_local, K]
    weight: torch.Tensor,       # [K, N]
    rank: int,
    num_ranks: int,
):
    """Complete example: Overlap AllGather with GEMM"""
    
    # 1. Setup streams
    comm_streams = [torch.cuda.Stream(priority=-1) for _ in range(num_ranks)]
    compute_stream = torch.cuda.current_stream()
    
    # 2. Allocate buffers
    M_local, K = local_input.shape
    M_total = M_local * num_ranks
    
    remote_buffers = pyrocshmem.rocshmem_create_tensor_list_intra_node(
        [M_total, K], local_input.dtype
    )
    signal_buffers = pyrocshmem.rocshmem_create_tensor_list_intra_node(
        [M_total // chunk_size], torch.int32
    )
    
    # 3. Synchronize streams (establish happens-before relationship)
    for comm_stream in comm_streams:
        comm_stream.wait_stream(compute_stream)
    
    # 4. Launch communication (producer)
    chunked_allgather_producer(
        rank, num_ranks, local_input, remote_buffers,
        signal_buffers, comm_streams, chunk_size=128
    )
    
    # 5. Launch computation (consumer) - OVERLAPS with communication!
    output = torch.zeros(M_total, N, device='cuda')
    grid = lambda meta: (triton.cdiv(M_total, meta['BLOCK_M']),)
    
    gemm_consumer_kernel[grid](
        remote_buffers[rank], weight, output,
        signal_buffers[rank],
        M_total, N, K, chunk_size=128,
        BLOCK_M=128,
    )
    
    # 6. Synchronize all streams
    torch.cuda.synchronize()
    
    return output
```

## 🔧 ROCm SHMEM Specific APIs

### Stream-Aware Barrier
```python
# Barrier on specific stream (graph-compatible)
current_stream = torch.cuda.current_stream()
pyrocshmem.rocshmem_barrier_all_on_stream(current_stream.cuda_stream)
```

### Module Initialization for CUDA Graphs
```python
# Initialize rocSHMEM context for a kernel module
# Required for torch.cuda.graph() compatibility
kernel_module = my_kernel.cache[0][1].module
module_handle = kernel_module.__getattribute__('module')
hip_stream = torch.cuda.current_stream().cuda_stream

ret = pyrocshmem.rocshmem_hipmodule_init(module_handle, hip_stream)
if ret != 0:
    raise RuntimeError(f"rocshmem_hipmodule_init failed: {ret}")
```

### Device Context Management
```python
# Get device context pointer (for passing to kernels)
ctx = pyrocshmem.rocshmem_get_device_ctx()

# In Triton kernel, set context
@triton.jit
def my_kernel(ctx, ...):
    libshmem.set_rocshmem_ctx(ctx)
    mype = libshmem.my_pe()
    npes = libshmem.n_pes()
    # ... rocSHMEM operations ...
```

## 📊 Performance Tuning Guide

| Parameter | Recommended Value | Reasoning |
|-----------|------------------|-----------|
| **# of comm streams** | `num_ranks` (up to 8) | Maximize AMD GPU link utilization |
| **Stream priority** | Communication: `-1`, Compute: `0` | Lower latency for communication |
| **Chunk size** | 128-256 rows | Balance overhead vs. overlap opportunity |
| **Memory copy kind** | `hipMemcpyDeviceToDeviceNoCU` | Use dedicated Copy Engines |
| **Signal mechanism** | Memcpy small values | Faster than `hipStreamWaitValue` on AMD |

### Measuring Overlap Efficiency

```python
# Use events on different streams
comm_start = torch.cuda.Event(enable_timing=True)
comm_end = torch.cuda.Event(enable_timing=True)
compute_start = torch.cuda.Event(enable_timing=True)
compute_end = torch.cuda.Event(enable_timing=True)

# Record on different streams
comm_start.record(comm_streams[0])
# ... launch communication ...
comm_end.record(comm_streams[0])

compute_start.record(compute_stream)
# ... launch computation ...
compute_end.record(compute_stream)

torch.cuda.synchronize()

comm_time = comm_start.elapsed_time(comm_end)
compute_time = compute_start.elapsed_time(compute_end)
overlap_efficiency = 1.0 - (total_time - max(comm_time, compute_time)) / min(comm_time, compute_time)
```

## 🎨 Visual Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GPU Architecture                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────┐  ┌───────────────────────────┐   │
│  │   Compute Units (CUs)    │  │   Copy Engines (DMA)      │   │
│  │                          │  │                           │   │
│  │  Used for:               │  │  Used for:                │   │
│  │  - Triton kernels        │  │  - hipMemcpyAsync +       │   │
│  │  - GEMM, attention       │  │    NoCU flag              │   │
│  │  - General compute       │  │  - Background transfers   │   │
│  │                          │  │  - Signal updates         │   │
│  └──────────────────────────┘  └───────────────────────────┘   │
│         ▲                             ▲                          │
│         │                             │                          │
│         │                             │                          │
│   torch.cuda.Stream()          torch.cuda.Stream()              │
│     (priority=0)                 (priority=-1)                   │
│         │                             │                          │
│         │                             │                          │
│   Compute Kernels             Communication                      │
│   (Triton GEMM)              (ROCm SHMEM copies)                │
│                                                                   │
│   ✅ CONCURRENT EXECUTION - No CU contention!                    │
└─────────────────────────────────────────────────────────────────┘

Timeline:
═══════════════════════════════════════════════════════════════
Comm Stream:  [Chunk 0][Chunk 1][Chunk 2][Chunk 3]
              └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
                 │        │        │        │
              signal   signal   signal   signal
                 │        │        │        │
                 ▼        ▼        ▼        ▼
Compute Stream:  [wait][GEMM Tile 0][GEMM Tile 1][GEMM Tile 2]
═══════════════════════════════════════════════════════════════
```

## 📚 Examples in Codebase

1. **Basic AllGather with streams**: `python/triton_dist/test/amd/test_rocshmem_api.py:test_rocshmem_memcpy()`
2. **Overlapped AllGather+GEMM**: `tutorials/09-AMD-overlapping-allgather-gemm.py`
3. **Overlapped GEMM+ReduceScatter**: `tutorials/10-AMD-overlapping-gemm-reduce-scatter.py`
4. **Production AllGather**: `python/triton_dist/kernels/amd/all_gather_gemm.py`

## 🚀 Getting Started

1. **Run the example test:**
   ```bash
   cd /dev/data/diprajap/workspace/rocm7/Triton-distributed
   
   # Single node, 2 GPUs
   WORLD_SIZE=2 LOCAL_WORLD_SIZE=2 \
   python python/triton_dist/test/amd/test_overlap_comm_compute.py
   ```

2. **Run tutorial:**
   ```bash
   bash scripts/launch_amd.sh tutorials/09-AMD-overlapping-allgather-gemm.py
   ```

3. **Measure performance:**
   ```bash
   # With profiling
   python -m triton_dist.test.amd.test_rocshmem_api --profile
   ```

## ⚠️ Common Pitfalls

1. **Forgetting `wait_stream()`**: Communication and compute streams are independent; you must explicitly synchronize dependencies.
   
   ```python
   # ❌ Wrong - no synchronization
   comm_stream.launch_copy(...)
   compute_stream.launch_kernel(...)
   
   # ✅ Correct
   comm_stream.wait_stream(compute_stream)  # Comm waits for compute
   comm_stream.launch_copy(...)
   compute_stream.launch_kernel(...)  # Can now overlap!
   ```

2. **Using `hipMemcpyDeviceToDevice` instead of `NoCU`**: This defeats the purpose as copies will compete for CUs!

3. **Too many streams**: More streams != better performance. Use `num_ranks` streams (typically 2-8).

4. **Not initializing signals**: Always zero-initialize signal buffers before use.

5. **CUDA graph without `rocshmem_hipmodule_init()`**: Leads to "cannot capture legacy stream" errors.

## 🔍 Debugging Tips

```python
# Enable HIP error checking
import os
os.environ['HIP_LAUNCH_BLOCKING'] = '1'

# Check stream order
print(f"Stream {stream_id}: {stream.cuda_stream}")

# Verify NoCU flag usage
from hip import hip
assert copy_kind == hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU

# Profile with rocprof
# rocprof --hip-trace python your_script.py
```

## 📖 References

- [AMD HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [PyTorch CUDA Streams](https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams)
- [Triton Documentation](https://triton-lang.org/)
- [ROCm SHMEM Specification](https://rocmshmem.docs.amd.com/)

