# Quick Reference: Communication-Computation Overlap APIs

## 🎯 Two Approaches

### 1️⃣ Copy Engine (Simple)
Uses dedicated DMA hardware, **zero CU usage**
```python
hip.hipMemcpyAsync(..., hip.hipMemcpyDeviceToDeviceNoCU, stream)
```

### 2️⃣ CU Partitioning (Advanced)
Explicitly assign CUs to different streams
```python
hip.hipExtStreamCreateWithCUMask(stream, mask_size, cu_mask)
```

---

## 📋 Copy Engine Cheat Sheet

### Basic Setup (3 lines)
```python
from hip import hip

comm_stream = torch.cuda.Stream(priority=-1)
hip.hipMemcpyAsync(dst, src, nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU, comm_stream.cuda_stream)
triton_kernel[grid](...)  # Runs concurrently!
```

### Multiple Streams Pattern
```python
# Create stream pool
comm_streams = [torch.cuda.Stream(priority=-1) for _ in range(num_ranks)]

# Launch communication
for i, peer in enumerate(peers):
    hip.hipMemcpyAsync(
        dst_ptr, src_ptr, nbytes,
        hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
        comm_streams[i].cuda_stream
    )

# Launch computation (overlaps!)
compute_kernel[grid](...)
```

### ROCm SHMEM Integration
```python
# Stream-aware barrier
pyrocshmem.rocshmem_barrier_all_on_stream(stream.cuda_stream)

# CUDA graph compatibility
pyrocshmem.rocshmem_hipmodule_init(module, stream.cuda_stream)
```

---

## 📋 CU Partitioning Cheat Sheet

### Complete Workflow (7 steps)
```python
from hip import hip
import ctypes

# 1. Query CUs
total_cus = torch.cuda.get_device_properties(0).multi_processor_count

# 2. Partition CUs (30% comm, 70% compute)
num_comm = int(total_cus * 0.3)
comm_cus = list(range(num_comm))
compute_cus = list(range(num_comm, total_cus))

# 3. Create bit masks
def create_cu_mask(cu_list, total):
    mask = [0] * ((total + 31) // 32)
    for cu in cu_list:
        mask[cu // 32] |= (1 << (cu % 32))
    return mask

comm_mask = create_cu_mask(comm_cus, total_cus)
compute_mask = create_cu_mask(compute_cus, total_cus)

# 4. Create CU-masked streams
def create_stream_with_mask(mask):
    mask_array = (ctypes.c_uint32 * len(mask))(*mask)
    # Python binding returns (error, stream) tuple
    err, stream = hip.hipExtStreamCreateWithCUMask(len(mask), mask_array)
    if err != hip.hipError_t.hipSuccess:
        raise RuntimeError(f"Failed: {err}")
    return stream

comm_stream = create_stream_with_mask(comm_mask)
compute_stream = create_stream_with_mask(compute_mask)

# 5. Launch on comm CUs
hip.hipMemcpyAsync(dst, src, nbytes, hip.hipMemcpyDeviceToDevice, comm_stream)

# 6. Launch on compute CUs
class StreamWrapper:
    def __init__(self, h): self.cuda_stream = h

with torch.cuda.stream(StreamWrapper(compute_stream)):
    compute_kernel[grid](...)

# 7. Cleanup
hip.hipStreamDestroy(comm_stream)
hip.hipStreamDestroy(compute_stream)
```

---

## 🔑 Key API Functions

| API | Purpose | Example |
|-----|---------|---------|
| `torch.cuda.Stream(priority=-1)` | Create high-priority stream | `s = torch.cuda.Stream(priority=-1)` |
| `torch.cuda.current_stream()` | Get current stream | `cur = torch.cuda.current_stream()` |
| `stream.cuda_stream` | Get HIP stream handle | `hip_s = stream.cuda_stream` |
| `torch.cuda.stream(s)` | Execute in stream context | `with torch.cuda.stream(s): ...` |
| `stream.wait_stream(other)` | Synchronize streams | `s1.wait_stream(s2)` |
| `torch.cuda.synchronize()` | Wait for all streams | `torch.cuda.synchronize()` |
| `hip.hipMemcpyAsync()` | Async memory copy | `hip.hipMemcpyAsync(dst, src, n, kind, stream)` |
| `hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU` | Copy Engine flag | Use with `hipMemcpyAsync` |
| `hip.hipExtStreamCreateWithCUMask()` | Create CU-masked stream | `hip.hipExtStreamCreateWithCUMask(s, size, mask)` |
| `hip.hipStreamDestroy()` | Destroy stream | `hip.hipStreamDestroy(stream)` |
| `torch.cuda.get_device_properties()` | Get GPU info | `.multi_processor_count` |
| `torch.cuda.Event(enable_timing=True)` | Create timing event | `e = torch.cuda.Event(enable_timing=True)` |
| `event.record(stream)` | Record event | `e.record(s)` |
| `event.elapsed_time(other)` | Measure time | `e1.elapsed_time(e2)` |

---

## 🎨 Memory Copy Kinds

| Flag | Hardware | CU Usage | When to Use |
|------|----------|----------|-------------|
| `hipMemcpyDeviceToDevice` | CUs | ✅ Uses CUs | With CU-masked streams |
| `hipMemcpyDeviceToDeviceNoCU` | Copy Engines | ❌ Zero | For simple data movement |
| `hipMemcpyHostToDevice` | DMA | ❌ Zero | CPU → GPU |
| `hipMemcpyDeviceToHost` | DMA | ❌ Zero | GPU → CPU |

---

## 📊 Performance Tips

### Stream Priorities
```python
# Communication: high priority (lower latency)
comm_stream = torch.cuda.Stream(priority=-1)

# Computation: normal priority
compute_stream = torch.cuda.Stream(priority=0)  # or torch.cuda.current_stream()
```

### Optimal CU Ratios

| Workload | Comm % | Compute % |
|----------|--------|-----------|
| GEMM-heavy | 10-20% | 80-90% |
| Balanced | 30-40% | 60-70% |
| Comm-heavy | 40-50% | 50-60% |

### CU Partition Strategies
```python
# Sequential
comm_cus = list(range(33))
compute_cus = list(range(33, 110))

# Interleaved (better memory BW)
comm_cus = [i for i in range(110) if i % 2 == 1]
compute_cus = [i for i in range(110) if i % 2 == 0]

# NUMA-aware (MI250X with 2 GCDs)
comm_cus = list(range(55))      # GCD 0
compute_cus = list(range(55, 110))  # GCD 1
```

---

## ⚡ Common Patterns

### Pattern 1: AllGather + GEMM
```python
# Copy Engine approach
ag_streams = [torch.cuda.Stream(priority=-1) for _ in range(world_size)]

for peer in range(world_size):
    hip.hipMemcpyAsync(
        remote_bufs[peer].data_ptr(), local_buf.data_ptr(), nbytes,
        hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
        ag_streams[peer].cuda_stream
    )

gemm_kernel[grid](gathered_data, weight, output)
```

### Pattern 2: Pipelined Stages
```python
# CU Partition approach
stage_size = total_cus // 4
stages = [
    create_stream_with_cu_mask(create_cu_mask(
        list(range(i * stage_size, (i+1) * stage_size)), total_cus
    ))
    for i in range(4)
]

# Launch pipeline
for stage, stream in enumerate(stages):
    with torch.cuda.stream(StreamWrapper(stream)):
        stage_kernel[grid](...)
```

### Pattern 3: Producer-Consumer with Signals
```python
# Producer (communication)
for chunk_id in range(num_chunks):
    hip.hipMemcpyAsync(data_dst, data_src, chunk_size, ..., comm_stream)
    hip.hipMemcpyAsync(signal_dst, signal_src, 4, ..., comm_stream)  # Signal ready

# Consumer (computation)
@triton.jit
def consumer_kernel(data, signal, ...):
    # Wait for signal
    tl.wait_value_eq(signal + chunk_id, 1)
    # Process data
    ...
```

---

## 🐛 Debugging Checklist

- [ ] **Verify stream synchronization:** Use `stream.wait_stream()` correctly
- [ ] **Check CU mask validity:** No overlap, at least 1 CU per stream
- [ ] **Confirm NoCU flag:** `hipMemcpyDeviceToDeviceNoCU` (not just `DeviceToDevice`)
- [ ] **Profile overlap:** Use `torch.cuda.Event` to measure timing
- [ ] **Test sequential first:** Verify correctness before optimizing
- [ ] **Check memory bandwidth:** `rocprof --hsa-trace` to see if saturated
- [ ] **Validate masks:** Print enabled CUs to verify partitioning

---

## 📈 Profiling Commands

```bash
# Trace HIP API calls
rocprof --hip-trace python script.py

# Detailed statistics
rocprof --stats --hip-trace python script.py

# Kernel timeline
rocprof --hsa-trace python script.py

# Check concurrent execution
rocprof --hip-trace python script.py 2>&1 | grep -A 5 "Concurrent"
```

---

## 🔗 Quick Links

- **Detailed Copy Engine Guide:** `OVERLAP_COMM_COMPUTE_GUIDE.md`
- **Detailed CU Partitioning Guide:** `CU_PARTITIONING_GUIDE.md`
- **Comparison & Decision Guide:** `OVERLAP_STRATEGY_COMPARISON.md`
- **Example Tests:**
  - `test_overlap_comm_compute.py` (Copy Engine)
  - `test_cu_partitioning.py` (CU Partitioning)
- **Production Examples:**
  - `tutorials/09-AMD-overlapping-allgather-gemm.py`
  - `python/triton_dist/kernels/amd/all_gather_gemm.py`

---

## 🚀 Getting Started (30 seconds)

```python
from hip import hip
import torch

# 1. Create stream
comm_stream = torch.cuda.Stream(priority=-1)

# 2. Launch communication
hip.hipMemcpyAsync(
    dst_ptr, src_ptr, nbytes,
    hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,  # ← Add this flag!
    comm_stream.cuda_stream
)

# 3. Launch computation (overlaps automatically!)
my_triton_kernel[grid](...)

# 4. Synchronize
torch.cuda.synchronize()

# Done! You now have overlapped communication and computation! 🎉
```

---

**Questions?** Check the detailed guides or run the example tests!

