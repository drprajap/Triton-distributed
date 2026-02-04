# Explicit CU Partitioning with hipExtStreamCreateWithCUMask

This guide explains how to **explicitly partition Compute Units (CUs)** between communication and computation kernels using AMD's `hipExtStreamCreateWithCUMask` API.

## 🎯 Core Concept: Manual CU Assignment

Unlike using AMD Copy Engines (`hipMemcpyDeviceToDeviceNoCU`), this approach gives you **explicit control** over which CUs execute which kernels. This is useful when:

1. **Copy Engines are insufficient** for your communication bandwidth needs
2. You want **fine-grained control** over resource allocation
3. You're doing **compute-intensive communication** (e.g., compression, reduction)
4. You need to **isolate critical workloads** from interference

## 📋 Quick Reference

### Key API: `hipExtStreamCreateWithCUMask`

```cpp
hipError_t hipExtStreamCreateWithCUMask(
    hipStream_t* stream,       // [out] Pointer to the new stream
    uint32_t cuMaskSize,       // [in] Size of CU mask array (in uint32s)
    const uint32_t* cuMask     // [in] Bit mask specifying which CUs to use
);
```

**Python Wrapper:**

```python
from hip import hip
import ctypes

def create_stream_with_cu_mask(cu_mask: List[int]) -> int:
    """
    Create a HIP stream bound to specific CUs.
    
    Args:
        cu_mask: List of uint32 representing the CU bit mask
        
    Returns:
        hipStream_t handle (as integer)
    """
    mask_size = len(cu_mask)
    mask_array = (ctypes.c_uint32 * mask_size)(*cu_mask)
    
    # Python binding returns (hipError_t, hipStream_t) tuple
    err, stream = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)
    
    if err != hip.hipError_t.hipSuccess:
        raise RuntimeError(f"hipExtStreamCreateWithCUMask failed: {err}")
    
    return stream
```

### CU Mask Management

#### 1. Query Available CUs

```python
import torch

def get_num_cus() -> int:
    """Get total CUs on current GPU"""
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return device_props.multi_processor_count

# Example: AMD MI250X has 110 CUs per GCD
# Example: AMD MI300X has 304 CUs
num_cus = get_num_cus()
print(f"GPU has {num_cus} Compute Units")
```

#### 2. Create CU Masks

```python
def create_cu_mask(cu_list: List[int], total_cus: int) -> List[int]:
    """
    Create a bit mask from a list of CU indices.
    
    Args:
        cu_list: List of CU indices to enable (e.g., [0, 2, 4, 6])
        total_cus: Total number of CUs on the device
        
    Returns:
        List of uint32 representing the bit mask
    """
    # Each uint32 holds 32 bits (32 CUs)
    mask_size = (total_cus + 31) // 32
    cu_mask = [0] * mask_size
    
    # Set bits for specified CUs
    for cu_idx in cu_list:
        word_idx = cu_idx // 32   # Which uint32 in the array
        bit_idx = cu_idx % 32     # Which bit in that uint32
        cu_mask[word_idx] |= (1 << bit_idx)
    
    return cu_mask

# Example: Enable CUs 0, 1, 2, 3
cu_mask = create_cu_mask([0, 1, 2, 3], total_cus=110)
# Result: [0x0000000F, 0, 0, ...]  (first 4 bits set)
```

#### 3. Partition CUs

```python
def partition_cus(
    total_cus: int,
    comm_ratio: float = 0.3
) -> Tuple[List[int], List[int]]:
    """
    Partition CUs between communication and computation.
    
    Args:
        total_cus: Total number of CUs
        comm_ratio: Fraction for communication (e.g., 0.3 = 30%)
        
    Returns:
        (comm_cu_list, compute_cu_list)
    """
    num_comm_cus = max(1, int(total_cus * comm_ratio))
    num_compute_cus = total_cus - num_comm_cus
    
    # Strategy 1: Sequential allocation
    # comm_cus = list(range(num_comm_cus))
    # compute_cus = list(range(num_comm_cus, total_cus))
    
    # Strategy 2: Interleaved (better memory bandwidth distribution)
    comm_cus = [i for i in range(total_cus) if i % 2 == 1][:num_comm_cus]
    compute_cus = [i for i in range(total_cus) if i % 2 == 0][:num_compute_cus]
    
    # Strategy 3: Block-based (for NUMA awareness on MI250X)
    # half = total_cus // 2
    # comm_cus = list(range(num_comm_cus))
    # compute_cus = list(range(half, half + num_compute_cus))
    
    return comm_cus, compute_cus
```

#### 4. Debug/Visualize Masks

```python
def print_cu_mask(cu_mask: List[int], total_cus: int, name: str):
    """Print which CUs are enabled in a mask"""
    enabled_cus = []
    for word_idx, word in enumerate(cu_mask):
        for bit_idx in range(32):
            cu_idx = word_idx * 32 + bit_idx
            if cu_idx >= total_cus:
                break
            if word & (1 << bit_idx):
                enabled_cus.append(cu_idx)
    
    print(f"{name}: CUs {enabled_cus} (total: {len(enabled_cus)})")

# Example output:
# Communication CUs: CUs [1, 3, 5, 7, 9, ...] (total: 33)
# Computation CUs: CUs [0, 2, 4, 6, 8, ...] (total: 77)
```

## 🔧 Complete Workflow

### Step-by-Step Implementation

```python
import torch
import triton
import triton.language as tl
from hip import hip
import ctypes
from typing import List, Tuple

# ========================================
# Step 1: Query and Partition CUs
# ========================================
total_cus = torch.cuda.get_device_properties(0).multi_processor_count
print(f"Total CUs: {total_cus}")

# Allocate 30% for communication, 70% for computation
comm_cus, compute_cus = partition_cus(total_cus, comm_ratio=0.3)
print(f"Communication CUs: {len(comm_cus)}")
print(f"Computation CUs: {len(compute_cus)}")

# ========================================
# Step 2: Create CU Masks
# ========================================
comm_cu_mask = create_cu_mask(comm_cus, total_cus)
compute_cu_mask = create_cu_mask(compute_cus, total_cus)

print_cu_mask(comm_cu_mask, total_cus, "Comm")
print_cu_mask(compute_cu_mask, total_cus, "Compute")

# ========================================
# Step 3: Create CU-Masked Streams
# ========================================
comm_stream_handle = create_stream_with_cu_mask(comm_cu_mask)
compute_stream_handle = create_stream_with_cu_mask(compute_cu_mask)

print(f"Created comm stream: {hex(comm_stream_handle)}")
print(f"Created compute stream: {hex(compute_stream_handle)}")

# ========================================
# Step 4: Launch Kernels on Partitioned CUs
# ========================================

# 4a. Communication on comm CUs
for peer in range(num_peers):
    hip.hipMemcpyAsync(
        dst_ptr, src_ptr, nbytes,
        hip.hipMemcpyKind.hipMemcpyDeviceToDevice,  # Standard (uses CUs)
        comm_stream_handle,  # Stream bound to comm CUs
    )

# 4b. Computation on compute CUs
class StreamWrapper:
    """Wrapper to make HIP stream handle work with PyTorch"""
    def __init__(self, handle):
        self.cuda_stream = handle

with torch.cuda.stream(StreamWrapper(compute_stream_handle)):
    # Triton kernel runs on compute CUs
    my_triton_kernel[grid](input, output, ...)

# ========================================
# Step 5: Synchronize and Cleanup
# ========================================
torch.cuda.synchronize()

hip.hipStreamDestroy(comm_stream_handle)
hip.hipStreamDestroy(compute_stream_handle)
```

## 📊 Partitioning Strategies

### 1. Sequential Allocation

```python
# Communication: CUs 0-32
# Computation: CUs 33-109
comm_cus = list(range(33))
compute_cus = list(range(33, 110))
```

**Pros:** Simple, predictable
**Cons:** May not balance memory bandwidth

### 2. Interleaved Allocation

```python
# Communication: Odd CUs (1, 3, 5, ...)
# Computation: Even CUs (0, 2, 4, ...)
comm_cus = [i for i in range(total_cus) if i % 2 == 1]
compute_cus = [i for i in range(total_cus) if i % 2 == 0]
```

**Pros:** Better memory bandwidth distribution
**Cons:** May increase cross-CU communication overhead

### 3. Block-Based (NUMA-Aware)

```python
# For AMD MI250X with 2 GCDs (110 CUs each)
# GCD 0: CUs 0-54 (communication)
# GCD 1: CUs 55-109 (computation)
half = total_cus // 2
comm_cus = list(range(half))
compute_cus = list(range(half, total_cus))
```

**Pros:** NUMA-aware, minimizes cross-die traffic
**Cons:** Only works on multi-GCD GPUs

### 4. Ratio-Based Dynamic

```python
def adaptive_partition(workload_ratio: float):
    """Adjust CU allocation based on workload characteristics"""
    if workload_ratio > 2.0:  # Compute-heavy
        return partition_cus(total_cus, comm_ratio=0.2)
    elif workload_ratio < 0.5:  # Communication-heavy
        return partition_cus(total_cus, comm_ratio=0.4)
    else:  # Balanced
        return partition_cus(total_cus, comm_ratio=0.3)
```

## 🎯 Use Cases

### 1. AllGather + GEMM

```python
# 30% CUs for AllGather, 70% for GEMM
comm_cus, compute_cus = partition_cus(total_cus, 0.3)

# Communication stream
comm_stream = create_stream_with_cu_mask(create_cu_mask(comm_cus, total_cus))

# Computation stream
compute_stream = create_stream_with_cu_mask(create_cu_mask(compute_cus, total_cus))

# Launch AllGather on comm CUs
launch_allgather(local_data, comm_stream)

# Launch GEMM on compute CUs (overlaps!)
with torch.cuda.stream(StreamWrapper(compute_stream)):
    triton_gemm[grid](A, B, C, ...)
```

### 2. Pipelined Communication-Computation

```python
# 25% CUs for each stage in pipeline
stage_size = total_cus // 4

stage_cus = [
    list(range(i * stage_size, (i+1) * stage_size))
    for i in range(4)
]

# Create streams for each stage
streams = [
    create_stream_with_cu_mask(create_cu_mask(cus, total_cus))
    for cus in stage_cus
]

# Pipeline: Comm1 -> Compute1 -> Comm2 -> Compute2
launch_kernel(stage=0, stream=streams[0])  # Comm1
launch_kernel(stage=1, stream=streams[1])  # Compute1
launch_kernel(stage=2, stream=streams[2])  # Comm2
launch_kernel(stage=3, stream=streams[3])  # Compute2
```

### 3. Priority-Based Allocation

```python
# Critical path gets more CUs
critical_cus = list(range(80))      # 80 CUs for critical
background_cus = list(range(80, 110))  # 30 CUs for background

critical_stream = create_stream_with_cu_mask(
    create_cu_mask(critical_cus, total_cus)
)
background_stream = create_stream_with_cu_mask(
    create_cu_mask(background_cus, total_cus)
)
```

## ⚠️ Important Considerations

### 1. CU Mask Validity

```python
# ❌ BAD: Overlapping masks
comm_cus = [0, 1, 2, 3, 4]
compute_cus = [2, 3, 4, 5, 6]  # CUs 2,3,4 overlap!

# ✅ GOOD: Disjoint masks
comm_cus = [0, 1, 2, 3, 4]
compute_cus = [5, 6, 7, 8, 9]  # No overlap
```

**Impact of overlap:**
- **Undefined behavior**: Kernels may interfere
- **Performance degradation**: CU contention
- **Potential deadlocks**: Resource starvation

### 2. Minimum CU Allocation

```python
# Ensure each workload gets at least 1 CU
min_comm_cus = max(1, int(total_cus * comm_ratio))
min_compute_cus = max(1, total_cus - min_comm_cus)
```

### 3. Memory Bandwidth Awareness

```python
# On AMD MI250X: 110 CUs per GCD, 2 GCDs
# Strategy: Split by GCD for memory locality
gcd_0_cus = list(range(55))      # First GCD
gcd_1_cus = list(range(55, 110)) # Second GCD

# Assign communication to one GCD, computation to another
comm_stream = create_stream_with_cu_mask(create_cu_mask(gcd_0_cus, 110))
compute_stream = create_stream_with_cu_mask(create_cu_mask(gcd_1_cus, 110))
```

### 4. Stream Cleanup

```python
# Always destroy streams to prevent leaks
try:
    # ... use streams ...
finally:
    hip.hipStreamDestroy(comm_stream_handle)
    hip.hipStreamDestroy(compute_stream_handle)
```

## 📈 Performance Tuning

### Optimal CU Ratio

| Workload Type | Comm Ratio | Compute Ratio | Notes |
|---------------|-----------|---------------|-------|
| **GEMM-dominant** | 10-20% | 80-90% | Most time in computation |
| **AllReduce-heavy** | 30-40% | 60-70% | Balanced comm/compute |
| **AllGather + small GEMM** | 40-50% | 50-60% | Communication-bound |
| **Pipelined** | 25% each | 4 stages | Even distribution |

### Measuring Effectiveness

```python
import torch

def measure_cu_utilization(
    comm_stream_handle: int,
    compute_stream_handle: int,
):
    """Measure overlap efficiency"""
    
    # Events
    comm_start = torch.cuda.Event(enable_timing=True)
    comm_end = torch.cuda.Event(enable_timing=True)
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_end = torch.cuda.Event(enable_timing=True)
    
    # Record on different streams
    class StreamWrapper:
        def __init__(self, h):
            self.cuda_stream = h
    
    comm_wrapper = StreamWrapper(comm_stream_handle)
    compute_wrapper = StreamWrapper(compute_stream_handle)
    
    comm_start.record(comm_wrapper)
    # ... launch communication ...
    comm_end.record(comm_wrapper)
    
    compute_start.record(compute_wrapper)
    # ... launch computation ...
    compute_end.record(compute_wrapper)
    
    torch.cuda.synchronize()
    
    comm_time = comm_start.elapsed_time(comm_end)
    compute_time = compute_start.elapsed_time(compute_end)
    
    # Calculate overlap
    # Ideal: max(comm_time, compute_time)
    # Reality: May be higher due to memory bandwidth, etc.
    overlap_efficiency = min(comm_time, compute_time) / max(comm_time, compute_time)
    
    print(f"Communication time: {comm_time:.2f} ms")
    print(f"Computation time: {compute_time:.2f} ms")
    print(f"Overlap efficiency: {overlap_efficiency:.1%}")
    
    return overlap_efficiency
```

### Profiling with rocprof

```bash
# Profile CU utilization
rocprof --hip-trace --stats python test_cu_partitioning.py

# Look for:
# - CU utilization per kernel
# - Memory bandwidth utilization
# - Concurrent kernel execution
```

## 🆚 Comparison: Copy Engine vs. CU Partitioning

| Aspect | Copy Engine (`NoCU`) | CU Partitioning (`CUMask`) |
|--------|---------------------|---------------------------|
| **Hardware Used** | Dedicated DMA engines | Actual Compute Units |
| **CU Impact** | Zero (no CU usage) | Uses assigned CUs |
| **Bandwidth** | ~50-100 GB/s per copy engine | Full memory bandwidth |
| **Best For** | Simple memcpy operations | Compute-intensive comm |
| **Setup Complexity** | Low (just use `NoCU` flag) | Medium (need to manage masks) |
| **Flexibility** | Limited to memory copies | Can run any kernel |
| **Overlap Quality** | Perfect (separate hardware) | Good (if well-partitioned) |

**When to use Copy Engines (`NoCU`):**
- ✅ Simple AllGather/AllReduce with memcpy
- ✅ Background data movement
- ✅ Maximum simplicity

**When to use CU Partitioning (`CUMask`):**
- ✅ Compute-intensive communication (compression, reduction kernels)
- ✅ Need more bandwidth than copy engines provide
- ✅ Fine-grained control over resource allocation
- ✅ Custom communication patterns

## 🐛 Troubleshooting

### Issue 1: No Performance Improvement

**Symptoms:** CU partitioning doesn't improve performance over standard streams.

**Possible Causes:**
1. **Memory bandwidth bottleneck**: Both workloads saturate memory
2. **Overlapping masks**: CUs are shared between streams
3. **Insufficient workload size**: Overhead dominates

**Solutions:**
```python
# Check memory bandwidth usage
rocprof --hsa-trace python test.py

# Verify mask disjointness
def verify_disjoint(mask1, mask2, total_cus):
    enabled1 = set()
    enabled2 = set()
    for i in range(total_cus):
        word_idx, bit_idx = i // 32, i % 32
        if mask1[word_idx] & (1 << bit_idx):
            enabled1.add(i)
        if mask2[word_idx] & (1 << bit_idx):
            enabled2.add(i)
    
    overlap = enabled1 & enabled2
    if overlap:
        print(f"❌ Overlapping CUs: {overlap}")
        return False
    print(f"✅ Disjoint masks")
    return True
```

### Issue 2: Stream Creation Fails

**Error:** `hipExtStreamCreateWithCUMask` returns `hipErrorInvalidValue`

**Causes:**
1. Invalid mask size (must be `(total_cus + 31) // 32`)
2. Empty mask (no CUs enabled)
3. HIP version too old

**Solutions:**
```python
# Validate mask before creating stream
def validate_cu_mask(cu_mask, total_cus):
    expected_size = (total_cus + 31) // 32
    if len(cu_mask) != expected_size:
        raise ValueError(f"Mask size {len(cu_mask)} != expected {expected_size}")
    
    # Check if any CU is enabled
    any_enabled = any(word != 0 for word in cu_mask)
    if not any_enabled:
        raise ValueError("Empty mask: no CUs enabled")
    
    return True
```

### Issue 3: Triton Kernel Launch Fails

**Error:** Kernel doesn't run on CU-masked stream

**Solution:**
```python
# Wrap stream handle properly
class HIPStreamWrapper:
    def __init__(self, hip_stream_handle):
        self.cuda_stream = hip_stream_handle

# Use with torch.cuda.stream()
stream_wrapper = HIPStreamWrapper(cu_masked_stream_handle)
with torch.cuda.stream(stream_wrapper):
    triton_kernel[grid](...)
```

## 📚 Complete Example

See `test_cu_partitioning.py` for a full working example that includes:
- CU querying and partitioning
- Stream creation with CU masks
- Concurrent kernel launch
- Correctness verification
- Performance benchmarking

**Run it:**
```bash
cd /dev/data/diprajap/workspace/rocm7/Triton-distributed

WORLD_SIZE=2 LOCAL_WORLD_SIZE=2 \
python python/triton_dist/test/amd/test_cu_partitioning.py
```

## 🔗 References

- [HIP API Documentation - hipExtStreamCreateWithCUMask](https://rocmdocs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream.html)
- [AMD GPU Architecture](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [ROCm Programming Guide](https://rocm.docs.amd.com/)
- [Triton Documentation](https://triton-lang.org/)

