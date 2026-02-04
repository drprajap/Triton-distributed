# CU Partitioning Mechanisms in ROCm Ecosystem

## Comparison: `hipExtStreamCreateWithCUMask` vs Environment Variables

### 🎯 The Short Answer

**No**, RCCL's `NCCL_MAX_NCHANNELS` and hipBLASLt's `TENSILE_STREAMK_MAX_CUS` do **NOT** use `hipExtStreamCreateWithCUMask` API. They use **different mechanisms** for CU partitioning:

| Library | Environment Variable | Mechanism | CU Control Level |
|---------|---------------------|-----------|------------------|
| **Our Implementation** | N/A (programmatic) | `hipExtStreamCreateWithCUMask` | **Explicit** (bit mask per stream) |
| **RCCL** | `NCCL_MAX_NCHANNELS` | Channel-based parallelism | **Implicit** (logical channels) |
| **hipBLASLt/Tensile** | `TENSILE_STREAMK_MAX_CUS` | Kernel launch configuration | **Kernel-level** (grid size limitation) |

---

## 🔍 Deep Dive: Different CU Partitioning Approaches

### 1. **`hipExtStreamCreateWithCUMask`** (What We Implemented)

#### **Mechanism:**
- **Explicit API call** to create streams bound to specific CUs
- Uses **hardware-level CU masking** at the HIP runtime level
- Each stream has a **bit mask** specifying which CUs it can use

#### **Implementation:**
```cpp
// Create bit mask
uint32_t cu_mask[10] = {0};
for (int i = 0; i < 30; i++) {  // Enable CUs 0-29
    cu_mask[i / 32] |= (1 << (i % 32));
}

// Create CU-masked stream
hipStream_t stream;
hipExtStreamCreateWithCUMask(&stream, mask_size, cu_mask);

// All kernels/operations on this stream use ONLY these CUs
hipMemcpyAsync(..., stream);  // Limited to CUs 0-29
my_kernel<<<grid, block, 0, stream>>>(...);  // Limited to CUs 0-29
```

#### **Characteristics:**
- ✅ **Explicit control**: You decide exact CU assignment
- ✅ **Stream-level**: All operations on stream use same CUs
- ✅ **Hardware enforced**: HSA runtime enforces CU mask
- ✅ **Fine-grained**: Per-CU control
- ❌ **Manual management**: Requires explicit API calls
- ❌ **Not portable**: AMD HIP-specific

#### **Use Cases:**
- Overlapping communication and computation
- Isolating critical workloads
- Fine-grained resource management
- Research and optimization

---

### 2. **RCCL's `NCCL_MAX_NCHANNELS`**

#### **Mechanism:**
- **Channel-based parallelism** for collective operations
- Creates multiple **logical communication channels**
- Each channel may use multiple threads/blocks
- Does **NOT** directly control CU assignment

#### **What It Actually Does:**
```cpp
// Conceptually (simplified):
// NCCL_MAX_NCHANNELS=8

for (int channel = 0; channel < 8; channel++) {
    // Each channel launches its own kernel(s)
    allReduceKernel<<<blocks, threads, 0, streams[channel]>>>(...);
    // ^ Standard streams, NO CU masking
}

// The GPU scheduler distributes work across available CUs
// No explicit CU assignment - relies on GPU's hardware scheduler
```

#### **How RCCL Uses Channels:**
```
┌─────────────────────────────────────────┐
│  NCCL_MAX_NCHANNELS=8                   │
├─────────────────────────────────────────┤
│                                         │
│  Channel 0: Stream0 → GPU Schedule      │
│  Channel 1: Stream1 → GPU Schedule      │
│  Channel 2: Stream2 → GPU Schedule      │
│  ...                                    │
│  Channel 7: Stream7 → GPU Schedule      │
│                                         │
│  Each stream launches kernels           │
│  GPU scheduler assigns to CUs           │
│  No explicit CU masks                   │
└─────────────────────────────────────────┘
```

#### **Characteristics:**
- ✅ **Automatic**: GPU hardware handles CU distribution
- ✅ **Portable**: Works across CUDA/ROCm
- ✅ **Load balanced**: Hardware scheduler optimizes
- ✅ **Simple**: Just set environment variable
- ❌ **No CU control**: Can't isolate specific CUs
- ❌ **Implicit behavior**: Harder to predict exactly which CUs used

#### **Environment Variable:**
```bash
export NCCL_MAX_NCHANNELS=8  # Use 8 communication channels
# More channels → More parallelism (up to HW limits)
```

#### **Performance Impact:**
- More channels → More concurrent work
- But: Diminishing returns beyond hardware capability
- Typical values: 4-16 channels

---

### 3. **hipBLASLt's `TENSILE_STREAMK_MAX_CUS`**

#### **Mechanism:**
- **Kernel-level CU limitation** for Stream-K GEMM algorithm
- Modifies **kernel launch configuration** (grid size)
- Does **NOT** use `hipExtStreamCreateWithCUMask`
- Limits number of **thread blocks** to fit on specified CUs

#### **What It Actually Does:**
```cpp
// Without TENSILE_STREAMK_MAX_CUS:
int max_blocks = calculate_optimal_blocks(...);  // Could be 1000s

// With TENSILE_STREAMK_MAX_CUS=50:
int max_cus = get_env("TENSILE_STREAMK_MAX_CUS");  // 50
int blocks_per_cu = 4;  // Typical occupancy
int max_blocks = min(calculate_optimal_blocks(...), max_cus * blocks_per_cu);
// max_blocks now limited to 200

// Launch kernel with limited blocks
gemm_kernel<<<max_blocks, threads>>>();
// ^ Standard stream, but fewer blocks = uses fewer CUs
```

#### **Stream-K Algorithm Context:**
```
Stream-K GEMM:
┌──────────────────────────────────────┐
│  Persistent kernel approach          │
│  - Threads persist across work items │
│  - Dynamic work distribution         │
│  - CU limit controls active blocks   │
└──────────────────────────────────────┘

TENSILE_STREAMK_MAX_CUS limits how many CUs participate
→ Other CUs remain available for other work!
```

#### **Characteristics:**
- ✅ **Application-level**: Limits kernel resource usage
- ✅ **Leaves CUs for other work**: Explicit resource reservation
- ✅ **Algorithm-specific**: Optimized for Stream-K
- ✅ **Simple**: Just set environment variable
- ❌ **Indirect control**: Limits via grid size, not CU mask
- ❌ **Algorithm-dependent**: Only works with Stream-K GEMM

#### **Environment Variable:**
```bash
export TENSILE_STREAMK_MAX_CUS=50  # Limit GEMM to 50 CUs
# Leaves remaining ~54 CUs (on MI300X) for other work
```

#### **Use Case:**
```
GPU with 304 CUs:
┌────────────────────────────────────────┐
│  GEMM (50 CUs)    │  Other Work       │
│  Stream-K limited │  Communication    │
│  by env var       │  or other kernels │
└────────────────────────────────────────┘
```

---

## 📊 Comparison Matrix

| Aspect | `hipExtStreamCreateWithCUMask` | RCCL `NCCL_MAX_NCHANNELS` | hipBLASLt `TENSILE_STREAMK_MAX_CUS` |
|--------|-------------------------------|---------------------------|-------------------------------------|
| **Control Level** | Hardware (CU mask) | Logical (channels) | Application (grid size) |
| **Granularity** | Per-CU | Per-channel | Per-kernel |
| **API** | Explicit HIP API | Environment variable | Environment variable |
| **Enforcement** | HSA runtime | GPU scheduler | Kernel launch logic |
| **Portability** | AMD HIP only | CUDA + ROCm | Tensile/hipBLASLt only |
| **Complexity** | High (manual) | Low (automatic) | Low (automatic) |
| **Precision** | Exact CUs | Approximate | Approximate |
| **Use Case** | Fine-grained isolation | Communication parallelism | GEMM resource limiting |

---

## 🔬 Under the Hood: How They Work

### **1. `hipExtStreamCreateWithCUMask`**

```
Application Layer:
  hipExtStreamCreateWithCUMask(stream, size, mask)
          ↓
HIP Runtime Layer:
  Stores CU mask in stream metadata
          ↓
HSA Runtime Layer:
  hsa_amd_queue_cu_set_mask(queue, mask)
          ↓
GPU Hardware:
  Wavefront scheduler only dispatches to enabled CUs
  ✅ Hardware-enforced isolation
```

### **2. RCCL `NCCL_MAX_NCHANNELS`**

```
Application Layer:
  ncclAllReduce(..., comm)
          ↓
RCCL Library:
  Creates N channels (N = NCCL_MAX_NCHANNELS)
  Each channel has standard HIP stream
  Launches kernels on all channels
          ↓
HIP Runtime:
  All streams use default CU mask (all CUs)
          ↓
GPU Hardware:
  Wavefront scheduler distributes work across ALL CUs
  ⚠️ No explicit CU isolation
```

### **3. `TENSILE_STREAMK_MAX_CUS`**

```
Application Layer:
  hipblasLtMatmul(...)
          ↓
hipBLASLt Library:
  Reads TENSILE_STREAMK_MAX_CUS env var
  Calculates max_blocks = MAX_CUS * blocks_per_cu
  Limits grid size
          ↓
Kernel Launch:
  my_gemm<<<limited_blocks, threads, 0, stream>>>()
  ^ Fewer blocks = naturally uses fewer CUs
          ↓
GPU Hardware:
  Scheduler assigns blocks to CUs
  Fewer blocks → fewer CUs used
  ⚠️ No hard CU isolation, just fewer active CUs
```

---

## 💡 Key Insights

### **Why RCCL Doesn't Use `hipExtStreamCreateWithCUMask`:**

1. **Portability**: RCCL needs to work on both CUDA and ROCm
   - `hipExtStreamCreateWithCUMask` is AMD-specific
   - Channel-based approach is portable

2. **Flexibility**: Hardware scheduler can adapt to workload
   - More efficient load balancing
   - Adapts to other concurrent work

3. **Simplicity**: Channels are a logical abstraction
   - Users don't need to know GPU CU count
   - Works across different AMD GPU generations

### **Why hipBLASLt Uses Environment Variable:**

1. **User control without API changes**
   - No need to modify application code
   - Easy to experiment with different values

2. **Algorithm-specific optimization**
   - Stream-K benefits from controlled block count
   - Leaves resources for overlap opportunities

3. **Compatibility**: Works with existing code
   - No new API learning required

---

## 🎯 When to Use Each Approach

### **Use `hipExtStreamCreateWithCUMask` When:**
- ✅ Need **explicit CU isolation**
- ✅ Building **custom overlap strategies**
- ✅ **Research** or **advanced optimization**
- ✅ AMD-specific code is acceptable
- ✅ Need **fine-grained control**

### **Use RCCL Channels When:**
- ✅ Need **portable** collective operations
- ✅ Want **automatic** CU utilization
- ✅ Using standard **communication patterns**
- ✅ Trust **hardware scheduler**

### **Use `TENSILE_STREAMK_MAX_CUS` When:**
- ✅ Using **hipBLASLt GEMM**
- ✅ Want to **leave CUs** for other work
- ✅ Overlapping **GEMM with communication**
- ✅ **Simple** resource reservation needed

---

## 🔧 Practical Example: Combining Approaches

```cpp
// Scenario: AllGather + GEMM overlap

// Option 1: Our explicit approach (fine-grained)
uint32_t comm_mask[10], compute_mask[10];
create_cu_masks(304, 0.3, comm_mask, compute_mask);

hipStream_t comm_stream, compute_stream;
hipExtStreamCreateWithCUMask(&comm_stream, 10, comm_mask);
hipExtStreamCreateWithCUMask(&compute_stream, 10, compute_mask);

// Communication uses ONLY comm_mask CUs
ncclAllGather(..., comm_stream);

// GEMM uses ONLY compute_mask CUs
hipblasLtMatmul(..., compute_stream);

// Option 2: Environment variable approach (simple)
export TENSILE_STREAMK_MAX_CUS=200  # ~66% of 304 CUs for GEMM
export NCCL_MAX_NCHANNELS=8         # RCCL uses remaining CUs

// Application code unchanged
ncclAllGather(...);  // RCCL uses ~100 CUs (implicit)
hipblasLtMatmul(...); // GEMM limited to 200 CUs

// Both run concurrently, GPU scheduler handles distribution
```

---

## 📚 Summary

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **`hipExtStreamCreateWithCUMask`** | Explicit, precise, hardware-enforced | Complex, AMD-only, manual | Research, custom optimization |
| **RCCL Channels** | Portable, automatic, simple | No CU control, implicit | Standard communication patterns |
| **`TENSILE_STREAMK_MAX_CUS`** | Simple, leaves resources, compatible | Indirect, algorithm-specific | GEMM + overlap scenarios |

**The key takeaway:** Different tools for different needs! Our `hipExtStreamCreateWithCUMask` implementation provides the **most explicit control**, while RCCL and hipBLASLt offer **simpler, higher-level** resource management through environment variables.

