# CU Partitioning Buffer Configuration Guide

## Overview

The `test_cu_partitioning.py` benchmark now supports configurable buffer sizes and data types via environment variables. This allows testing with different data transfer sizes without modifying the code.

---

## Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CU_PART_BUFFER_ROWS` | Number of rows in buffer | `1024` | `16384` |
| `CU_PART_BUFFER_COLS` | Number of columns in buffer | `1024` | `16384` |
| `CU_PART_USE_FP16` | Use FP16 instead of FP32 | `0` (FP32) | `1` (FP16) |

---

## Pre-configured Test Scripts

### 1. Default Configuration (1k×1k FP32)
```bash
bash scripts/launch_amd.sh python/triton_dist/test/amd/test_cu_partitioning.py
```

**Data size:**
- Elements: 1,048,576 (1M)
- Buffer per PE: **4 MB**
- Total comm per PE (4 PEs): **12 MB**

### 2. Large Buffer Configuration (16k×16k FP16)
```bash
bash scripts/test_cu_16k_fp16.sh
```

**Data size:**
- Elements: 268,435,456 (256M)
- Buffer per PE: **512 MB**
- Total comm per PE (4 PEs): **1,536 MB** (1.5 GB)

---

## Custom Configurations

### Example 1: 8k×8k FP32
```bash
export CU_PART_BUFFER_ROWS=8192
export CU_PART_BUFFER_COLS=8192
export CU_PART_USE_FP16=0
export WORLD_SIZE=4

bash scripts/launch_amd.sh python/triton_dist/test/amd/test_cu_partitioning.py
```

**Data size:** 256 MB per PE, 768 MB total comm per PE

### Example 2: 32k×32k FP16
```bash
export CU_PART_BUFFER_ROWS=32768
export CU_PART_BUFFER_COLS=32768
export CU_PART_USE_FP16=1
export WORLD_SIZE=4

bash scripts/launch_amd.sh python/triton_dist/test/amd/test_cu_partitioning.py
```

**Data size:** 2 GB per PE, 6 GB total comm per PE

### Example 3: Non-square Buffer (16k×8k FP16)
```bash
export CU_PART_BUFFER_ROWS=16384
export CU_PART_BUFFER_COLS=8192
export CU_PART_USE_FP16=1
export WORLD_SIZE=4

bash scripts/launch_amd.sh python/triton_dist/test/amd/test_cu_partitioning.py
```

**Data size:** 256 MB per PE, 768 MB total comm per PE

---

## Data Size Calculations

### Formula

```python
nelems = BUFFER_ROWS × BUFFER_COLS
bytes_per_elem = 2 if FP16 else 4
buffer_size_bytes = nelems × bytes_per_elem

# Communication per PE
peers = WORLD_SIZE - 1
total_comm_bytes = buffer_size_bytes × peers
```

### Common Configurations

| Config | Elements | Buffer Size | Comm per PE (4 PEs) | Comm per PE (8 PEs) |
|--------|----------|-------------|---------------------|---------------------|
| 1k×1k FP32 | 1M | 4 MB | 12 MB | 28 MB |
| 2k×2k FP32 | 4M | 16 MB | 48 MB | 112 MB |
| 4k×4k FP32 | 16M | 64 MB | 192 MB | 448 MB |
| 8k×8k FP32 | 64M | 256 MB | 768 MB | 1.75 GB |
| **16k×16k FP16** | **256M** | **512 MB** | **1.5 GB** | **3.5 GB** |
| 16k×16k FP32 | 256M | 1 GB | 3 GB | 7 GB |
| 32k×32k FP16 | 1G | 2 GB | 6 GB | 14 GB |

---

## Profiling with Custom Buffers

### Profile 16k×16k FP16 with rocprofv3

```bash
# Set buffer configuration
export CU_PART_BUFFER_ROWS=16384
export CU_PART_BUFFER_COLS=16384
export CU_PART_USE_FP16=1
export WORLD_SIZE=4

# Run profiling
bash scripts/profile_cu_partitioning.sh quick
```

The profiling output will show:
- Actual bandwidth achieved with large buffers
- Comparison between standard and CU-partitioned approaches
- HIP trace data for detailed analysis

---

## Memory Considerations

### GPU Memory Requirements

Each PE requires symmetric heap for:
- Local buffer: `buffer_size_bytes`
- Remote buffers: `buffer_size_bytes × WORLD_SIZE`
- **Total per PE:** `buffer_size_bytes × (WORLD_SIZE + 1)`

**Example for 16k×16k FP16 with 4 PEs:**
```
Local buffer:   512 MB
Remote buffers: 512 MB × 4 = 2 GB
Total per PE:   ~2.5 GB
```

### Adjusting ROCm SHMEM Heap Size

For large buffers, you may need to increase the symmetric heap:

```bash
# Set heap size (bytes)
export ROCSHMEM_SYMMETRIC_SIZE=$((8 * 1024 * 1024 * 1024))  # 8 GB

# Or use multiplier
export ROCSHMEM_HEAP_SIZE_MB=8192  # 8 GB
```

---

## Performance Expectations

### Expected Bandwidth Scaling

| Buffer Size | Expected Standard | Expected CU-Part | Notes |
|-------------|------------------|------------------|-------|
| 4 MB | 1-2 GB/s | 30-40 GB/s | Latency dominated |
| 64 MB | 5-15 GB/s | 80-120 GB/s | Bandwidth limited |
| 512 MB | 10-30 GB/s | 150-250 GB/s | Near peak bandwidth |
| 2 GB+ | 20-50 GB/s | 250-400 GB/s | Peak bandwidth |

**Note:** Actual bandwidth depends on:
- GPU memory bandwidth (MI300X: ~5.3 TB/s HBM)
- Number of concurrent PEs
- GPU topology (NVLink/Infinity Fabric)
- System memory contention

---

## Troubleshooting

### Out of Memory Error
```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce buffer size or number of PEs
2. Increase symmetric heap: `export ROCSHMEM_SYMMETRIC_SIZE=...`
3. Check available GPU memory: `rocm-smi --showmeminfo vram`

### Process Crash (SIGABRT)
```
Signal 6 (SIGABRT) received
```

**Solution:**
1. Reduce WORLD_SIZE (try 4 instead of 8)
2. Increase symmetric heap size
3. Check GPU memory availability

### Slow Performance
```
Bandwidth much lower than expected
```

**Solution:**
1. Verify CU partitioning is active (check log output)
2. Ensure proper GPU topology (check with `rocm-smi --showtopo`)
3. Try larger buffer sizes (smaller buffers are latency-dominated)

---

## Example Output

### 16k×16k FP16 Configuration

```
[PE 0] Buffer configuration:
  Shape: 16384 × 16384
  Elements: 268,435,456
  Data type: torch.float16
  Buffer size: 512.00 MB per PE
  Total comm per PE: 1536.00 MB (3 peers)

[PE 0] Test 1: Standard streams (potential CU contention)...
[PE 0] Standard streams time: 125.34 ms
[PE 0] Standard streams bandwidth: 12.25 GB/s

[PE 0] Test 2: CU-partitioned streams (no contention)...
[PE 0] CU-partitioned time: 6.89 ms
[PE 0] CU-partitioned bandwidth: 223.08 GB/s

Summary (PE 0):
  Standard streams:      125.34 ms  (12.25 GB/s)
  CU-partitioned:        6.89 ms    (223.08 GB/s)
  Speedup:               18.19x
```

---

## References

- Main test: `python/triton_dist/test/amd/test_cu_partitioning.py`
- Launch script: `scripts/launch_amd.sh`
- 16k FP16 helper: `scripts/test_cu_16k_fp16.sh`
- CU Partitioning Guide: `CU_PARTITIONING_GUIDE.md`

