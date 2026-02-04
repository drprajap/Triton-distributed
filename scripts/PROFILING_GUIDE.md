# Profiling Guide for CU Partitioning Tests

## 🚀 Quick Start: Profile 8k×8k with 8 PEs

The easiest way to profile your specific configuration:

```bash
cd /dev/data/diprajap/workspace/rocm7/Triton-distributed

# Quick profiling (HIP trace, ~1 min)
bash scripts/profile_8k_8pe.sh quick

# Runtime profiling (recommended, ~2 min)
bash scripts/profile_8k_8pe.sh runtime

# Full profiling (with Perfetto, ~3-5 min)
bash scripts/profile_8k_8pe.sh full
```

---

## 📊 What Gets Profiled

### Test Configuration
- **Buffer size**: 8192×8192 FP16 (128 MB per PE)
- **Number of PEs**: 8
- **Total communication**: 896 MB per PE (7 peers × 128 MB)
- **CU allocation**: 32 comm + 272 compute (NOW FIXED! ✅)

### Profiling Modes

| Mode | What It Captures | Time | Use Case |
|------|------------------|------|----------|
| **quick** | HIP API calls | ~1 min | Quick check, API timing |
| **runtime** | HIP + kernels + memory ops | ~2 min | Standard analysis (recommended) |
| **full** | Everything + Perfetto timeline | ~3-5 min | Deep dive, visualization |

---

## 📂 Understanding Results

After profiling completes, results are in:
```
scripts/profiling_results/YYYYMMDD_HHMMSS/
```

### Files Generated

| File Type | Description | How to Use |
|-----------|-------------|------------|
| `*.log` | Test output with performance metrics | `grep Performance *.log` |
| `*.csv` | CSV trace data (HIP calls, kernel launches) | Open in spreadsheet or parse with scripts |
| `*.json` | JSON trace data | Parse programmatically |
| `*.db` | SQLite database | Query with SQL |
| `*.pftrace` | Perfetto timeline (full mode only) | Upload to https://ui.perfetto.dev |

---

## 🔍 Quick Analysis

### 1. Extract Performance Metrics

```bash
# Go to results directory
cd scripts/profiling_results/LATEST_DIR/

# View performance summary
grep -E "Performance Summary|Bandwidth|CU-partitioned|Speedup" *.log

# Should show:
#   Standard streams:      ~27-30 ms  (~30-33 GB/s per-PE)
#   CU-partitioned:        ~20 ms     (~44 GB/s per-PE)
#   Aggregate Bandwidth:   ~350 GB/s (8 PEs)
```

### 2. View HIP API Calls

```bash
# List all HIP API calls
head -30 *_hip_trace*.csv

# Count CU mask creation calls
grep -c "hipExtStreamCreateWithCUMask" *.csv

# Count memory copies
grep -c "hipMemcpyAsync" *.csv
```

### 3. Check CU Allocation

```bash
# Verify CU counts in log
grep "CU Partition" *.log

# Should show:
#   [PE 0] CU Partition: 32 comm, 272 compute  ✅ FIXED!
```

### 4. View Kernel Execution

```bash
# List GPU kernels (if kernel trace captured)
grep "kernel" *_kernel*.csv | head -20
```

---

## 🎯 Expected Results

### Performance Metrics

**Standard Streams (4 PEs):**
- Per-PE: ~30-33 GB/s
- Aggregate: ~229 GB/s
- Time: ~26-30 ms

**CU-Partitioned (8 PEs):**
- Per-PE: ~44 GB/s
- Aggregate: ~350 GB/s
- Time: ~20 ms
- **Speedup**: 1.3-1.5x

### CU Allocation (FIXED ✅)
- **Communication**: 32 CUs (odd indices: 1, 3, 5, ..., 63)
- **Computation**: 272 CUs (all remaining: 0, 2, 4, 6, ..., plus evens beyond 63)
- **Total**: 304 CUs (100% GPU utilization)

**Previous Bug**: Only 152 compute CUs were assigned (only even indices)  
**Fixed**: Now all 272 remaining CUs are assigned to computation

---

## 🐛 Troubleshooting

### No Trace Files Generated

**Symptom:**
```
Files found:
  HIP traces: 0
  HSA traces: 0
```

**Solutions:**

1. **Check rocprofv3 version:**
```bash
rocprofv3 --version
# Should be ROCm 7.x
```

2. **Try different mode:**
```bash
# If 'full' doesn't work, try 'runtime'
bash scripts/profile_8k_8pe.sh runtime
```

3. **Check permissions:**
```bash
ls -la scripts/profiling_results/
# Should be owned by your user
```

4. **Run without profiling first:**
```bash
# Make sure test works without profiling
bash scripts/test_cu_partitioning.sh 8192 8192 fp16 8
```

### Test Hangs During Profiling

**Symptom:** Profiling starts but never completes

**Solutions:**

1. **Check GPU availability:**
```bash
rocm-smi --showpids
# No other processes should be using GPUs
```

2. **Reduce buffer size:**
```bash
# Edit profile_8k_8pe.sh, change to 4k×4k
export CU_PART_BUFFER_ROWS=4096
export CU_PART_BUFFER_COLS=4096
```

3. **Reduce PEs:**
```bash
# Test with 4 PEs instead of 8
export WORLD_SIZE=4
export LOCAL_WORLD_SIZE=4
export ARNOLD_WORKER_GPU=4
```

### High Variance in Results

**Symptom:** Large differences between runs

**Possible causes:**
- Other processes on GPUs
- Thermal throttling
- NCCL initialization variance

**Solutions:**
1. Ensure GPUs are idle before profiling
2. Run multiple times and take average
3. Check GPU temperature with `rocm-smi`

---

## 📈 Advanced Analysis

### Using SQLite Database

If `*.db` files are generated:

```bash
sqlite3 full_full_results.db

# List tables
.tables

# Query API calls
SELECT name, COUNT(*) as count 
FROM rocpd_region 
GROUP BY name 
ORDER BY count DESC 
LIMIT 20;

# Exit
.quit
```

### Using Perfetto Timeline

If `*.pftrace` file is generated:

1. Go to https://ui.perfetto.dev
2. Click "Open trace file"
3. Select the `.pftrace` file
4. Explore timeline interactively

**What to look for:**
- Concurrent kernel execution (comm + compute overlap)
- Memory copy operations
- Synchronization points
- CU utilization patterns

---

## 🔧 Custom Profiling

For different configurations:

```bash
# Set your configuration
export CU_PART_BUFFER_ROWS=16384
export CU_PART_BUFFER_COLS=16384
export CU_PART_USE_FP16=1
export WORLD_SIZE=4
export LOCAL_WORLD_SIZE=4
export ARNOLD_WORKER_GPU=4
export ROCSHMEM_HEAP_SIZE=6GB

# Run profiling
bash scripts/profile_cu_partitioning.sh runtime
```

---

## 📝 Notes

### Recent Changes

**2024-12-17**: Fixed CU partitioning bug
- **Before**: Only 152 compute CUs assigned (even indices only)
- **After**: All 272 remaining CUs assigned to computation
- **Impact**: Better compute utilization, more accurate testing

### Profiling Overhead

Profiling adds overhead:
- **quick**: ~5-10% overhead
- **runtime**: ~10-20% overhead
- **full**: ~20-30% overhead

For accurate performance measurement without profiling:
```bash
bash scripts/test_cu_partitioning.sh 8192 8192 fp16 8
```

### Output Formats

ROCprofv3 generates multiple formats for flexibility:
- **CSV**: Easy to parse, open in Excel/LibreOffice
- **JSON**: Programmatic analysis with Python/scripts
- **DB**: SQL queries for complex analysis
- **Perfetto**: Visual timeline analysis

---

## 🆘 Getting Help

If profiling issues persist:

1. Check logs for errors:
```bash
cat scripts/profiling_results/LATEST_DIR/*.log | grep -i error
```

2. Run test without profiling:
```bash
bash scripts/test_cu_partitioning.sh 8192 8192 fp16 8
```

3. Check ROCm installation:
```bash
rocminfo | grep "Name:"
rocm-smi
```

4. Review main documentation:
- `scripts/CU_PARTITIONING_README.md`
- `scripts/BANDWIDTH_EXPLAINED.md`

---

**Last Updated**: 2024-12-17  
**Version**: 2.0 (with CU fix)




