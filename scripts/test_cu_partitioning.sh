#!/bin/bash
################################################################################
# CU Partitioning Test - Unified Script
################################################################################
# Usage:
#   ./test_cu_partitioning.sh [ROWS] [COLS] [DTYPE] [NUM_PES]
#
# Examples:
#   ./test_cu_partitioning.sh 8192 8192 fp16 4    # 8k×8k FP16 with 4 PEs
#   ./test_cu_partitioning.sh 16384 16384 fp16 4  # 16k×16k FP16 with 4 PEs
#   ./test_cu_partitioning.sh 4096 4096 fp32 2    # 4k×4k FP32 with 2 PEs
#   ./test_cu_partitioning.sh                     # Use defaults (2k×2k FP32, 4 PEs)
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="${SCRIPT_DIR}/launch_amd.sh"
TEST_SCRIPT="${SCRIPT_DIR}/../python/triton_dist/test/amd/test_cu_partitioning.py"

# ========================================
# Parse command-line arguments
# ========================================
BUFFER_ROWS=${1:-2048}      # Default: 2k
BUFFER_COLS=${2:-2048}      # Default: 2k
DTYPE=${3:-fp32}            # Default: fp32
NUM_PES=${4:-4}             # Default: 4 PEs

# Validate dtype
if [[ "$DTYPE" != "fp16" && "$DTYPE" != "fp32" ]]; then
    echo "❌ Error: DTYPE must be 'fp16' or 'fp32', got: $DTYPE"
    exit 1
fi

# Validate NUM_PES (1-8)
if [[ $NUM_PES -lt 1 || $NUM_PES -gt 8 ]]; then
    echo "❌ Error: NUM_PES must be between 1 and 8, got: $NUM_PES"
    exit 1
fi

# ========================================
# Calculate memory requirements
# ========================================
NELEMS=$((BUFFER_ROWS * BUFFER_COLS))
BYTES_PER_ELEM=4
USE_FP16=0

if [[ "$DTYPE" == "fp16" ]]; then
    BYTES_PER_ELEM=2
    USE_FP16=1
fi

BUFFER_BYTES=$((NELEMS * BYTES_PER_ELEM))
BUFFER_MB=$((BUFFER_BYTES / 1024 / 1024))

# Each PE needs:
#   - Local buffer: BUFFER_BYTES
#   - Remote buffers: BUFFER_BYTES * NUM_PES (for AllGather)
# Total per PE: BUFFER_BYTES * (1 + NUM_PES)
# Add 2x safety margin for ROCm SHMEM overhead
REQUIRED_HEAP_MB=$((BUFFER_MB * (1 + NUM_PES) * 2))

# Round up to nearest GB, with minimum based on buffer size
HEAP_GB=$(( (REQUIRED_HEAP_MB + 1023) / 1024 ))
if [[ $HEAP_GB -lt 1 ]]; then
    HEAP_GB=1
fi

# For larger buffers (>= 64 MB), double the heap to avoid initialization hangs
if [[ $BUFFER_MB -ge 64 ]]; then
    HEAP_GB=$((HEAP_GB * 2))
fi

# For very large buffers (>= 256 MB), increase heap further
# Testing shows 16k×16k (512 MB) needs 16 GB heap
if [[ $BUFFER_MB -ge 256 ]]; then
    if [[ $HEAP_GB -lt 16 ]]; then
        HEAP_GB=16
    fi
fi

# For high PE counts (>= 6), increase heap proportionally
# Testing shows PE count is a major factor in heap requirements
if [[ $NUM_PES -ge 6 ]]; then
    # Scale heap by PE ratio: 8 PEs needs ~3x more than 2 PEs
    PE_MULTIPLIER=$(( (NUM_PES + 3) / 4 ))
    HEAP_GB=$((HEAP_GB * PE_MULTIPLIER))
    
    # Minimum 24 GB for 8 PEs
    if [[ $NUM_PES -ge 8 ]] && [[ $HEAP_GB -lt 24 ]]; then
        HEAP_GB=24
    fi
fi

HEAP_SIZE=$((HEAP_GB * 1024 * 1024 * 1024))

# ========================================
# Set environment variables
# ========================================
export CU_PART_BUFFER_ROWS=$BUFFER_ROWS
export CU_PART_BUFFER_COLS=$BUFFER_COLS
export CU_PART_USE_FP16=$USE_FP16

# ✅ CRITICAL: Must set ARNOLD_WORKER_GPU (not WORLD_SIZE) for launch_amd.sh
export ARNOLD_WORKER_GPU=$NUM_PES
export WORLD_SIZE=$NUM_PES
export LOCAL_WORLD_SIZE=$NUM_PES

# ROCm SHMEM heap size
export ROCSHMEM_HEAP_SIZE=$HEAP_SIZE

# Clear Python cache
find "${SCRIPT_DIR}/../python" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ========================================
# Print configuration
# ========================================
echo "========================================"
echo "CU Partitioning Benchmark"
echo "========================================"
echo "Configuration:"
echo "  Buffer shape: ${BUFFER_ROWS} × ${BUFFER_COLS}"
echo "  Data type: ${DTYPE} (${BYTES_PER_ELEM} bytes/elem)"
echo "  Elements: $(printf "%'d" $NELEMS)"
echo "  Buffer size: ${BUFFER_MB} MB per PE"
echo ""
echo "Distributed setup:"
echo "  Number of PEs: ${NUM_PES}"
echo "  Peers per PE: $((NUM_PES - 1))"
echo "  Total comm per PE: $((BUFFER_MB * (NUM_PES - 1))) MB"
echo ""
echo "Memory:"
echo "  Required heap: ~${REQUIRED_HEAP_MB} MB"
echo "  Allocated heap: ${HEAP_GB} GB"
echo "  (Set via ROCSHMEM_HEAP_SIZE)"
echo ""
echo "Environment:"
echo "  ARNOLD_WORKER_GPU=${ARNOLD_WORKER_GPU}"
echo "  WORLD_SIZE=${WORLD_SIZE}"
echo "  CU_PART_BUFFER_ROWS=${CU_PART_BUFFER_ROWS}"
echo "  CU_PART_BUFFER_COLS=${CU_PART_BUFFER_COLS}"
echo "  CU_PART_USE_FP16=${CU_PART_USE_FP16}"
echo ""

# Estimate timing (rough approximations)
if [[ $BUFFER_MB -lt 50 ]]; then
    echo "Expected timing (small buffers):"
    echo "  Standard streams: ~5-20 ms"
    echo "  CU-partitioned: ~2-10 ms"
elif [[ $BUFFER_MB -lt 200 ]]; then
    echo "Expected timing (medium buffers):"
    echo "  Standard streams: ~20-100 ms"
    echo "  CU-partitioned: ~10-50 ms"
else
    echo "Expected timing (large buffers):"
    echo "  Standard streams: ~100-500 ms"
    echo "  CU-partitioned: ~50-250 ms"
fi
echo ""

# ========================================
# Run the test
# ========================================
echo "Starting test..."
echo "----------------------------------------"

bash ${LAUNCH_SCRIPT} ${TEST_SCRIPT}

ret=$?
echo ""
echo "========================================"
if [ $ret -eq 0 ]; then
    echo "✅ Test completed successfully!"
    echo ""
    echo "Verify results:"
    echo "  - Check that standard vs CU-masked times differ"
    echo "  - CU-partitioned should show reduced CU contention"
    echo "  - All verification checks should pass"
else
    echo "❌ Test failed with exit code: $ret"
    echo ""
    echo "Common issues:"
    echo "  - ROCSHMEM_HEAP_SIZE too small (increase NUM_PES or reduce buffer)"
    echo "  - GPU memory exhausted (reduce buffer size)"
    echo "  - NCCL initialization hang (check device mapping)"
fi
echo "========================================"

exit $ret

