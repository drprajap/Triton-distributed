#!/bin/bash

# Test CU Partitioning with Manual Instrumentation
# No profiler needed - timing is done with HIP events
# ============================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}/.."

# Get parameters (same as test_cu_partitioning.sh)
ROWS=${1:-8192}
COLS=${2:-8192}
DTYPE=${3:-fp16}
NUM_PES=${4:-4}

echo "============================================================"
echo "CU Partitioning Test - Manual Instrumentation"
echo "============================================================"
echo "Buffer size: ${ROWS}×${COLS}"
echo "Data type: ${DTYPE}"
echo "PEs: ${NUM_PES}"
echo "============================================================"
echo

# Export configuration
export CU_PART_BUFFER_ROWS=${ROWS}
export CU_PART_BUFFER_COLS=${COLS}
export CU_PART_USE_FP16=$( [[ "${DTYPE}" == "fp16" ]] && echo 1 || echo 0 )

# Calculate heap size (same logic as main script)
BUFFER_SIZE_MB=$(( ROWS * COLS * 2 / 1024 / 1024 ))
BASE_HEAP_GB=2

if [ "${BUFFER_SIZE_MB}" -ge 256 ]; then
    BASE_HEAP_GB=16
elif [ "${BUFFER_SIZE_MB}" -ge 64 ]; then
    BASE_HEAP_GB=4
fi

if [ "${NUM_PES}" -ge 6 ]; then
    HEAP_GB=$(( BASE_HEAP_GB * 3 ))
    if [ "${NUM_PES}" -ge 8 ]; then
        HEAP_GB=24
    fi
else
    HEAP_GB=${BASE_HEAP_GB}
fi

export ROCSHMEM_HEAP_SIZE="${HEAP_GB}GB"
export WORLD_SIZE=${NUM_PES}
export LOCAL_WORLD_SIZE=${NUM_PES}
export ARNOLD_WORKER_GPU=${NUM_PES}

echo "ROC SHMEM Heap: ${ROCSHMEM_HEAP_SIZE}"
echo

# Run with launch script
LAUNCH_SCRIPT="scripts/launch_amd.sh"
TEST_SCRIPT="python/triton_dist/test/amd/test_cu_partitioning_instrumented.py"

if [ ! -f "${LAUNCH_SCRIPT}" ]; then
    echo "❌ Error: ${LAUNCH_SCRIPT} not found"
    exit 1
fi

if [ ! -f "${TEST_SCRIPT}" ]; then
    echo "❌ Error: ${TEST_SCRIPT} not found"
    exit 1
fi

echo "Running instrumented test (no profiler overhead)..."
echo
timeout 300 bash "${LAUNCH_SCRIPT}" "${TEST_SCRIPT}" || {
    echo "❌ Test failed or timed out"
    exit 1
}

echo
echo "============================================================"
echo "✅ Test complete!"
echo "============================================================"




