#!/bin/bash
################################################################################
# Profile 8k×8k FP16 with 8 PEs
# Simplified profiling wrapper for CU partitioning test
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "========================================"
echo "Profile: 8k×8k FP16 with 8 PEs"
echo "========================================"
echo ""

# Configuration
export CU_PART_BUFFER_ROWS=8192
export CU_PART_BUFFER_COLS=8192
export CU_PART_USE_FP16=1
export WORLD_SIZE=8
export LOCAL_WORLD_SIZE=8
export ARNOLD_WORKER_GPU=8
# Profiling needs MORE heap due to instrumentation overhead
export ROCSHMEM_HEAP_SIZE=48GB

echo "Configuration:"
echo "  Buffer: ${CU_PART_BUFFER_ROWS}×${CU_PART_BUFFER_COLS}"
echo "  Data type: FP16"
echo "  Number of PEs: ${WORLD_SIZE}"
echo "  ROCm SHMEM heap: ${ROCSHMEM_HEAP_SIZE}"
echo ""

# Choose profiling mode
MODE=${1:-runtime}

case ${MODE} in
    quick)
        echo "Mode: Quick profiling (HIP trace only)"
        echo "  - Fast execution, low overhead"
        echo "  - Shows HIP API calls with timestamps"
        ;;
    runtime)
        echo "Mode: Runtime profiling (RECOMMENDED for 8 PEs)"
        echo "  - HIP runtime + memory ops + kernels"
        echo "  - Good balance of detail and speed"
        echo "  - Lower memory overhead than 'full'"
        ;;
    full)
        echo "Mode: Full profiling (most detailed, high memory usage)"
        echo "  - System-level trace with all instrumentation"
        echo "  - Includes Perfetto timeline"
        echo "  - Requires 48GB heap (double normal usage)"
        echo "  - May still fail with 8 PEs due to overhead"
        ;;
    *)
        echo "Usage: $0 [quick|runtime|full]"
        echo ""
        echo "Modes:"
        echo "  quick   - Fast HIP trace (recommended for quick checks)"
        echo "  runtime - Runtime trace with kernels (recommended for analysis)"
        echo "  full    - Complete trace with Perfetto (most detailed)"
        echo ""
        echo "Default: runtime"
        exit 1
        ;;
esac

echo ""
echo "Starting profiling in 3 seconds..."
sleep 1
echo "  3..."
sleep 1
echo "  2..."
sleep 1
echo "  1..."
echo ""

# Run profiling via the main profiling script
bash "${SCRIPT_DIR}/profile_cu_partitioning.sh" ${MODE}

RESULT=$?

echo ""
echo "========================================"
if [ $RESULT -eq 0 ]; then
    echo "✅ Profiling Complete!"
else
    echo "❌ Profiling Failed (exit code: $RESULT)"
fi
echo "========================================"
echo ""

# Show where results are
LATEST_DIR=$(ls -td ${SCRIPT_DIR}/profiling_results/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_DIR" ]; then
    echo "Results directory:"
    echo "  ${LATEST_DIR}"
    echo ""
    echo "Generated files:"
    ls -lh "${LATEST_DIR}" | tail -n +2 | awk '{printf "  %-10s  %s\n", $5, $9}'
    echo ""
    
    # Check for trace files
    CSV_COUNT=$(find "${LATEST_DIR}" -name "*.csv" 2>/dev/null | wc -l)
    JSON_COUNT=$(find "${LATEST_DIR}" -name "*.json" 2>/dev/null | wc -l)
    DB_COUNT=$(find "${LATEST_DIR}" -name "*.db" 2>/dev/null | wc -l)
    PFTRACE_COUNT=$(find "${LATEST_DIR}" -name "*.pftrace" 2>/dev/null | wc -l)
    
    echo "Trace files found:"
    echo "  CSV files:     $CSV_COUNT"
    echo "  JSON files:    $JSON_COUNT"
    echo "  DB files:      $DB_COUNT"
    echo "  Perfetto:      $PFTRACE_COUNT"
    echo ""
    
    if [ $CSV_COUNT -gt 0 ] || [ $JSON_COUNT -gt 0 ]; then
        echo "✅ Profiling data captured successfully!"
        
        # Check if test actually completed
        if grep -q "Test completed successfully" "${LATEST_DIR}"/*.log 2>/dev/null; then
            echo "✅ Test completed successfully during profiling"
        elif grep -q "SIGSEGV\|Aborted\|failed" "${LATEST_DIR}"/*.log 2>/dev/null; then
            echo ""
            echo "⚠️  Warning: Test crashed during profiling (likely memory exhaustion)"
            echo ""
            echo "Suggestions:"
            echo "  1. Try 'runtime' mode (lower overhead):"
            echo "     bash scripts/profile_8k_8pe.sh runtime"
            echo ""
            echo "  2. Profile with fewer PEs (4 instead of 8):"
            echo "     export WORLD_SIZE=4"
            echo "     export ARNOLD_WORKER_GPU=4"
            echo "     bash scripts/profile_cu_partitioning.sh full"
            echo ""
            echo "  3. Run without profiling to verify functionality:"
            echo "     bash scripts/test_cu_partitioning.sh 8192 8192 fp16 8"
            echo ""
        fi
        echo ""
        echo "Next steps:"
        echo "  1. View test output:"
        echo "     cat ${LATEST_DIR}/*.log | grep -E 'Performance|Bandwidth|Speedup'"
        echo ""
        echo "  2. Extract performance metrics:"
        echo "     cat ${LATEST_DIR}/*.log | grep 'CU-partitioned'"
        echo ""
        if [ $CSV_COUNT -gt 0 ]; then
            echo "  3. View HIP API calls:"
            echo "     head -30 ${LATEST_DIR}/*.csv"
            echo ""
        fi
        if [ $PFTRACE_COUNT -gt 0 ]; then
            echo "  4. Open Perfetto timeline:"
            echo "     Upload .pftrace file to: https://ui.perfetto.dev"
            echo ""
        fi
    else
        echo "⚠️  Warning: No CSV/JSON trace files found"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Check if rocprofv3 generated output:"
        echo "     ls -la ${LATEST_DIR}"
        echo ""
        echo "  2. Check log for errors:"
        echo "     cat ${LATEST_DIR}/*.log | grep -i error"
        echo ""
        echo "  3. Try with different mode:"
        echo "     bash scripts/profile_8k_8pe.sh quick"
        echo ""
    fi
fi

exit $RESULT

