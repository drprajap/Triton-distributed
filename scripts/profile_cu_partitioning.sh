#!/bin/bash
################################################################################
# ROCProfV3 Profiling Script for CU Partitioning Test
# Updated for ROCm 7.x rocprofv3 API
#
# This script profiles the CU partitioning test using rocprofv3, which measures
# the performance of explicit CU resource partitioning with hipExtStreamCreateWithCUMask.
#
# The test is launched via launch_amd.sh to ensure proper environment setup:
# - ROCm SHMEM library paths
# - Python paths for triton_dist and pyrocshmem
# - MPI environment
# - AMD-specific environment variables
#
# Usage:
#   ./profile_cu_partitioning.sh [quick|detailed|runtime|full|bandwidth]
#
# Environment Variables:
#   WORLD_SIZE        - Number of processes (default: 4)
#   LOCAL_WORLD_SIZE  - Number of local processes (default: 4)
################################################################################

set -e

# Configuration - use environment variables if already set, otherwise use defaults
export CU_PART_BUFFER_ROWS=${CU_PART_BUFFER_ROWS:-16384}
export CU_PART_BUFFER_COLS=${CU_PART_BUFFER_COLS:-16384}
export CU_PART_USE_FP16=${CU_PART_USE_FP16:-1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="${SCRIPT_DIR}/launch_amd.sh"
TEST_SCRIPT="${SCRIPT_DIR}/../python/triton_dist/test/amd/test_cu_partitioning.py"
OUTPUT_DIR="${SCRIPT_DIR}/profiling_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}"

# Create output directory
mkdir -p "${RUN_DIR}"

echo "================================"
echo "ROCProfV3 Profiling Setup"
echo "================================"
echo "Output directory: ${RUN_DIR}"
echo ""

# Test configuration - use environment variables if already set
export WORLD_SIZE=${WORLD_SIZE:-8}
export LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-8}
export ARNOLD_WORKER_GPU=${ARNOLD_WORKER_GPU:-${WORLD_SIZE}}
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-24GB}

# Enable profiler compatibility mode - use regular streams instead of CU-masked streams
# CU-masked streams (hipExtStreamCreateWithCUMask) cause crashes with rocprof instrumentation
export ROCPROF_COMPAT_MODE=1

echo "⚠️  PROFILER COMPATIBILITY MODE ENABLED"
echo "   Using regular streams instead of CU-masked streams"
echo "   (CU masking will not be active during profiling)"
echo ""

# Function to run with specific profiling mode
run_profiling() {
    local mode=$1
    local output_prefix=$2
    local extra_args=$3
    
    echo "Running ${mode} profiling..."
    
    case ${mode} in
        "hip-trace")
            # HIP API trace - shows all HIP calls with timestamps
            rocprofv3 \
                --hip-trace \
                -d "${RUN_DIR}" \
                -o "${output_prefix}_hip_trace" \
                -f csv json \
                -- bash ${LAUNCH_SCRIPT} ${TEST_SCRIPT} 2>&1 | tee "${RUN_DIR}/${output_prefix}_hip_trace.log"
            ;;
        
        "kernel-trace")
            # Kernel execution trace - shows GPU kernel launches
            rocprofv3 \
                --kernel-trace \
                -d "${RUN_DIR}" \
                -o "${output_prefix}_kernel_trace" \
                -f csv json \
                -- bash ${LAUNCH_SCRIPT} ${TEST_SCRIPT} 2>&1 | tee "${RUN_DIR}/${output_prefix}_kernel_trace.log"
            ;;
        
        "runtime-trace")
            # Runtime trace - HIP runtime, memory operations, kernel dispatches
            rocprofv3 \
                -r \
                -d "${RUN_DIR}" \
                -o "${output_prefix}_runtime" \
                -f csv json \
                -- bash ${LAUNCH_SCRIPT} ${TEST_SCRIPT} 2>&1 | tee "${RUN_DIR}/${output_prefix}_runtime.log"
            ;;
        
        "sys-trace")
            # System trace - includes HIP, HSA, memory ops, kernels
            rocprofv3 \
                -s \
                -d "${RUN_DIR}" \
                -o "${output_prefix}_sys" \
                -f csv json \
                -- bash ${LAUNCH_SCRIPT} ${TEST_SCRIPT} 2>&1 | tee "${RUN_DIR}/${output_prefix}_sys.log"
            ;;
        
        "memory")
            # Memory operations trace with HIP
            rocprofv3 \
                --hip-trace \
                --kernel-trace \
                -d "${RUN_DIR}" \
                -o "${output_prefix}_memory" \
                -f csv json \
                -- bash ${LAUNCH_SCRIPT} ${TEST_SCRIPT} 2>&1 | tee "${RUN_DIR}/${output_prefix}_memory.log"
            ;;
        
        "full")
            # Complete profiling - sys-trace includes everything + explicit kernel traces
            rocprofv3 \
                --hip-trace \
                --hsa-trace \
                --kernel-trace \
                --memory-copy-trace \
                -d "${RUN_DIR}" \
                -o "${output_prefix}_full" \
                -f csv json pftrace \
                -- bash ${LAUNCH_SCRIPT} ${TEST_SCRIPT} 2>&1 | tee "${RUN_DIR}/${output_prefix}_full.log"
            ;;
        
        *)
            echo "Unknown profiling mode: ${mode}"
            exit 1
            ;;
    esac
    
    echo "✅ ${mode} profiling complete"
    echo ""
}

# Main profiling execution
echo "================================"
echo "Starting Profiling Session"
echo "================================"
echo ""

# Check if launch script exists
if [ ! -f "${LAUNCH_SCRIPT}" ]; then
    echo "❌ Error: launch_amd.sh not found at ${LAUNCH_SCRIPT}"
    exit 1
fi

# Check if test script exists
if [ ! -f "${TEST_SCRIPT}" ]; then
    echo "❌ Error: test script not found at ${TEST_SCRIPT}"
    exit 1
fi

# Check if rocprofv3 is available
if ! command -v rocprofv3 &> /dev/null; then
    echo "❌ Error: rocprofv3 not found!"
    echo "Please ensure ROCm 7.x is installed and rocprofv3 is in PATH"
    exit 1
fi

echo "Launch script: ${LAUNCH_SCRIPT}"
echo "Test script: ${TEST_SCRIPT}"
echo ""
echo "ROCProfiler version:"
rocprofv3 --version
echo ""

# Parse command line arguments
PROFILE_MODE=${1:-"quick"}

case ${PROFILE_MODE} in
    "quick")
        echo "Running quick profiling (HIP trace only)..."
        run_profiling "hip-trace" "quick"
        ;;
    
    "detailed")
        echo "Running detailed profiling (HIP + Kernel traces)..."
        run_profiling "hip-trace" "detailed"
        run_profiling "kernel-trace" "detailed"
        ;;
    
    "runtime")
        echo "Running runtime profiling (HIP runtime + memory ops + kernels)..."
        run_profiling "runtime-trace" "runtime"
        ;;
    
    "full")
        echo "Running full profiling (system-level trace with all details)..."
        run_profiling "full" "full"
        ;;
    
    "bandwidth")
        echo "Running bandwidth-focused profiling..."
        run_profiling "memory" "bandwidth"
        ;;
    
    *)
        echo "Usage: $0 [quick|detailed|runtime|full|bandwidth]"
        echo ""
        echo "Modes:"
        echo "  quick    - Fast HIP API trace"
        echo "  detailed - HIP + Kernel traces"
        echo "  runtime  - HIP runtime + memory ops + kernels"
        echo "  full     - Complete trace with HIP/HSA/Kernels/Memory (+ Perfetto timeline)"
        echo "  bandwidth - Memory-focused profiling"
        echo ""
        echo "Output Formats:"
        echo "  - CSV files for easy parsing"
        echo "  - JSON files for programmatic analysis"
        echo "  - SQLite DB for SQL queries"
        echo "  - Perfetto timeline (full mode only) for visualization"
        exit 1
        ;;
esac

# Generate summary
echo "================================"
echo "Profiling Complete!"
echo "================================"
echo ""
echo "Results saved to: ${RUN_DIR}"
echo ""
echo "Files generated:"
ls -lh "${RUN_DIR}"
echo ""
echo "To analyze results:"
echo "  1. View logs:"
echo "     cat ${RUN_DIR}/*.log"
echo ""
echo "  2. List generated files:"
echo "     ls -lh ${RUN_DIR}/"
echo "     # Should see: .csv, .json, .db, and .pftrace files"
echo ""
echo "  3. View CSV traces:"
echo "     cat ${RUN_DIR}/*.csv | head -20"
echo ""
echo "  4. Analyze with script:"
echo "     python ${SCRIPT_DIR}/analyze_cu_profiling.py ${RUN_DIR}"
echo ""
echo "  5. Open pftrace with Perfetto UI (full mode only):"
echo "     # Upload .pftrace file to: https://ui.perfetto.dev"
echo ""

# Create a summary file
cat > "${RUN_DIR}/README.md" << EOF
# CU Partitioning Profiling Results

**Timestamp:** ${TIMESTAMP}
**Mode:** ${PROFILE_MODE}
**World Size:** ${WORLD_SIZE}
**Local World Size:** ${LOCAL_WORLD_SIZE}
**Launch Script:** ${LAUNCH_SCRIPT}
**Test Script:** ${TEST_SCRIPT}

## Files

\`\`\`
$(ls -1 "${RUN_DIR}")
\`\`\`

## Quick Analysis

### View HIP Trace CSV
\`\`\`bash
cat *_hip_trace*.csv | head -20
\`\`\`

### View Kernel Trace
\`\`\`bash
cat *_kernel*.csv | head -20
\`\`\`

### Extract Timing Information
\`\`\`bash
grep -E "time|ms|bandwidth|Speedup" *.log
\`\`\`

## Performance Metrics

Extract from logs:
- CU-partitioned time
- Standard streams time
- Bandwidth measurements
- Speedup factors

## ROCProfV3 Output Formats

All profiling modes generate multiple output formats:

- **CSV**: Human-readable tables (.csv files) - easy to parse with scripts
- **JSON**: Structured data (.json files) - for programmatic analysis
- **SQLite**: Database format (.db files) - queryable with SQL
- **Perfetto** (full mode only): Timeline visualization (.pftrace files)
  - Upload to https://ui.perfetto.dev for interactive timeline view

### Analyzing Output Files

**CSV Files:**
\`\`\`bash
# View HIP API calls
head -50 *_hip_trace*.csv

# Count specific API calls
grep -c "hipExtStreamCreateWithCUMask" *.csv
grep -c "hipMemcpyAsync" *.csv
\`\`\`

**JSON Files:**
\`\`\`bash
# Pretty print JSON
python -m json.tool quick_hip_trace.json | head -100

# Extract specific fields
jq '.events[] | select(.name | contains("hipExtStream"))' quick_hip_trace.json
\`\`\`

**SQLite Database:**
\`\`\`bash
# Query the database (if sqlite3 is installed)
sqlite3 quick_hip_trace_results.db ".tables"
sqlite3 quick_hip_trace_results.db "SELECT name, COUNT(*) FROM rocpd_region GROUP BY name;"
\`\`\`

## Next Steps

1. Compare HIP trace between standard and CU-partitioned runs
2. Analyze memory bandwidth utilization
3. Check for concurrent kernel execution
4. Verify CU mask application via hipExtStreamCreateWithCUMask calls
EOF

echo "Summary written to: ${RUN_DIR}/README.md"
