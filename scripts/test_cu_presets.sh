#!/bin/bash
################################################################################
# CU Partitioning Test - Quick Presets
################################################################################
# This script provides convenient presets for common test configurations.
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_cu_partitioning.sh"

show_usage() {
    echo "Usage: $0 [PRESET]"
    echo ""
    echo "Available presets:"
    echo "  small     - 2k×2k FP32, 2 PEs  (32 MB, quick test)"
    echo "  medium    - 4k×4k FP16, 4 PEs  (64 MB)"
    echo "  large     - 8k×8k FP16, 4 PEs  (128 MB)"
    echo "  xlarge    - 16k×16k FP16, 4 PEs  (512 MB, bandwidth test)"
    echo "  xxlarge   - 32k×32k FP16, 4 PEs  (2 GB, stress test)"
    echo ""
    echo "Or use test_cu_partitioning.sh directly:"
    echo "  $TEST_SCRIPT [ROWS] [COLS] [DTYPE] [NUM_PES]"
    echo ""
    echo "Examples:"
    echo "  $0 large"
    echo "  $TEST_SCRIPT 8192 8192 fp16 4"
}

if [[ $# -eq 0 ]]; then
    show_usage
    exit 1
fi

PRESET=$1

case "$PRESET" in
    small)
        echo "Running SMALL preset: 2k×2k FP32, 2 PEs"
        bash "$TEST_SCRIPT" 2048 2048 fp32 2
        ;;
    medium)
        echo "Running MEDIUM preset: 4k×4k FP16, 4 PEs"
        bash "$TEST_SCRIPT" 4096 4096 fp16 4
        ;;
    large)
        echo "Running LARGE preset: 8k×8k FP16, 4 PEs"
        bash "$TEST_SCRIPT" 8192 8192 fp16 4
        ;;
    xlarge)
        echo "Running XLARGE preset: 16k×16k FP16, 2 PEs"
        echo "  (Using 2 PEs instead of 4 for stability at this scale)"
        bash "$TEST_SCRIPT" 16384 16384 fp16 2
        ;;
    xxlarge)
        echo "Running XXLARGE preset: 32k×32k FP16, 4 PEs"
        bash "$TEST_SCRIPT" 32768 32768 fp16 4
        ;;
    help|--help|-h)
        show_usage
        exit 0
        ;;
    *)
        echo "❌ Unknown preset: $PRESET"
        echo ""
        show_usage
        exit 1
        ;;
esac

