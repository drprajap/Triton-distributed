#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"
LAUNCH_SCRIPT="${SCRIPT_DIR}/launch_amd.sh"
TUTORIAL="${PROJECT_DIR}/tutorials/09-AMD-overlapping-allgather-gemm.py"

OUT_ROOT="${1:-${PROJECT_DIR}/trace_outputs/ag_gemm_static_dynamic}"
M="${M:-8192}"
N="${N:-11008}"
K="${K:-4096}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
WARMUP="${WARMUP:-3}"
REPEATS="${REPEATS:-10}"
DYNAMIC_ACTIVE_CUS="${DYNAMIC_ACTIVE_CUS:-272}"
RDZV_PORT_BASE="${RDZV_PORT_BASE:-29600}"

mkdir -p "${OUT_ROOT}/static" "${OUT_ROOT}/dynamic"

# Default to two deterministic GPUs; caller can override.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4,5}"
export ARNOLD_WORKER_GPU="${ARNOLD_WORKER_GPU:-2}"

echo "Output root: ${OUT_ROOT}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
echo "Running static scheduling trace..."
ARNOLD_WORKER_0_PORT="${RDZV_PORT_BASE}" \
rocprofv3 --kernel-trace --hip-trace -f pftrace -d "${OUT_ROOT}/static" -- \
    "${LAUNCH_SCRIPT}" "${TUTORIAL}" \
    --M "${M}" --N "${N}" --K "${K}" \
    --chunk-size "${CHUNK_SIZE}" \
    --warmup "${WARMUP}" --repeats "${REPEATS}"

echo "Running dynamic scheduling trace..."
ARNOLD_WORKER_0_PORT="$((RDZV_PORT_BASE + 1))" \
rocprofv3 --kernel-trace --hip-trace -f pftrace -d "${OUT_ROOT}/dynamic" -- \
    "${LAUNCH_SCRIPT}" "${TUTORIAL}" \
    --M "${M}" --N "${N}" --K "${K}" \
    --chunk-size "${CHUNK_SIZE}" \
    --warmup "${WARMUP}" --repeats "${REPEATS}" \
    --num-sms-override "${DYNAMIC_ACTIVE_CUS}"

echo "Done. Perfetto traces under:"
echo "  ${OUT_ROOT}/static"
echo "  ${OUT_ROOT}/dynamic"
