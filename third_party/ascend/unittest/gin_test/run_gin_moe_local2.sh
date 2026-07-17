#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd /home/kaixin
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f /home/workspace/wrq/pkg/miniconda/etc/profile.d/conda.sh ]; then
  source /home/workspace/wrq/pkg/miniconda/etc/profile.d/conda.sh
fi
conda activate kaixin
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
source /home/kaixin/set_env.sh
export PYTHONPATH="/home/kaixin/triton-ascend/python:${SCRIPT_DIR}:${PYTHONPATH:-}"
export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"

if [ -z "${TRITON_ASCEND_GIN_RUNTIME_LIB:-}" ]; then
  gin_so="$(find /home/kaixin/triton-ascend/python/build -path '*/lib/gin/libtriton_ascend_gin_runtime.so' -print -quit 2>/dev/null || true)"
  if [ -n "$gin_so" ]; then
    export TRITON_ASCEND_GIN_RUNTIME_LIB="$gin_so"
  fi
fi
if [ -n "${TRITON_ASCEND_GIN_RUNTIME_LIB:-}" ]; then
  export LD_LIBRARY_PATH="$(dirname "$TRITON_ASCEND_GIN_RUNTIME_LIB"):${LD_LIBRARY_PATH:-}"
fi

TAG="${TRITON_GIN_MOE_TAG:-moe_$(date +%Y%m%d_%H%M%S)}"
N="${TRITON_GIN_MOE_N:-128}"
BLOCK="${TRITON_GIN_MOE_BLOCK:-32}"
LOG_DIR="/home/kaixin/triton_gin_moe_${TAG}"
mkdir -p "$LOG_DIR"
rm -f "/tmp/triton_gin_moe_${TAG}".*

echo "TAG=$TAG"
echo "LOG_DIR=$LOG_DIR"
echo "TRITON_ASCEND_GIN_RUNTIME_LIB=${TRITON_ASCEND_GIN_RUNTIME_LIB:-}"

python "${SCRIPT_DIR}/ascend_gin_moe_dispatch_combine_run.py" \
  --rank 0 --rank-size 2 --device 0 --tag "$TAG" --n "$N" --block "$BLOCK" \
  >"$LOG_DIR/rank0.log" 2>&1 &
pid0=$!

python "${SCRIPT_DIR}/ascend_gin_moe_dispatch_combine_run.py" \
  --rank 1 --rank-size 2 --device 1 --tag "$TAG" --n "$N" --block "$BLOCK" \
  >"$LOG_DIR/rank1.log" 2>&1 &
pid1=$!

status0=0
status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?

echo "rank0_status=$status0"
echo "rank1_status=$status1"
echo "---- rank0.log ----"
tail -n 120 "$LOG_DIR/rank0.log" || true
echo "---- rank1.log ----"
tail -n 120 "$LOG_DIR/rank1.log" || true

if [ "$status0" -ne 0 ] || [ "$status1" -ne 0 ]; then
  exit 1
fi
