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

BASE_TAG="${TRITON_GIN_SUITE_TAG:-suite_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="/home/kaixin/triton_gin_tutorial_suite_${BASE_TAG}"
mkdir -p "$LOG_ROOT"

run_pair() {
  local name="$1"
  local script="$2"
  shift 2
  local tag="${BASE_TAG}_${name}"
  local log_dir="${LOG_ROOT}/${name}"
  mkdir -p "$log_dir"
  rm -f "/tmp/triton_gin_"*"${tag}"* 2>/dev/null || true

  echo "==== ${name} ===="
  echo "script=${script}"
  echo "tag=${tag}"

  python "$script" --rank 0 --rank-size 2 --device 0 --tag "$tag" "$@" >"${log_dir}/rank0.log" 2>&1 &
  local pid0=$!
  python "$script" --rank 1 --rank-size 2 --device 1 --tag "$tag" "$@" >"${log_dir}/rank1.log" 2>&1 &
  local pid1=$!

  local status0=0
  local status1=0
  wait "$pid0" || status0=$?
  wait "$pid1" || status1=$?

  echo "rank0_status=${status0}"
  echo "rank1_status=${status1}"
  tail -n 80 "${log_dir}/rank0.log" || true
  tail -n 80 "${log_dir}/rank1.log" || true
  if [ "$status0" -ne 0 ] || [ "$status1" -ne 0 ]; then
    echo "FAILED ${name}; logs in ${log_dir}" >&2
    exit 1
  fi
}

TEST_DIR="${SCRIPT_DIR}"

echo "BASE_TAG=${BASE_TAG}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "TRITON_ASCEND_GIN_RUNTIME_LIB=${TRITON_ASCEND_GIN_RUNTIME_LIB:-}"

run_pair notify_wait_ops "${TEST_DIR}/ascend_gin_op_coverage_run.py" --backend tilexr --n 128 --block 32
run_pair allgather_gemm_pattern "${TEST_DIR}/ascend_gin_fused_all_gather_run.py" --backend tilexr --n 128 --block 32
run_pair gemm_fusion_patterns "${TEST_DIR}/ascend_gin_gemm_fusion_patterns_run.py" --case all --m 16 --n 16 --k 16
run_pair collective_patterns "${TEST_DIR}/ascend_gin_collective_patterns_run.py" --backend tilexr --case all --n 128 --block 32
run_pair moe_dispatch_combine "${TEST_DIR}/ascend_gin_moe_dispatch_combine_run.py" --n 128 --block 32

echo "GIN tutorial suite local2: PASS"
