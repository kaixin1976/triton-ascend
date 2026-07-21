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
export HCCL_INDEPENDENT_OP="${HCCL_INDEPENDENT_OP:-1}"

if [ -z "${TRITON_ASCEND_GIN_RUNTIME_LIB:-}" ]; then
  gin_so="$(find /home/kaixin/triton-ascend/python/build -path '*/lib/gin/libtriton_ascend_gin_runtime.so' -print -quit 2>/dev/null || true)"
  if [ -n "$gin_so" ]; then
    export TRITON_ASCEND_GIN_RUNTIME_LIB="$gin_so"
  fi
fi
if [ -n "${TRITON_ASCEND_GIN_RUNTIME_LIB:-}" ]; then
  export LD_LIBRARY_PATH="$(dirname "$TRITON_ASCEND_GIN_RUNTIME_LIB"):${LD_LIBRARY_PATH:-}"
fi

: "${TRITON_GIN_RDMA_MASTER_ADDR:?set TRITON_GIN_RDMA_MASTER_ADDR to node0 HCCN/RDMA reachable host}"
: "${TRITON_GIN_NODE_RANK:?set TRITON_GIN_NODE_RANK, for example 0 on A2 and 1 on A3}"

NNODES="${TRITON_GIN_NNODES:-2}"
LOCAL_RANKS="${TRITON_GIN_LOCAL_RANKS:-8}"
NODE_RANK="${TRITON_GIN_NODE_RANK}"
MASTER_ADDR="${TRITON_GIN_RDMA_MASTER_ADDR}"
MASTER_PORT="${TRITON_GIN_RDMA_MASTER_PORT:-29630}"
RANK_SIZE=$((NNODES * LOCAL_RANKS))
BASE_RANK=$((NODE_RANK * LOCAL_RANKS))
BASE_TAG="${TRITON_GIN_RDMA_TAG:-rdma_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="/home/kaixin/triton_gin_rdma_${BASE_TAG}_node${NODE_RANK}"
HCCL_ENGINE="${TRITON_GIN_HCCL_CHANNEL_ENGINE:-aicpu}"
HCCL_OP_MODE="${TRITON_GIN_HCCL_OP_EXPANSION_MODE:-aicpu}"
mkdir -p "$LOG_ROOT"

export TRITON_ASCEND_GIN_HCCL_ROOT_HOST="$MASTER_ADDR"
export TRITON_ASCEND_GIN_HCCL_ROOT_PORT="$MASTER_PORT"
export TRITON_ASCEND_GIN_BARRIER_HOST="$MASTER_ADDR"
export TRITON_ASCEND_GIN_BARRIER_PORT="$((MASTER_PORT + 1))"
export TRITON_ASCEND_GIN_SKIP_HCCL_DESTROY="${TRITON_ASCEND_GIN_SKIP_HCCL_DESTROY:-1}"

run_local_ranks() {
  local name="$1"
  local script="$2"
  shift 2
  local tag="${BASE_TAG}_${name}"
  local log_dir="${LOG_ROOT}/${name}"
  mkdir -p "$log_dir"
  rm -f "/tmp/triton_gin_"*"${tag}"* 2>/dev/null || true
  rm -rf "/tmp/triton_gin_hccl_root_${tag}" 2>/dev/null || true

  echo "==== ${name} node=${NODE_RANK} ===="
  echo "script=${script}"
  echo "tag=${tag}"
  echo "rank_size=${RANK_SIZE} base_rank=${BASE_RANK} local_ranks=${LOCAL_RANKS}"

  local pids=()
  for local_rank in $(seq 0 $((LOCAL_RANKS - 1))); do
    local global_rank=$((BASE_RANK + local_rank))
    python "$script" \
      --rank "$global_rank" --rank-size "$RANK_SIZE" --device "$local_rank" --tag "$tag" "$@" \
      >"${log_dir}/rank${global_rank}.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done

  for local_rank in $(seq 0 $((LOCAL_RANKS - 1))); do
    local global_rank=$((BASE_RANK + local_rank))
    echo "---- rank${global_rank}.log ----"
    tail -n 100 "${log_dir}/rank${global_rank}.log" || true
  done
  if [ "$failed" -ne 0 ]; then
    echo "FAILED ${name}; logs in ${log_dir}" >&2
    exit 1
  fi
}

COMMON_HCCL_ARGS=(
  --backend hccl_channel
  --hccl-channel-engine "$HCCL_ENGINE"
  --hccl-op-expansion-mode "$HCCL_OP_MODE"
)

echo "BASE_TAG=${BASE_TAG}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "NNODES=${NNODES}"
echo "LOCAL_RANKS=${LOCAL_RANKS}"
echo "RANK_SIZE=${RANK_SIZE}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "HCCL_ENGINE=${HCCL_ENGINE}"
echo "HCCL_OP_MODE=${HCCL_OP_MODE}"
echo "TRITON_ASCEND_GIN_RUNTIME_LIB=${TRITON_ASCEND_GIN_RUNTIME_LIB:-}"

run_local_ranks rdma_channel_probe "${SCRIPT_DIR}/ascend_gin_hccl_channel_probe_run.py" \
  --hccl-channel-engine "$HCCL_ENGINE" --hccl-op-expansion-mode "$HCCL_OP_MODE" \
  --require-roce --require-acquire
run_local_ranks rdma_ops "${SCRIPT_DIR}/ascend_gin_op_coverage_run.py" \
  "${COMMON_HCCL_ARGS[@]}" --n 128 --block 32
run_local_ranks rdma_fused_allgather "${SCRIPT_DIR}/ascend_gin_fused_all_gather_run.py" \
  "${COMMON_HCCL_ARGS[@]}" --n 128 --block 32
run_local_ranks rdma_collective_patterns "${SCRIPT_DIR}/ascend_gin_collective_patterns_run.py" \
  "${COMMON_HCCL_ARGS[@]}" --case all --n 128 --block 32

echo "GIN HCCL RDMA node ${NODE_RANK}: PASS"
