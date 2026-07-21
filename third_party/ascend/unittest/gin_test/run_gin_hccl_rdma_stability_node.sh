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
LOCAL_RANKS="${TRITON_GIN_LOCAL_RANKS:-1}"
NODE_RANK="${TRITON_GIN_NODE_RANK}"
MASTER_ADDR="${TRITON_GIN_RDMA_MASTER_ADDR}"
MASTER_PORT_BASE="${TRITON_GIN_RDMA_MASTER_PORT:-29900}"
RANK_SIZE=$((NNODES * LOCAL_RANKS))
BASE_RANK=$((NODE_RANK * LOCAL_RANKS))
BASE_TAG="${TRITON_GIN_RDMA_TAG:-rdma_stability_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="/home/kaixin/triton_gin_rdma_stability_${BASE_TAG}_node${NODE_RANK}"
HCCL_ENGINE="${TRITON_GIN_HCCL_CHANNEL_ENGINE:-cpu_ts}"
HCCL_OP_MODE="${TRITON_GIN_HCCL_OP_EXPANSION_MODE:-aicpu}"
CASE="${TRITON_GIN_RDMA_CASE:-all}"
SHAPES_CSV="${TRITON_GIN_RDMA_SHAPES:-128,4096,65536}"
REPEAT="${TRITON_GIN_RDMA_REPEAT:-3}"
mkdir -p "$LOG_ROOT"

if [ "$RANK_SIZE" -ne 2 ]; then
  echo "RDMA stability currently expects exactly 2 ranks; got ${RANK_SIZE}" >&2
  exit 1
fi

echo "BASE_TAG=${BASE_TAG}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "NNODES=${NNODES}"
echo "LOCAL_RANKS=${LOCAL_RANKS}"
echo "RANK_SIZE=${RANK_SIZE}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT_BASE=${MASTER_PORT_BASE}"
echo "HCCL_ENGINE=${HCCL_ENGINE}"
echo "HCCL_OP_MODE=${HCCL_OP_MODE}"
echo "CASE=${CASE}"
echo "SHAPES=${SHAPES_CSV}"
echo "REPEAT=${REPEAT}"
echo "TRITON_ASCEND_GIN_RUNTIME_LIB=${TRITON_ASCEND_GIN_RUNTIME_LIB:-}"

IFS=',' read -r -a SHAPES <<< "$SHAPES_CSV"

shape_index=0
for n in "${SHAPES[@]}"; do
  n="$(echo "$n" | xargs)"
  if [ -z "$n" ]; then
    continue
  fi
  tag="${BASE_TAG}_n${n}"
  log_dir="${LOG_ROOT}/n${n}"
  mkdir -p "$log_dir"

  export TRITON_ASCEND_GIN_HCCL_ROOT_HOST="$MASTER_ADDR"
  export TRITON_ASCEND_GIN_HCCL_ROOT_PORT="$((MASTER_PORT_BASE + shape_index * 10))"
  export TRITON_ASCEND_GIN_BARRIER_HOST="$MASTER_ADDR"
  export TRITON_ASCEND_GIN_BARRIER_PORT="$((MASTER_PORT_BASE + shape_index * 10 + 1))"
  export TRITON_ASCEND_GIN_SKIP_HCCL_DESTROY="${TRITON_ASCEND_GIN_SKIP_HCCL_DESTROY:-1}"

  echo "==== rdma_stability n=${n} node=${NODE_RANK} tag=${tag} ===="
  echo "root_port=${TRITON_ASCEND_GIN_HCCL_ROOT_PORT} barrier_port=${TRITON_ASCEND_GIN_BARRIER_PORT}"

  pids=()
  for local_rank in $(seq 0 $((LOCAL_RANKS - 1))); do
    global_rank=$((BASE_RANK + local_rank))
    python "${SCRIPT_DIR}/ascend_gin_hccl_rdma_collective_run.py" \
      --rank "$global_rank" --rank-size "$RANK_SIZE" --device "$local_rank" \
      --tag "$tag" --hccl-channel-engine "$HCCL_ENGINE" \
      --hccl-op-expansion-mode "$HCCL_OP_MODE" \
      --case "$CASE" --n "$n" --repeat "$REPEAT" \
      >"${log_dir}/rank${global_rank}.log" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done

  for local_rank in $(seq 0 $((LOCAL_RANKS - 1))); do
    global_rank=$((BASE_RANK + local_rank))
    echo "---- n${n} rank${global_rank}.log ----"
    tail -n 160 "${log_dir}/rank${global_rank}.log" || true
  done
  if [ "$failed" -ne 0 ]; then
    echo "FAILED rdma_stability n=${n}; logs in ${log_dir}" >&2
    exit 1
  fi
  shape_index=$((shape_index + 1))
done

python - "$LOG_ROOT" "$NODE_RANK" <<'PY'
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
node_rank = sys.argv[2]
records = []
for log_path in sorted(log_root.glob("n*/rank*.log")):
    n_match = re.search(r"n(\d+)$", log_path.parent.name)
    n_value = int(n_match.group(1)) if n_match else -1
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            marker = "RDMA_COPY_JSON="
            if marker not in line:
                continue
            payload = line.split(marker, 1)[1].strip()
            try:
                rec = json.loads(payload)
            except json.JSONDecodeError:
                continue
            phase = rec.get("phase", "")
            phase = re.sub(r"^iter\d+_", "", phase)
            op = re.sub(r"_rank\d+_to_rank\d+$", "", phase)
            rec["n"] = n_value
            rec["op"] = op
            records.append(rec)

def percentile(values, pct):
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]

groups = {}
for rec in records:
    key = (rec.get("n"), rec.get("op"), rec.get("bytes"))
    groups.setdefault(key, []).append(float(rec.get("elapsed_ms", 0.0)))

csv_path = log_root / f"rdma_stability_summary_node{node_rank}.csv"
md_path = log_root / f"rdma_stability_summary_node{node_rank}.md"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["n", "bytes", "op", "count", "avg_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"])
    for (n, op, bytes_value), values in sorted(groups.items()):
        writer.writerow([
            n,
            bytes_value,
            op,
            len(values),
            f"{statistics.fmean(values):.6f}",
            f"{percentile(values, 50):.6f}",
            f"{percentile(values, 95):.6f}",
            f"{percentile(values, 99):.6f}",
            f"{min(values):.6f}",
            f"{max(values):.6f}",
        ])

with md_path.open("w", encoding="utf-8") as handle:
    handle.write(f"# RDMA stability summary node {node_rank}\n\n")
    handle.write(f"records: {len(records)}\n\n")
    handle.write("| N | bytes | op | count | avg ms | p50 ms | p95 ms | p99 ms | min ms | max ms |\n")
    handle.write("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for (n, op, bytes_value), values in sorted(groups.items()):
        handle.write(
            f"| {n} | {bytes_value} | {op} | {len(values)} | "
            f"{statistics.fmean(values):.3f} | {percentile(values, 50):.3f} | "
            f"{percentile(values, 95):.3f} | {percentile(values, 99):.3f} | "
            f"{min(values):.3f} | {max(values):.3f} |\n"
        )

print(f"RDMA stability records={len(records)}")
print(f"RDMA stability csv={csv_path}")
print(f"RDMA stability md={md_path}")
if not records:
    raise SystemExit("no RDMA_COPY_JSON records found")
PY

echo "GIN HCCL RDMA stability node ${NODE_RANK}: PASS"
