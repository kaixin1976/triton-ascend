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

: "${TRITON_GIN_RDMA_MASTER_ADDR:?set TRITON_GIN_RDMA_MASTER_ADDR to node0 host IP}"
: "${TRITON_GIN_NODE_RANK:?set TRITON_GIN_NODE_RANK, for example 0 on A2 and 1 on A3}"

NNODES="${TRITON_GIN_NNODES:-2}"
LOCAL_RANKS="${TRITON_GIN_LOCAL_RANKS:-8}"
NODE_RANK="${TRITON_GIN_NODE_RANK}"
MASTER_ADDR="${TRITON_GIN_RDMA_MASTER_ADDR}"
MASTER_PORT="${TRITON_GIN_RDMA_MASTER_PORT:-29940}"
RANK_SIZE=$((NNODES * LOCAL_RANKS))
BASE_RANK=$((NODE_RANK * LOCAL_RANKS))
BASE_TAG="${TRITON_GIN_RDMA_TAG:-rdma_layered16_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="/home/kaixin/triton_gin_rdma_layered16_${BASE_TAG}_node${NODE_RANK}"
HCCL_ENGINE="${TRITON_GIN_HCCL_CHANNEL_ENGINE:-cpu_ts}"
HCCL_OP_MODE="${TRITON_GIN_HCCL_OP_EXPANSION_MODE:-aicpu}"
N="${TRITON_GIN_RDMA_N:-128}"
REPEAT="${TRITON_GIN_RDMA_REPEAT:-1}"
mkdir -p "$LOG_ROOT"

if [ "$NNODES" -ne 2 ]; then
  echo "RDMA layered16 currently expects exactly 2 nodes; got ${NNODES}" >&2
  exit 1
fi
if [ "$LOCAL_RANKS" -ne 8 ]; then
  echo "RDMA layered16 expects 8 local ranks per node; got ${LOCAL_RANKS}" >&2
  exit 1
fi
if [ "$RANK_SIZE" -ne 16 ]; then
  echo "RDMA layered16 expects rank_size=16; got ${RANK_SIZE}" >&2
  exit 1
fi

export TRITON_ASCEND_GIN_HCCL_ROOT_HOST="$MASTER_ADDR"
export TRITON_ASCEND_GIN_HCCL_ROOT_PORT="$MASTER_PORT"
export TRITON_ASCEND_GIN_BARRIER_HOST="$MASTER_ADDR"
export TRITON_ASCEND_GIN_BARRIER_PORT="$((MASTER_PORT + 1))"
export TRITON_ASCEND_GIN_SKIP_HCCL_DESTROY="${TRITON_ASCEND_GIN_SKIP_HCCL_DESTROY:-1}"

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
echo "N=${N}"
echo "REPEAT=${REPEAT}"
echo "TRITON_ASCEND_GIN_RUNTIME_LIB=${TRITON_ASCEND_GIN_RUNTIME_LIB:-}"

pids=()
for local_rank in $(seq 0 $((LOCAL_RANKS - 1))); do
  global_rank=$((BASE_RANK + local_rank))
  python "${SCRIPT_DIR}/ascend_gin_hccl_rdma_layered_run.py" \
    --rank "$global_rank" --rank-size "$RANK_SIZE" --device "$local_rank" \
    --tag "$BASE_TAG" --hccl-channel-engine "$HCCL_ENGINE" \
    --hccl-op-expansion-mode "$HCCL_OP_MODE" \
    --node-rank "$NODE_RANK" --nnodes "$NNODES" --local-ranks "$LOCAL_RANKS" \
    --n "$N" --repeat "$REPEAT" \
    >"${LOG_ROOT}/rank${global_rank}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

for local_rank in $(seq 0 $((LOCAL_RANKS - 1))); do
  global_rank=$((BASE_RANK + local_rank))
  echo "---- rank${global_rank}.log ----"
  tail -n 160 "${LOG_ROOT}/rank${global_rank}.log" || true
done

python - "$LOG_ROOT" "$NODE_RANK" <<'PY'
import csv
import json
import re
import statistics
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
node_rank = sys.argv[2]
records = []
summaries = []
for log_path in sorted(log_root.glob("rank*.log")):
    rank_match = re.search(r"rank(\d+)\.log$", log_path.name)
    log_rank = int(rank_match.group(1)) if rank_match else -1
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "RDMA_LAYERED_COPY_JSON=" in line:
                payload = line.split("RDMA_LAYERED_COPY_JSON=", 1)[1].strip()
                rec = json.loads(payload)
                rec["log_rank"] = log_rank
                records.append(rec)
            if "RDMA_LAYERED_SUMMARY_JSON=" in line:
                payload = line.split("RDMA_LAYERED_SUMMARY_JSON=", 1)[1].strip()
                summaries.append(json.loads(payload))

csv_path = log_root / f"rdma_layered16_summary_node{node_rank}.csv"
md_path = log_root / f"rdma_layered16_summary_node{node_rank}.md"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "phase",
        "participants",
        "count",
        "avg_ms",
        "min_ms",
        "max_ms",
        "protocol",
        "engine",
        "first_error_statuses",
    ])
    grouped = {}
    for rec in records:
        phase = re.sub(r"^iter\d+_", "", rec.get("phase", ""))
        key = (
            phase,
            f"{rec.get('src_rank')}->{rec.get('dst_rank')}",
            rec.get("protocol_name"),
            rec.get("engine_name"),
        )
        grouped.setdefault(key, []).append(rec)
    for (phase, participants, protocol, engine), rows in sorted(grouped.items()):
        values = [float(r.get("elapsed_ms", 0.0)) for r in rows]
        errors = sorted({str(r.get("first_error_status")) for r in rows})
        writer.writerow([
            phase,
            participants,
            len(rows),
            f"{statistics.fmean(values):.6f}",
            f"{min(values):.6f}",
            f"{max(values):.6f}",
            protocol,
            engine,
            ";".join(errors),
        ])

with md_path.open("w", encoding="utf-8") as handle:
    handle.write(f"# RDMA layered16 summary node {node_rank}\n\n")
    handle.write(f"copy_records: {len(records)}\n\n")
    handle.write(f"rank_summaries: {len(summaries)}\n\n")
    handle.write("| phase | participants | count | avg ms | min ms | max ms | protocol | engine | first_error_statuses |\n")
    handle.write("| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |\n")
    grouped = {}
    for rec in records:
        phase = re.sub(r"^iter\d+_", "", rec.get("phase", ""))
        key = (
            phase,
            f"{rec.get('src_rank')}->{rec.get('dst_rank')}",
            rec.get("protocol_name"),
            rec.get("engine_name"),
        )
        grouped.setdefault(key, []).append(rec)
    for (phase, participants, protocol, engine), rows in sorted(grouped.items()):
        values = [float(r.get("elapsed_ms", 0.0)) for r in rows]
        errors = sorted({str(r.get("first_error_status")) for r in rows})
        handle.write(
            f"| {phase} | {participants} | {len(rows)} | "
            f"{statistics.fmean(values):.3f} | {min(values):.3f} | "
            f"{max(values):.3f} | {protocol} | {engine} | {';'.join(errors)} |\n"
        )

print(f"RDMA layered16 copy_records={len(records)}")
print(f"RDMA layered16 rank_summaries={len(summaries)}")
print(f"RDMA layered16 csv={csv_path}")
print(f"RDMA layered16 md={md_path}")
if not records or not summaries:
    raise SystemExit("missing RDMA layered16 records")
PY

if [ "$failed" -ne 0 ]; then
  echo "GIN HCCL RDMA layered16 node ${NODE_RANK}: FAILED; logs in ${LOG_ROOT}" >&2
  exit 1
fi

echo "GIN HCCL RDMA layered16 node ${NODE_RANK}: PASS"
