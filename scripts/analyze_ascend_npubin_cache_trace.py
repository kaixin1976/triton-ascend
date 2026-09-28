#!/usr/bin/env python3
"""Summarize TRITON_ASCEND_NPUBIN_TIMING output from a workload."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


PREFIX = "[TRITON_ASCEND_NPUBIN_TIMING] "


def load_records(paths):
    records = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            offset = line.find(PREFIX)
            if offset < 0:
                continue
            record = json.loads(line[offset + len(PREFIX):])
            if record.get("stage") == "linalg_to_npubin_a2_a3":
                records.append(record)
    return records


def summarize(records):
    keyed = [record for record in records if record.get("content_cache_key")]
    keys = Counter(record["content_cache_key"] for record in keyed)
    hits = sum(bool(record.get("cache_hit")) for record in keyed)
    misses = len(keyed) - hits
    compiler_times = [record.get("interval_ms", {}).get("compiler_process") for record in keyed]
    compiler_times = [value for value in compiler_times if isinstance(value, (int, float))]
    median_compiler_ms = statistics.median(compiler_times) if compiler_times else 0.0
    kernels = Counter(record.get("kernel_name") or "<unknown>" for record in keyed)
    return {
        "npubin_requests": len(keyed),
        "unique_final_ir_keys": len(keys),
        "compiler_runs": misses,
        "content_cache_hits": hits,
        "equivalent_ir_duplicate_requests": len(keyed) - len(keys),
        "equivalent_ir_duplicate_rate": ((len(keyed) - len(keys)) / len(keyed) if keyed else 0.0),
        "measured_hit_rate": (hits / len(keyed) if keyed else 0.0),
        "median_external_compiler_ms": median_compiler_ms,
        "estimated_external_compile_ms_avoided": hits * median_compiler_ms,
        "requests_by_kernel": dict(sorted(kernels.items())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", help="stderr or JSONL timing trace")
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args(argv)
    report = summarize(load_records(args.trace))
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
