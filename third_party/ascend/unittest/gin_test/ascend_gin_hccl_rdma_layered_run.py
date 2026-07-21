import argparse
import json
import os
import time

import torch
import torch_npu
import triton.language.extra.cann.gin_runtime as gin_runtime


RUNTIME_LIB = os.getenv("TRITON_ASCEND_GIN_RUNTIME_LIB")
HCCL_LIB = os.getenv(
    "TRITON_ASCEND_HCCL_LIB",
    "/home/kaixin/Ascend/cann-9.1.0-beta.1/aarch64-linux/lib64/libhcomm.so",
)


def skip_hccl_destroy():
    value = os.getenv("TRITON_ASCEND_GIN_SKIP_HCCL_DESTROY", "")
    return value and value != "0"


def wait_files(prefix, rank_size, rank, phase, timeout_s=300):
    gin_runtime.rendezvous_barrier(prefix, rank_size, rank, phase, timeout_s)


def payload(n, rank, direction):
    return torch.arange(n, dtype=torch.float32) + float(rank * 1000 + direction * 100)


def check_exact(name, got, expected):
    diff = (got - expected).abs()
    maxerr = float(diff.max()) if diff.numel() else 0.0
    bad_idx = int(diff.argmax()) if diff.numel() else -1
    print(
        f"{name}: maxerr={maxerr} got_head={got.flatten()[:8].tolist()}",
        flush=True,
    )
    if maxerr != 0.0:
        raise AssertionError(
            f"{name} failed maxerr={maxerr} bad_idx={bad_idx} "
            f"got={float(got.flatten()[bad_idx])} "
            f"expected={float(expected.flatten()[bad_idx])}"
        )
    return maxerr


def trim_result(result):
    keys = (
        "rank",
        "nranks",
        "peer",
        "src_rank",
        "dst_rank",
        "engine_name",
        "thread_engine_name",
        "protocol_name",
        "bytes",
        "channel_acquire_ret",
        "channel_get_hccl_buffer_ret",
        "thread_acquire_ret",
        "local_copy_ret",
        "write_ret",
        "notify_ready_ret",
        "wait_ready_ret",
        "notify_done_ret",
        "wait_done_ret",
        "thread_sync_ret",
        "first_error_status",
    )
    return {key: result[key] for key in keys if key in result}


def rdma_pair_copy(comm, args, send_tensor, recv_tensor, src_rank, dst_rank, phase, metrics):
    prefix = f"/tmp/triton_gin_hccl_rdma_layered_{args.tag}"
    full_phase = f"iter{args.iteration}_{phase}"
    wait_files(prefix, args.rank_size, args.rank, f"{full_phase}_ready")
    start = time.perf_counter()
    result = gin_runtime.probe_hccl_rdma_p2p(
        comm,
        send_tensor,
        recv_tensor,
        runtime_library=RUNTIME_LIB,
        hccl_library=args.hccl_lib,
        engine=args.hccl_channel_engine,
        src_rank=src_rank,
        dst_rank=dst_rank,
        nbytes=args.n * 4,
    )
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    wait_files(prefix, args.rank_size, args.rank, f"{full_phase}_done")

    if args.rank in (src_rank, dst_rank):
        record = trim_result(result)
        record["iteration"] = args.iteration
        record["phase"] = full_phase
        record["elapsed_ms"] = elapsed_ms
        metrics.append(record)
        print("RDMA_LAYERED_COPY_JSON=" + json.dumps(record, sort_keys=True), flush=True)


def run_layered_peer_sweep(comm, args, metrics):
    if args.rank_size != args.nnodes * args.local_ranks:
        raise ValueError(
            f"rank_size={args.rank_size} does not match "
            f"nnodes*local_ranks={args.nnodes * args.local_ranks}"
        )
    if args.nnodes != 2:
        raise ValueError("layered smoke currently expects exactly 2 nodes")

    local_rank = args.rank % args.local_ranks
    remote_rank = (1 - args.node_rank) * args.local_ranks + local_rank

    send_forward = payload(args.n, args.rank, direction=0).to("npu")
    recv_forward = torch.full((args.n,), -777.0, dtype=torch.float32, device="npu")
    send_reverse = payload(args.n, args.rank, direction=1).to("npu")
    recv_reverse = torch.full((args.n,), -777.0, dtype=torch.float32, device="npu")
    torch.npu.synchronize()

    for pair_idx in range(args.local_ranks):
        src_rank = pair_idx
        dst_rank = args.local_ranks + pair_idx
        rdma_pair_copy(
            comm,
            args,
            send_forward,
            recv_forward,
            src_rank,
            dst_rank,
            f"pair{pair_idx}_node0_to_node1",
            metrics,
        )
        rdma_pair_copy(
            comm,
            args,
            send_reverse,
            recv_reverse,
            dst_rank,
            src_rank,
            f"pair{pair_idx}_node1_to_node0",
            metrics,
        )

    if args.node_rank == 0:
        expected = payload(args.n, remote_rank, direction=1)
        maxerr = check_exact(
            f"layered_pair{local_rank}_node1_to_node0_rank{args.rank}",
            recv_reverse.cpu(),
            expected,
        )
    else:
        expected = payload(args.n, remote_rank, direction=0)
        maxerr = check_exact(
            f"layered_pair{local_rank}_node0_to_node1_rank{args.rank}",
            recv_forward.cpu(),
            expected,
        )
    return maxerr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rank-size", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--hccl-lib", default=HCCL_LIB)
    parser.add_argument("--hccl-op-expansion-mode", default="aicpu")
    parser.add_argument("--hccl-channel-engine", default="cpu_ts")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=2)
    parser.add_argument("--local-ranks", type=int, default=8)
    parser.add_argument("--node-rank", type=int, required=True)
    args = parser.parse_args()

    if args.rank < 0 or args.rank >= args.rank_size:
        raise ValueError("rank must be in [0, rank-size)")
    if args.node_rank < 0 or args.node_rank >= args.nnodes:
        raise ValueError("node-rank must be in [0, nnodes)")
    expected_base = args.node_rank * args.local_ranks
    if args.rank < expected_base or args.rank >= expected_base + args.local_ranks:
        raise ValueError(
            f"rank={args.rank} is not in node_rank={args.node_rank} "
            f"range [{expected_base}, {expected_base + args.local_ranks})"
        )
    if args.repeat < 1:
        raise ValueError("repeat must be >= 1")

    os.environ.setdefault("HCCL_INDEPENDENT_OP", "1")
    torch.npu.set_device(args.device)
    comm = gin_runtime.create_hccl_root_info_comm(
        rank=args.rank,
        rank_size=args.rank_size,
        rendezvous_id=args.tag,
        hccl_library=args.hccl_lib,
        use_config=True,
        op_expansion_mode=args.hccl_op_expansion_mode,
    )

    metrics = []
    success = False
    try:
        print(
            f"HCCL RDMA layered config: rank={args.rank} rank_size={args.rank_size} "
            f"node_rank={args.node_rank} local_ranks={args.local_ranks} "
            f"device={args.device} engine={args.hccl_channel_engine} "
            f"op_expansion_mode={args.hccl_op_expansion_mode} n={args.n} "
            f"repeat={args.repeat}",
            flush=True,
        )
        maxerr = 0.0
        for iteration in range(args.repeat):
            args.iteration = iteration
            maxerr = max(maxerr, run_layered_peer_sweep(comm, args, metrics))

        active = [m for m in metrics if args.rank in (m.get("src_rank"), m.get("dst_rank"))]
        summary = {
            "rank": args.rank,
            "rank_size": args.rank_size,
            "node_rank": args.node_rank,
            "local_ranks": args.local_ranks,
            "n": args.n,
            "repeat": args.repeat,
            "copies": len(active),
            "maxerr": maxerr,
            "copy_elapsed_ms_sum": sum(float(m.get("elapsed_ms", 0.0)) for m in active),
            "route": "16-rank root-info comm; 8 cross-node ROCE/RDMA peer pairs",
            "engine": args.hccl_channel_engine,
        }
        print("RDMA_LAYERED_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
        print(f"HCCL RDMA layered: PASS rank={args.rank}", flush=True)
        success = True
    finally:
        if success and skip_hccl_destroy():
            os._exit(0)
        if not skip_hccl_destroy():
            gin_runtime.destroy_hccl_comm(comm, hccl_library=args.hccl_lib)


if __name__ == "__main__":
    main()
