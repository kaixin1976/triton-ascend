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


def wait_files(prefix, rank_size, rank, phase, timeout_s=240):
    gin_runtime.rendezvous_barrier(prefix, rank_size, rank, phase, timeout_s)


def tensor_payload(n, src_rank, dst_rank=0, scale=1.0):
    base = torch.arange(n, dtype=torch.float32)
    return base * scale + float(src_rank * 1000 + dst_rank * 100)


def check_close(name, got, expected, atol=0.0):
    diff = (got - expected).abs()
    maxerr = float(diff.max()) if diff.numel() else 0.0
    bad_idx = int(diff.argmax()) if diff.numel() else -1
    print(
        f"{name}: maxerr={maxerr} got_head={got.flatten()[:8].tolist()}",
        flush=True,
    )
    if maxerr > atol:
        raise AssertionError(
            f"{name} failed maxerr={maxerr} bad_idx={bad_idx} "
            f"got={float(got.flatten()[bad_idx])} expected={float(expected.flatten()[bad_idx])}"
        )


def trim_result(result):
    keys = (
        "rank",
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


def rdma_copy(comm, args, send_tensor, recv_tensor, src_rank, dst_rank, phase, metrics):
    prefix = f"/tmp/triton_gin_hccl_rdma_collective_{args.tag}"
    iteration = int(getattr(args, "iteration", 0))
    full_phase = f"iter{iteration}_{phase}"
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
        record["iteration"] = iteration
        record["phase"] = full_phase
        record["elapsed_ms"] = elapsed_ms
        metrics.append(record)
        print("RDMA_COPY_JSON=" + json.dumps(record, sort_keys=True), flush=True)


def exchange_vector(comm, args, local, peer_recv, phase, metrics):
    if args.rank_size != 2:
        raise ValueError("exchange_vector currently expects rank_size=2")
    rdma_copy(comm, args, local, peer_recv, 0, 1, f"{phase}_rank0_to_rank1", metrics)
    rdma_copy(comm, args, local, peer_recv, 1, 0, f"{phase}_rank1_to_rank0", metrics)


def alltoall_exchange(comm, args, send_matrix, recv_matrix, phase, metrics):
    if args.rank_size != 2:
        raise ValueError("alltoall_exchange currently expects rank_size=2")
    recv_matrix[args.rank].copy_(send_matrix[args.rank])
    torch.npu.synchronize()
    rdma_copy(
        comm,
        args,
        send_matrix[1],
        recv_matrix[0],
        0,
        1,
        f"{phase}_rank0_to_rank1",
        metrics,
    )
    rdma_copy(
        comm,
        args,
        send_matrix[0],
        recv_matrix[1],
        1,
        0,
        f"{phase}_rank1_to_rank0",
        metrics,
    )


def run_allgather(comm, args, metrics):
    local = tensor_payload(args.n, args.rank).to("npu")
    gathered = torch.full(
        (args.rank_size, args.n), -777.0, dtype=torch.float32, device="npu"
    )
    gathered[args.rank].copy_(local)
    peer_recv = gathered[1 - args.rank]
    torch.npu.synchronize()

    exchange_vector(comm, args, local, peer_recv, "allgather", metrics)

    expected = torch.stack(
        [tensor_payload(args.n, src) for src in range(args.rank_size)], dim=0
    )
    check_close("rdma_allgather", gathered.cpu(), expected)


def run_allreduce(comm, args, metrics):
    local = tensor_payload(args.n, args.rank).to("npu")
    peer = torch.full((args.n,), -777.0, dtype=torch.float32, device="npu")
    exchange_vector(comm, args, local, peer, "allreduce_exchange", metrics)
    out = local + peer
    torch.npu.synchronize()

    expected = sum(tensor_payload(args.n, src) for src in range(args.rank_size))
    check_close("rdma_allreduce_sum", out.cpu(), expected)


def run_alltoall(comm, args, metrics):
    send_rows = [
        tensor_payload(args.n, args.rank, dst_rank=dst)
        for dst in range(args.rank_size)
    ]
    send = torch.stack(send_rows, dim=0).to("npu")
    recv = torch.full(
        (args.rank_size, args.n), -777.0, dtype=torch.float32, device="npu"
    )

    alltoall_exchange(comm, args, send, recv, "alltoall", metrics)

    expected = torch.stack(
        [
            tensor_payload(args.n, src_rank=src, dst_rank=args.rank)
            for src in range(args.rank_size)
        ],
        dim=0,
    )
    check_close("rdma_alltoall", recv.cpu(), expected)


def run_allgather_scale(comm, args, metrics):
    local = (tensor_payload(args.n, args.rank) * 0.5).to("npu")
    gathered = torch.full(
        (args.rank_size, args.n), -777.0, dtype=torch.float32, device="npu"
    )
    gathered[args.rank].copy_(local)
    peer_recv = gathered[1 - args.rank]
    torch.npu.synchronize()

    exchange_vector(comm, args, local, peer_recv, "allgather_scale_comm", metrics)
    out = gathered.sum(dim=0) * 2.0 + float(args.rank)
    torch.npu.synchronize()

    expected_gathered = torch.stack(
        [tensor_payload(args.n, src) * 0.5 for src in range(args.rank_size)], dim=0
    )
    expected = expected_gathered.sum(dim=0) * 2.0 + float(args.rank)
    check_close("rdma_allgather_scale_fusion", out.cpu(), expected)


def run_allreduce_post(comm, args, metrics):
    local = tensor_payload(args.n, args.rank, scale=0.25).to("npu")
    peer = torch.full((args.n,), -777.0, dtype=torch.float32, device="npu")
    exchange_vector(comm, args, local, peer, "allreduce_post_comm", metrics)
    out = torch.relu(local + peer - 10.0) + float(args.rank)
    torch.npu.synchronize()

    expected = sum(
        tensor_payload(args.n, src, scale=0.25) for src in range(args.rank_size)
    )
    expected = torch.relu(expected - 10.0) + float(args.rank)
    check_close("rdma_allreduce_post_fusion", out.cpu(), expected)


def run_moe_dispatch_combine(comm, args, metrics):
    dispatch_rows = [
        tensor_payload(args.n, args.rank, dst_rank=expert, scale=0.125)
        for expert in range(args.rank_size)
    ]
    dispatch_send = torch.stack(dispatch_rows, dim=0).to("npu")
    expert_in = torch.full(
        (args.rank_size, args.n), -777.0, dtype=torch.float32, device="npu"
    )
    alltoall_exchange(comm, args, dispatch_send, expert_in, "moe_dispatch", metrics)

    expert_out = expert_in * 2.0 + float(args.rank * 10)
    return_send = torch.empty(
        (args.rank_size, args.n), dtype=torch.float32, device="npu"
    )
    for source in range(args.rank_size):
        return_send[source].copy_(expert_out[source])
    combined = torch.full(
        (args.rank_size, args.n), -777.0, dtype=torch.float32, device="npu"
    )
    alltoall_exchange(comm, args, return_send, combined, "moe_combine", metrics)

    expected = torch.stack(
        [
            tensor_payload(args.n, args.rank, dst_rank=expert, scale=0.125) * 2.0
            + float(expert * 10)
            for expert in range(args.rank_size)
        ],
        dim=0,
    )
    check_close("rdma_moe_dispatch_combine", combined.cpu(), expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rank-size", type=int, default=2)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--hccl-lib", default=HCCL_LIB)
    parser.add_argument("--hccl-op-expansion-mode", default="aicpu")
    parser.add_argument("--hccl-channel-engine", default="cpu_ts")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--case",
        choices=(
            "allgather",
            "allreduce",
            "alltoall",
            "allgather_scale",
            "allreduce_post",
            "moe",
            "all",
        ),
        default="all",
    )
    args = parser.parse_args()

    if args.rank_size != 2:
        raise ValueError("RDMA collective smoke currently expects rank_size=2")
    if args.rank not in (0, 1):
        raise ValueError("RDMA collective smoke currently expects ranks 0 and 1")
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
            f"HCCL RDMA collective config: rank={args.rank} rank_size={args.rank_size} "
            f"device={args.device} engine={args.hccl_channel_engine} "
            f"op_expansion_mode={args.hccl_op_expansion_mode} n={args.n} "
            f"case={args.case} repeat={args.repeat}",
            flush=True,
        )
        for iteration in range(args.repeat):
            args.iteration = iteration
            print(
                f"RDMA collective iteration={iteration} rank={args.rank} "
                f"case={args.case} n={args.n}",
                flush=True,
            )
            if args.case in ("allgather", "all"):
                run_allgather(comm, args, metrics)
            if args.case in ("allreduce", "all"):
                run_allreduce(comm, args, metrics)
            if args.case in ("alltoall", "all"):
                run_alltoall(comm, args, metrics)
            if args.case in ("allgather_scale", "all"):
                run_allgather_scale(comm, args, metrics)
            if args.case in ("allreduce_post", "all"):
                run_allreduce_post(comm, args, metrics)
            if args.case in ("moe", "all"):
                run_moe_dispatch_combine(comm, args, metrics)

        active = [m for m in metrics if args.rank in (m.get("src_rank"), m.get("dst_rank"))]
        total_ms = sum(float(m.get("elapsed_ms", 0.0)) for m in active)
        print(
            "RDMA_COLLECTIVE_SUMMARY_JSON="
            + json.dumps(
                {
                    "rank": args.rank,
                    "case": args.case,
                    "n": args.n,
                    "repeat": args.repeat,
                    "copies": len(active),
                    "copy_elapsed_ms_sum": total_ms,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print(f"HCCL RDMA collective: PASS rank={args.rank} case={args.case}", flush=True)
        success = True
    finally:
        if success and skip_hccl_destroy():
            os._exit(0)
        if not skip_hccl_destroy():
            gin_runtime.destroy_hccl_comm(comm, hccl_library=args.hccl_lib)


if __name__ == "__main__":
    main()
