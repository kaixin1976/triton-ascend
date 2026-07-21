import argparse
import json
import os

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


def wait_files(prefix, rank_size, rank, phase, timeout_s=180):
    gin_runtime.rendezvous_barrier(prefix, rank_size, rank, phase, timeout_s)


def add_hex(result):
    for key in (
        "channel",
        "thread",
        "local_ccl_buffer_ptr",
        "remote_ccl_buffer_ptr",
    ):
        result[f"{key}_hex"] = f"0x{int(result[key]):x}"
    return result


def expected_payload(n, src_rank):
    return torch.arange(n, dtype=torch.float32) + float(src_rank * 1000)


def run_one(comm, args, src_rank, dst_rank, phase):
    x = expected_payload(args.n, args.rank).to("npu")
    y = torch.full((args.n,), -777.0, dtype=torch.float32, device="npu")
    torch.npu.synchronize()

    prefix = f"/tmp/triton_gin_hccl_rdma_p2p_{args.tag}"
    wait_files(prefix, args.rank_size, args.rank, f"{phase}_ready")
    result = gin_runtime.probe_hccl_rdma_p2p(
        comm,
        x,
        y,
        runtime_library=RUNTIME_LIB,
        hccl_library=args.hccl_lib,
        engine=args.hccl_channel_engine,
        src_rank=src_rank,
        dst_rank=dst_rank,
        nbytes=args.n * 4,
    )
    add_hex(result)
    print(f"HCCL_RDMA_P2P_JSON_{phase}=" + json.dumps(result, sort_keys=True), flush=True)
    torch.npu.synchronize()
    wait_files(prefix, args.rank_size, args.rank, f"{phase}_done")

    if args.rank == dst_rank:
        expected = expected_payload(args.n, src_rank)
        got = y.cpu()
        diff = (got - expected).abs()
        maxerr = float(diff.max()) if diff.numel() else 0.0
        bad_idx = int(diff.argmax()) if diff.numel() else -1
        print(
            f"rank={args.rank} phase={phase} src={src_rank} dst={dst_rank} "
            f"maxerr={maxerr} got_head={got[:min(8, args.n)].tolist()}",
            flush=True,
        )
        if maxerr != 0.0:
            raise AssertionError(
                f"RDMA P2P {phase} failed maxerr={maxerr} bad_idx={bad_idx} "
                f"got={float(got[bad_idx])} expected={float(expected[bad_idx])}"
            )
    wait_files(prefix, args.rank_size, args.rank, f"{phase}_checked")


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
    args = parser.parse_args()

    if args.rank_size != 2:
        raise ValueError("RDMA P2P smoke currently expects rank_size=2")
    if args.rank not in (0, 1):
        raise ValueError("RDMA P2P smoke currently expects ranks 0 and 1")

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

    success = False
    try:
        print(
            f"HCCL RDMA P2P config: rank={args.rank} rank_size={args.rank_size} "
            f"device={args.device} engine={args.hccl_channel_engine} "
            f"op_expansion_mode={args.hccl_op_expansion_mode} n={args.n}",
            flush=True,
        )
        run_one(comm, args, 0, 1, "rank0_to_rank1")
        run_one(comm, args, 1, 0, "rank1_to_rank0")
        print(f"HCCL RDMA P2P: PASS rank={args.rank}", flush=True)
        success = True
    finally:
        if success and skip_hccl_destroy():
            os._exit(0)
        if not skip_hccl_destroy():
            gin_runtime.destroy_hccl_comm(comm, hccl_library=args.hccl_lib)


if __name__ == "__main__":
    main()
