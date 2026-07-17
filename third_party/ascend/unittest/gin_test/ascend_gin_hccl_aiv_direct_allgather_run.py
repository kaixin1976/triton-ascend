import argparse
import json
import os
import pathlib
import time

import torch
import torch_npu
import triton.language.extra.cann.gin_runtime as gin_runtime


RUNTIME_LIB = os.getenv("TRITON_ASCEND_GIN_RUNTIME_LIB")
HCCL_LIB = os.getenv(
    "TRITON_ASCEND_HCCL_LIB",
    "/home/kaixin/Ascend/cann-9.1.0-beta.1/aarch64-linux/lib64/libhcomm.so",
)


def wait_files(prefix, rank_size, rank, phase, timeout_s=180):
    marker = pathlib.Path(f"{prefix}.{phase}.{rank}")
    marker.write_text("ready")
    deadline = time.time() + timeout_s
    expected = [pathlib.Path(f"{prefix}.{phase}.{i}") for i in range(rank_size)]
    while time.time() < deadline:
        if all(p.exists() for p in expected):
            return
        time.sleep(0.05)
    missing = [str(p) for p in expected if not p.exists()]
    raise TimeoutError(f"barrier {phase} timeout, missing={missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rank-size", type=int, default=2)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--hccl-lib", default=HCCL_LIB)
    parser.add_argument("--op-expansion-mode", default="aiv")
    args = parser.parse_args()

    if args.rank_size < 2:
        raise ValueError("rank-size must be at least 2")

    os.environ.setdefault("HCCL_INDEPENDENT_OP", "1")
    torch.npu.set_device(args.device)

    comm = gin_runtime.create_hccl_root_info_comm(
        rank=args.rank,
        rank_size=args.rank_size,
        rendezvous_id=args.tag,
        hccl_library=args.hccl_lib,
        use_config=True,
        op_expansion_mode=args.op_expansion_mode,
    )
    gin = None
    prefix = f"/tmp/triton_gin_hccl_aiv_direct_allgather_{args.tag}"
    try:
        gin = gin_runtime.create_from_hccl_channel(
            comm,
            runtime_library=RUNTIME_LIB,
            hccl_library=args.hccl_lib,
            engine="aiv",
        )
        send = torch.arange(args.count, dtype=torch.float32, device="npu") + args.rank * 100.0
        recv = torch.full((args.rank_size * args.count,), -1.0, dtype=torch.float32, device="npu")

        wait_files(prefix, args.rank_size, args.rank, "ready")
        gin.hccl_aiv_allgather(send, recv, count=args.count, rendezvous_id=args.tag)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "done")

        expected = torch.cat(
            [torch.arange(args.count, dtype=torch.float32) + r * 100.0 for r in range(args.rank_size)]
        )
        got = recv.cpu()
        maxerr = float((got - expected).abs().max())
        result = {
            "rank": args.rank,
            "device": args.device,
            "count": args.count,
            "maxerr": maxerr,
            "send_ptr": int(send.data_ptr()),
            "recv_ptr": int(recv.data_ptr()),
        }
        print("TRITON_GIN_HCCL_AIV_DIRECT_ALLGATHER_JSON=" + json.dumps(result, sort_keys=True), flush=True)
        if maxerr != 0.0:
            raise AssertionError(result)
    finally:
        if gin is not None:
            gin.close()
        gin_runtime.destroy_hccl_comm(comm, hccl_library=args.hccl_lib)


if __name__ == "__main__":
    main()
