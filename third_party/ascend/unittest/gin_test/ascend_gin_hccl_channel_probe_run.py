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
SIGNAL_SLOTS = 128


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


def add_hex_masks(result):
    for key in (
        "peer_mask",
        "protocol_mask",
        "hccs_peer_mask",
        "roce_peer_mask",
        "pcie_peer_mask",
        "ubc_ctp_peer_mask",
        "ubc_tp_peer_mask",
        "ub_mem_peer_mask",
        "acquire_peer_mask",
        "ccl_buffer_ptr",
        "aiv_comm_info_ptr",
        "aiv_comm_info_mem_handle",
        "aiv_comm_info_local_gm_in",
        "aiv_comm_info_local_gm_out",
        "aiv_comm_info_local_flag_base",
        "aiv_descriptor_via_cpu_first_channel",
        "aiv_descriptor_via_cpu_acquire_peer_mask",
        "aiv_descriptor_via_cpu_hccl_buffer_peer_mask",
        "aiv_descriptor_via_cpu_remote_mem_peer_mask",
        "aiv_descriptor_via_cpu_readback_gm_in_mask",
        "aiv_descriptor_via_cpu_readback_gm_out_mask",
        "aiv_descriptor_via_cpu_symmetric_gm_out_peer_mask",
    ):
        result[f"{key}_hex"] = f"0x{int(result[key]):x}"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rank-size", type=int, default=2)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--hccl-lib", default=HCCL_LIB)
    parser.add_argument("--hccl-op-expansion-mode", default="aiv")
    parser.add_argument("--hccl-channel-engine", default="aiv")
    parser.add_argument("--no-acquire", action="store_true")
    parser.add_argument("--add-exchange-info", action="store_true")
    parser.add_argument("--require-ub-mem", action="store_true")
    parser.add_argument("--require-acquire", action="store_true")
    args = parser.parse_args()

    if args.rank_size < 2:
        raise ValueError("rank-size must be at least 2")
    if args.rank < 0 or args.rank >= args.rank_size:
        raise ValueError("rank must be in [0, rank-size)")

    os.environ.setdefault("HCCL_INDEPENDENT_OP", "1")
    if args.no_acquire:
        os.environ["TRITON_ASCEND_GIN_HCCL_PROBE_ACQUIRE"] = "0"
    if args.add_exchange_info:
        os.environ["TRITON_ASCEND_GIN_HCCL_PROBE_ADD_EXCHANGE_INFO"] = "1"

    torch.npu.set_device(args.device)
    mode_value = gin_runtime.hccl_op_expansion_mode_value(args.hccl_op_expansion_mode)
    engine_value = gin_runtime.hccl_channel_engine_value(args.hccl_channel_engine)
    capability = gin_runtime.hccl_comm_config_capability(hccl_library=args.hccl_lib)
    capability_text = "none" if capability is None else f"0x{capability:x}"
    print(
        f"HCCL channel probe config: rank={args.rank} rank_size={args.rank_size} "
        f"device={args.device} op_expansion_mode={args.hccl_op_expansion_mode} "
        f"mode_value={mode_value} channel_engine={args.hccl_channel_engine} "
        f"engine_value={engine_value} capability={capability_text}",
        flush=True,
    )

    comm = gin_runtime.create_hccl_root_info_comm(
        rank=args.rank,
        rank_size=args.rank_size,
        rendezvous_id=args.tag,
        hccl_library=args.hccl_lib,
        use_config=True,
        op_expansion_mode=args.hccl_op_expansion_mode,
    )

    prefix = f"/tmp/triton_gin_hccl_channel_probe_{args.tag}"
    try:
        result = gin_runtime.probe_hccl_channel(
            comm,
            runtime_library=RUNTIME_LIB,
            hccl_library=args.hccl_lib,
            signal_slots=SIGNAL_SLOTS,
            engine=args.hccl_channel_engine,
        )
        add_hex_masks(result)
        print("HCCL_CHANNEL_PROBE_JSON=" + json.dumps(result, sort_keys=True), flush=True)

        peer_mask = int(result["peer_mask"])
        ub_mem_peer_mask = int(result["ub_mem_peer_mask"])
        acquire_peer_mask = int(result["acquire_peer_mask"])
        if args.require_ub_mem and (ub_mem_peer_mask & peer_mask) != peer_mask:
            raise AssertionError(
                f"HCCL channel probe did not expose UB_MEM for all peers: "
                f"peer_mask=0x{peer_mask:x} ub_mem_peer_mask=0x{ub_mem_peer_mask:x}"
            )
        if args.require_acquire and (acquire_peer_mask & peer_mask) != peer_mask:
            raise AssertionError(
                f"HCCL channel probe did not acquire all peers: "
                f"peer_mask=0x{peer_mask:x} acquire_peer_mask=0x{acquire_peer_mask:x}"
            )
        wait_files(prefix, args.rank_size, args.rank, "probed")
        print(f"HCCL channel probe: DONE rank={args.rank}", flush=True)
    finally:
        gin_runtime.destroy_hccl_comm(comm, hccl_library=args.hccl_lib)


if __name__ == "__main__":
    main()
