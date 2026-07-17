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


def add_hex_fields(result):
    for key in (
        "ccl_buffer_ptr_before",
        "ccl_buffer_ptr_after",
        "in_aiv_opbase_devmem_ptr",
        "in_aiv_opbase_ptr",
        "out_aiv_opbase_devmem_ptr",
        "out_aiv_opbase_ptr",
        "aiv_comm_info_devmem_ptr",
        "aiv_comm_info_ptr",
    ):
        result[f"{key}_hex"] = f"0x{int(result[key]):x}"
    return result


def require_success(result):
    zero_status_fields = (
        "acl_set_device_status",
        "get_independent_ccl_before_status",
        "create_comm_aiv_buffer_status",
        "create_comm_info_aiv_buffer_status",
        "get_independent_ccl_after_status",
        "clear_comm_aiv_buffer_status",
        "release_comm_aiv_buffer_status",
        "first_error_status",
    )
    for key in zero_status_fields:
        if int(result[key]) != 0:
            raise AssertionError(f"{key}={result[key]} expected 0")

    positive_fields = (
        "manager_storage_bytes",
        "ccl_buffer_ptr_before",
        "ccl_buffer_bytes_before",
        "ccl_buffer_ptr_after",
        "ccl_buffer_bytes_after",
        "in_aiv_opbase_devmem_ptr",
        "in_aiv_opbase_ptr",
        "in_aiv_opbase_bytes",
        "out_aiv_opbase_devmem_ptr",
        "out_aiv_opbase_ptr",
        "out_aiv_opbase_bytes",
        "aiv_comm_info_devmem_ptr",
        "aiv_comm_info_ptr",
        "aiv_comm_info_bytes",
    )
    for key in positive_fields:
        if int(result[key]) <= 0:
            raise AssertionError(f"{key}={result[key]} expected > 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hccl-lib", default=HCCL_LIB)
    parser.add_argument("--hccl-alg-lib", default=None)
    parser.add_argument("--runtime-lib", default=RUNTIME_LIB)
    parser.add_argument("--no-require", action="store_true")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    result = gin_runtime.probe_hccl_alg_buffers(
        runtime_library=args.runtime_lib,
        hccl_alg_library=args.hccl_alg_lib,
        hccl_library=args.hccl_lib,
        device=args.device,
    )
    add_hex_fields(result)
    print("HCCL_ALG_BUFFER_PROBE_JSON=" + json.dumps(result, sort_keys=True), flush=True)

    if not args.no_require:
        require_success(result)
    print(f"HCCL alg buffer probe: DONE device={args.device}", flush=True)


if __name__ == "__main__":
    main()
