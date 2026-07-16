import ctypes
import ctypes.util
import os
import time
from pathlib import Path


_DEFAULT_IPC_DATA_OFFSET = 2 * 1024 * 1024
_DEFAULT_WINDOW_BYTES = 100 * 1024 * 1024
_DEFAULT_SIGNAL_STRIDE = 8
_DEFAULT_SIGNAL_SLOTS = 2048

_HCCL_LIBRARIES = []
_ACTIVE_HCCL_ALLOCATOR = None

_HCCL_ROOT_INFO_BYTES = 4108
_HCCL_COMM_CONFIG_INFO_BYTES = 24
_HCCL_COMM_NAME_MAX_LENGTH = 128
_HCCL_UDI_MAX_LENGTH = 128
_HCCL_COMM_ALGO_MAX_LENGTH = 1600
_HCCL_COMM_RETRY_ENABLE_MAX_LENGTH = 50
_HCCL_COMM_RETRY_PARAMS_MAX_LENGTH = 128
_HCCL_BUFFER_NAME_MAX_LENGTH = 128
_HCCL_COMM_CONFIG_MAGIC_WORD = 0xF0F0F0F0
_HCCL_COMM_CONFIG_VERSION = 10
_HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_COMM_DEFAULT_OP_EXPANSION_MODE = 0
_HCCL_DATA_TYPE_FP32 = 4
_HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_COMM_QOS_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET = -1
_HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE_GB = 16

_HCCL_OP_EXPANSION_MODES = {
    "default": _HCCL_COMM_DEFAULT_OP_EXPANSION_MODE,
    "host": 1,
    "ai_cpu": 2,
    "aicpu": 2,
    "aiv": 3,
}

_HCCL_CHANNEL_ENGINES = {
    "cpu": 0,
    "host": 0,
    "cpu_ts": 1,
    "host_ts": 1,
    "aicpu": 2,
    "ai_cpu": 2,
    "aicpu_ts": 3,
    "ai_cpu_ts": 3,
    "aiv": 4,
    "ccu": 5,
}

_HCCL_CHANNEL_PROTOCOLS = {
    0: "hccs",
    1: "roce",
    2: "pcie",
    3: "sio",
    4: "ubc_ctp",
    5: "ubc_tp",
    6: "ub_mem",
    7: "uboe",
}


class TileXROptions(ctypes.Structure):
    _fields_ = [
        ("ipc_data_offset", ctypes.c_uint64),
        ("window_bytes", ctypes.c_uint64),
        ("signal_stride", ctypes.c_uint64),
        ("signal_slots", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("tilexr_library_path", ctypes.c_char_p),
    ]


class HcclPeerMemOptions(ctypes.Structure):
    _fields_ = [
        ("window_bytes", ctypes.c_uint64),
        ("signal_stride", ctypes.c_uint64),
        ("signal_slots", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("hccl_library_path", ctypes.c_char_p),
    ]


class HcclChannelOptions(ctypes.Structure):
    _fields_ = [
        ("window_bytes", ctypes.c_uint64),
        ("signal_stride", ctypes.c_uint64),
        ("signal_slots", ctypes.c_uint32),
        ("engine", ctypes.c_uint32),
        ("hccl_library_path", ctypes.c_char_p),
    ]


class HcclChannelProbe(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("nranks", ctypes.c_uint32),
        ("engine", ctypes.c_uint32),
        ("layer_count", ctypes.c_uint32),
        ("first_error_peer", ctypes.c_uint32),
        ("peer_mask", ctypes.c_uint64),
        ("protocol_mask", ctypes.c_uint64),
        ("hccs_peer_mask", ctypes.c_uint64),
        ("roce_peer_mask", ctypes.c_uint64),
        ("pcie_peer_mask", ctypes.c_uint64),
        ("ubc_ctp_peer_mask", ctypes.c_uint64),
        ("ubc_tp_peer_mask", ctypes.c_uint64),
        ("ub_mem_peer_mask", ctypes.c_uint64),
        ("acquire_peer_mask", ctypes.c_uint64),
        ("ccl_buffer_ptr", ctypes.c_uint64),
        ("ccl_buffer_bytes", ctypes.c_uint64),
        ("aiv_comm_info_ptr", ctypes.c_uint64),
        ("aiv_comm_info_bytes", ctypes.c_uint64),
        ("aiv_comm_info_mem_handle", ctypes.c_uint64),
        ("ccl_buffer_status", ctypes.c_int32),
        ("aiv_comm_info_status", ctypes.c_int32),
        ("first_error_status", ctypes.c_int32),
        ("reserved0", ctypes.c_uint32),
        ("first_acquire_ret", ctypes.c_int32),
        ("first_acquire_peer", ctypes.c_uint32),
        ("first_acquire_engine", ctypes.c_uint32),
        ("first_acquire_protocol", ctypes.c_uint32),
        ("first_acquire_channel_count", ctypes.c_uint32),
        ("first_acquire_mem_handle_count", ctypes.c_uint32),
        ("first_acquire_channel", ctypes.c_uint64),
        ("exchange_info_enabled", ctypes.c_uint32),
        ("exchange_info_symbol_available", ctypes.c_uint32),
        ("exchange_info_status", ctypes.c_int32),
        ("exchange_info_op_execute_config", ctypes.c_uint32),
        ("exchange_info_ccl_buffer_bytes", ctypes.c_uint64),
        ("exchange_info_count", ctypes.c_uint64),
        ("exchange_info_data_type", ctypes.c_uint32),
        ("exchange_info_aiv_core_limit", ctypes.c_uint32),
        ("aiv_comm_info_layout_status", ctypes.c_int32),
        ("reserved1", ctypes.c_uint32),
        ("aiv_comm_info_gm_in_offset", ctypes.c_uint64),
        ("aiv_comm_info_gm_out_offset", ctypes.c_uint64),
        ("aiv_comm_info_local_gm_in", ctypes.c_uint64),
        ("aiv_comm_info_local_gm_out", ctypes.c_uint64),
        ("aiv_comm_info_local_flag_base", ctypes.c_uint64),
        ("aiv_descriptor_via_cpu_enabled", ctypes.c_uint32),
        ("aiv_descriptor_via_cpu_status", ctypes.c_int32),
        ("aiv_descriptor_via_cpu_first_ret", ctypes.c_int32),
        ("aiv_descriptor_via_cpu_first_peer", ctypes.c_uint32),
        ("aiv_descriptor_via_cpu_first_protocol", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("aiv_descriptor_via_cpu_first_channel", ctypes.c_uint64),
        ("aiv_descriptor_via_cpu_acquire_peer_mask", ctypes.c_uint64),
        ("aiv_descriptor_via_cpu_hccl_buffer_peer_mask", ctypes.c_uint64),
        ("aiv_descriptor_via_cpu_remote_mem_peer_mask", ctypes.c_uint64),
        ("aiv_descriptor_via_cpu_readback_gm_in_mask", ctypes.c_uint64),
        ("aiv_descriptor_via_cpu_readback_gm_out_mask", ctypes.c_uint64),
        ("aiv_descriptor_via_cpu_symmetric_gm_out_peer_mask", ctypes.c_uint64),
    ]


class HcclAlgBufferProbe(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("device", ctypes.c_uint32),
        ("manager_storage_bytes", ctypes.c_uint64),
        ("ccl_buffer_ptr_before", ctypes.c_uint64),
        ("ccl_buffer_bytes_before", ctypes.c_uint64),
        ("ccl_buffer_ptr_after", ctypes.c_uint64),
        ("ccl_buffer_bytes_after", ctypes.c_uint64),
        ("in_aiv_opbase_devmem_ptr", ctypes.c_uint64),
        ("in_aiv_opbase_ptr", ctypes.c_uint64),
        ("in_aiv_opbase_bytes", ctypes.c_uint64),
        ("out_aiv_opbase_devmem_ptr", ctypes.c_uint64),
        ("out_aiv_opbase_ptr", ctypes.c_uint64),
        ("out_aiv_opbase_bytes", ctypes.c_uint64),
        ("aiv_comm_info_devmem_ptr", ctypes.c_uint64),
        ("aiv_comm_info_ptr", ctypes.c_uint64),
        ("aiv_comm_info_bytes", ctypes.c_uint64),
        ("acl_set_device_status", ctypes.c_int32),
        ("get_independent_ccl_before_status", ctypes.c_int32),
        ("create_comm_aiv_buffer_status", ctypes.c_int32),
        ("create_comm_info_aiv_buffer_status", ctypes.c_int32),
        ("get_independent_ccl_after_status", ctypes.c_int32),
        ("clear_comm_aiv_buffer_status", ctypes.c_int32),
        ("release_comm_aiv_buffer_status", ctypes.c_int32),
        ("first_error_status", ctypes.c_int32),
    ]


class HcclRootInfo(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * _HCCL_ROOT_INFO_BYTES)]


class HcclCommConfigInfo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magicWord", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64),
    ]


class HcclCommConfig(ctypes.Structure):
    _fields_ = [
        ("reserved", ctypes.c_char * _HCCL_COMM_CONFIG_INFO_BYTES),
        ("hcclBufferSize", ctypes.c_uint32),
        ("hcclDeterministic", ctypes.c_uint32),
        ("hcclCommName", ctypes.c_char * _HCCL_COMM_NAME_MAX_LENGTH),
        ("hcclUdi", ctypes.c_char * _HCCL_UDI_MAX_LENGTH),
        ("hcclOpExpansionMode", ctypes.c_uint32),
        ("hcclRdmaTrafficClass", ctypes.c_uint32),
        ("hcclRdmaServiceLevel", ctypes.c_uint32),
        ("hcclWorldRankID", ctypes.c_uint32),
        ("hcclJobID", ctypes.c_uint64),
        ("aclGraphZeroCopyEnable", ctypes.c_uint8),
        ("hcclExecTimeOut", ctypes.c_int32),
        ("hcclAlgo", ctypes.c_char * _HCCL_COMM_ALGO_MAX_LENGTH),
        ("hcclRetryEnable", ctypes.c_char * _HCCL_COMM_RETRY_ENABLE_MAX_LENGTH),
        ("hcclRetryParams", ctypes.c_char * _HCCL_COMM_RETRY_PARAMS_MAX_LENGTH),
        ("hcclBufferName", ctypes.c_char * _HCCL_BUFFER_NAME_MAX_LENGTH),
        ("hcclQos", ctypes.c_uint32),
        ("hcclSymWinMaxMemSizePerRank", ctypes.c_uint64),
    ]


class GinComm:
    def __init__(self, library, handle, backend="tilexr"):
        self._library = library
        self._handle = ctypes.c_void_p(handle)
        self._backend = backend

    @property
    def handle(self):
        return int(self._handle.value or 0)

    @property
    def dev_comm(self):
        out = ctypes.c_uint64()
        _check(self._library, self._library.TritonAscendGinGetDevComm(self._handle, ctypes.byref(out)))
        return int(out.value)

    def hccl_buffer_info(self):
        ptr = ctypes.c_uint64()
        nbytes = ctypes.c_uint64()
        _check(
            self._library,
            self._library.TritonAscendGinGetHcclBuffer(
                self._handle,
                ctypes.byref(ptr),
                ctypes.byref(nbytes),
            ),
        )
        return int(ptr.value), int(nbytes.value)

    def hccl_aiv_opbase_buffer_info(self, which="out"):
        if isinstance(which, str):
            text = which.strip().lower()
            if text == "in":
                which_value = 0
            elif text == "out":
                which_value = 1
            else:
                raise ValueError("which must be 'in', 'out', 0, or 1")
        else:
            which_value = int(which)
        ptr = ctypes.c_uint64()
        nbytes = ctypes.c_uint64()
        _check(
            self._library,
            self._library.TritonAscendGinGetHcclAivOpbaseBuffer(
                self._handle,
                ctypes.c_uint32(which_value),
                ctypes.byref(ptr),
                ctypes.byref(nbytes),
            ),
        )
        return int(ptr.value), int(nbytes.value)

    def hccl_buffer_tensor(self, shape, dtype, device=None, *, nbytes=None):
        import torch
        import torch_npu

        if device is None:
            device = f"npu:{torch_npu.npu.current_device()}"
        device = torch.device(device)
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be torch.dtype, got {type(dtype)}")

        needed = _shape_nbytes(shape, dtype)
        ptr, available = self.hccl_buffer_info()
        if nbytes is not None:
            needed = int(nbytes)
        if needed > available:
            raise ValueError(
                f"HCCL buffer has {available} bytes, but tensor requires {needed} bytes"
            )
        stride = _contiguous_stride(shape)
        metadata = {
            "data_ptr": ptr,
            "device": device,
            "nbytes": needed,
            "dtype": dtype,
            "size": shape,
            "stride": stride,
            "storage_offset": 0,
        }
        storage = torch_npu._C._construct_storage_from_data_pointer(
            metadata["data_ptr"], metadata["device"], metadata["nbytes"]
        )
        return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)

    def hccl_aiv_opbase_tensor(self, shape, dtype, device=None, *, which="out", nbytes=None):
        import torch
        import torch_npu

        if device is None:
            device = f"npu:{torch_npu.npu.current_device()}"
        device = torch.device(device)
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be torch.dtype, got {type(dtype)}")

        needed = _shape_nbytes(shape, dtype)
        ptr, available = self.hccl_aiv_opbase_buffer_info(which=which)
        if nbytes is not None:
            needed = int(nbytes)
        if needed > available:
            raise ValueError(
                f"HCCL AIV opbase buffer has {available} bytes, but tensor requires {needed} bytes"
            )
        stride = _contiguous_stride(shape)
        metadata = {
            "data_ptr": ptr,
            "device": device,
            "nbytes": needed,
            "dtype": dtype,
            "size": shape,
            "stride": stride,
            "storage_offset": 0,
        }
        storage = torch_npu._C._construct_storage_from_data_pointer(
            metadata["data_ptr"], metadata["device"], metadata["nbytes"]
        )
        return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)

    def hccl_aiv_allgather(self, send, recv, *, count=None, stream=None, rendezvous_id=None):
        import torch

        if not hasattr(send, "data_ptr") or not hasattr(recv, "data_ptr"):
            raise TypeError("send and recv must be tensor-like objects with data_ptr()")
        if send.dtype is not torch.float32 or recv.dtype is not torch.float32:
            raise TypeError("hccl_aiv_allgather currently supports torch.float32 tensors")
        if count is None:
            count = int(send.numel())
        if rendezvous_id is None:
            rendezvous_id = os.getenv("TRITON_ASCEND_GIN_HCCL_AIV_DIRECT_ID", "default")
        if stream is None:
            device_index = send.device.index
            if device_index is None:
                device_index = torch.npu.current_device()
            stream = torch.npu.current_stream(device_index).npu_stream
        _check(
            self._library,
            self._library.TritonAscendGinHcclAivAllGather(
                self._handle,
                ctypes.c_void_p(int(send.data_ptr())),
                ctypes.c_void_p(int(recv.data_ptr())),
                ctypes.c_uint64(int(count)),
                ctypes.c_uint32(_HCCL_DATA_TYPE_FP32),
                ctypes.c_void_p(int(stream)),
                _encode_path(rendezvous_id),
            ),
        )

    def refresh(self):
        if self._backend == "tilexr":
            _check(self._library, self._library.TritonAscendGinRefreshFromTileXR(self._handle))

    def window_handle(self, tensor=None, *, nbytes=None, rendezvous_id=None):
        if tensor is None:
            return 0
        if not hasattr(tensor, "data_ptr"):
            raise TypeError("window_handle expects a tensor-like object with data_ptr()")
        if nbytes is None:
            nbytes = _tensor_nbytes(tensor)
        if rendezvous_id is None:
            rendezvous_id = os.getenv("TRITON_ASCEND_GIN_WINDOW_ID", "default")
        out = ctypes.c_uint64()
        _check(
            self._library,
            self._library.TritonAscendGinRegisterWindow(
                self._handle,
                ctypes.c_void_p(int(tensor.data_ptr())),
                ctypes.c_uint64(int(nbytes)),
                _encode_path(rendezvous_id),
                ctypes.byref(out),
            ),
        )
        return int(out.value)

    def signal_window_handle(self, tensor, *, nbytes=None, rendezvous_id=None):
        if not hasattr(tensor, "data_ptr"):
            raise TypeError("signal_window_handle expects a tensor-like object with data_ptr()")
        if nbytes is None:
            nbytes = _tensor_nbytes(tensor)
        if rendezvous_id is None:
            rendezvous_id = os.getenv("TRITON_ASCEND_GIN_SIGNAL_WINDOW_ID", "default_signal")
        out = ctypes.c_uint64()
        _check(
            self._library,
            self._library.TritonAscendGinRegisterSignalWindow(
                self._handle,
                ctypes.c_void_p(int(tensor.data_ptr())),
                ctypes.c_uint64(int(nbytes)),
                _encode_path(rendezvous_id),
                ctypes.byref(out),
            ),
        )
        return int(out.value)

    def close(self):
        if self._handle.value:
            self._library.TritonAscendGinDestroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        self.close()


def create_from_tilexr(
    tilexr_comm,
    *,
    runtime_library=None,
    tilexr_library=None,
    ipc_data_offset=_DEFAULT_IPC_DATA_OFFSET,
    window_bytes=_DEFAULT_WINDOW_BYTES,
    signal_stride=_DEFAULT_SIGNAL_STRIDE,
    signal_slots=_DEFAULT_SIGNAL_SLOTS,
):
    library = _load_runtime_library(runtime_library)
    options = TileXROptions(
        int(ipc_data_offset),
        int(window_bytes),
        int(signal_stride),
        int(signal_slots),
        0,
        _encode_path(tilexr_library),
    )
    handle = ctypes.c_void_p()
    ret = library.TritonAscendGinCreateFromTileXR(
        ctypes.c_void_p(_as_pointer_value(tilexr_comm)),
        ctypes.byref(options),
        ctypes.byref(handle),
    )
    _check(library, ret)
    return GinComm(library, handle.value, backend="tilexr")


def create_from_hccl_peer_mem(
    hccl_comm,
    *,
    runtime_library=None,
    hccl_library=None,
    window_bytes=_DEFAULT_WINDOW_BYTES,
    signal_stride=_DEFAULT_SIGNAL_STRIDE,
    signal_slots=_DEFAULT_SIGNAL_SLOTS,
):
    library = _load_runtime_library(runtime_library)
    options = HcclPeerMemOptions(
        int(window_bytes),
        int(signal_stride),
        int(signal_slots),
        0,
        _encode_path(hccl_library),
    )
    handle = ctypes.c_void_p()
    ret = library.TritonAscendGinCreateFromHcclPeerMem(
        ctypes.c_void_p(_as_pointer_value(hccl_comm)),
        ctypes.byref(options),
        ctypes.byref(handle),
    )
    _check(library, ret)
    return GinComm(library, handle.value, backend="hccl_peer_mem")


def create_from_hccl_channel(
    hccl_comm,
    *,
    runtime_library=None,
    hccl_library=None,
    window_bytes=_DEFAULT_WINDOW_BYTES,
    signal_stride=_DEFAULT_SIGNAL_STRIDE,
    signal_slots=_DEFAULT_SIGNAL_SLOTS,
    engine="aiv",
):
    os.environ.setdefault("HCCL_INDEPENDENT_OP", "1")
    library = _load_runtime_library(runtime_library)
    options = HcclChannelOptions(
        int(window_bytes),
        int(signal_stride),
        int(signal_slots),
        int(hccl_channel_engine_value(engine)),
        _encode_path(hccl_library),
    )
    handle = ctypes.c_void_p()
    ret = library.TritonAscendGinCreateFromHcclChannel(
        ctypes.c_void_p(_as_pointer_value(hccl_comm)),
        ctypes.byref(options),
        ctypes.byref(handle),
    )
    _check(library, ret)
    return GinComm(library, handle.value, backend="hccl_channel")


def probe_hccl_channel(
    hccl_comm,
    *,
    runtime_library=None,
    hccl_library=None,
    window_bytes=_DEFAULT_WINDOW_BYTES,
    signal_stride=_DEFAULT_SIGNAL_STRIDE,
    signal_slots=_DEFAULT_SIGNAL_SLOTS,
    engine="aiv",
):
    os.environ.setdefault("HCCL_INDEPENDENT_OP", "1")
    library = _load_runtime_library(runtime_library)
    options = HcclChannelOptions(
        int(window_bytes),
        int(signal_stride),
        int(signal_slots),
        int(hccl_channel_engine_value(engine)),
        _encode_path(hccl_library),
    )
    probe = HcclChannelProbe()
    ret = library.TritonAscendGinProbeHcclChannel(
        ctypes.c_void_p(_as_pointer_value(hccl_comm)),
        ctypes.byref(options),
        ctypes.byref(probe),
    )
    _check(library, ret)
    result = {name: getattr(probe, name) for name, _ctype in probe._fields_}
    result["protocols"] = [
        name for value, name in _HCCL_CHANNEL_PROTOCOLS.items()
        if result["protocol_mask"] & (1 << value)
    ]
    result["engine_name"] = _hccl_channel_engine_name(result["engine"])
    return result


def probe_hccl_alg_buffers(
    *,
    runtime_library=None,
    hccl_alg_library=None,
    hccl_library=None,
    device=None,
):
    if hccl_alg_library is None:
        hccl_alg_library = _derive_hccl_alg_library(hccl_library)
    if device is None:
        import torch_npu

        device = torch_npu.npu.current_device()

    library = _load_runtime_library(runtime_library)
    probe = HcclAlgBufferProbe()
    ret = library.TritonAscendGinProbeHcclAlgBuffers(
        _encode_path(hccl_alg_library),
        ctypes.c_uint32(int(device)),
        ctypes.byref(probe),
    )
    _check(library, ret)
    return {name: getattr(probe, name) for name, _ctype in probe._fields_}


def hccl_comm_handle_from_name(comm_name, *, hccl_library=None):
    """Resolve a torch/HCCL communication-domain name to an HcclComm handle."""

    if not isinstance(comm_name, (str, bytes, os.PathLike)):
        raise TypeError("comm_name must be str, bytes, or path-like")

    library = _load_hccl_library(hccl_library)
    handle = ctypes.c_void_p()
    status = library.HcclCommGetHandleWithName(_encode_path(comm_name), ctypes.byref(handle))
    if status != 0 or not handle.value:
        raise RuntimeError(
            f"HcclCommGetHandleWithName({comm_name!r}) failed with status {status}"
        )
    return int(handle.value)


def hccl_comm_handle_from_process_group(group=None, *, rank=None, device=None, hccl_library=None):
    """Return the HcclComm handle owned by a torch HCCL process group."""

    import torch
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _get_default_group

    if group is None:
        group = _get_default_group()
    if rank is None:
        if hasattr(group, "rank"):
            rank = int(group.rank())
        else:
            rank = int(dist.get_rank(group=group))
    if device is None:
        device = torch.device("npu")

    backend = group
    if hasattr(group, "_get_backend"):
        backend = group._get_backend(device)

    if hasattr(backend, "get_hccl_comm_name"):
        comm_name = backend.get_hccl_comm_name(int(rank))
    elif hasattr(group, "get_hccl_comm_name"):
        comm_name = group.get_hccl_comm_name(int(rank))
    else:
        raise RuntimeError("the process group does not expose get_hccl_comm_name")

    return hccl_comm_handle_from_name(comm_name, hccl_library=hccl_library)


def create_from_torch_hccl_process_group(
    group=None,
    *,
    rank=None,
    device=None,
    runtime_library=None,
    hccl_library=None,
    window_bytes=_DEFAULT_WINDOW_BYTES,
    signal_stride=_DEFAULT_SIGNAL_STRIDE,
    signal_slots=_DEFAULT_SIGNAL_SLOTS,
):
    hccl_comm = hccl_comm_handle_from_process_group(
        group,
        rank=rank,
        device=device,
        hccl_library=hccl_library,
    )
    return create_from_hccl_peer_mem(
        hccl_comm,
        runtime_library=runtime_library,
        hccl_library=hccl_library,
        window_bytes=window_bytes,
        signal_stride=signal_stride,
        signal_slots=signal_slots,
    )


def create_hccl_root_info_comm(
    *,
    rank,
    rank_size,
    rendezvous_id,
    hccl_library=None,
    use_config=True,
    sym_win_max_mem_gb=_HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE_GB,
    op_expansion_mode=None,
    timeout_s=180,
):
    """Create an HcclComm with the same root-info path used by HCCL tests."""

    library = _load_hccl_library(hccl_library)
    _bind_hccl_root_info_apis(library)

    rank = int(rank)
    rank_size = int(rank_size)
    root_info = HcclRootInfo()
    rendezvous_dir = Path("/tmp") / f"triton_gin_hccl_root_{rendezvous_id}"
    root_path = rendezvous_dir / "hccl_root_info.bin"
    ready_path = rendezvous_dir / "hccl_root_info.ready"
    rendezvous_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        for path in (root_path, ready_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        ret = library.HcclGetRootInfo(ctypes.byref(root_info))
        if ret != 0:
            raise RuntimeError(f"HcclGetRootInfo failed with status {ret}")
        root_path.write_bytes(ctypes.string_at(ctypes.byref(root_info), _HCCL_ROOT_INFO_BYTES))
        ready_path.write_text("ready")
    else:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if ready_path.exists() and root_path.exists():
                break
            time.sleep(0.05)
        else:
            raise TimeoutError(f"timeout waiting for HCCL root info at {root_path}")
        payload = root_path.read_bytes()
        if len(payload) != _HCCL_ROOT_INFO_BYTES:
            raise RuntimeError(f"unexpected HCCL root info size {len(payload)}")
        ctypes.memmove(ctypes.byref(root_info), payload, len(payload))

    comm = ctypes.c_void_p()
    if use_config:
        config = _make_hccl_comm_config(
            rank,
            rendezvous_id,
            sym_win_max_mem_gb,
            op_expansion_mode=op_expansion_mode,
        )
        ret = library.HcclCommInitRootInfoConfig(
            ctypes.c_uint32(rank_size),
            ctypes.byref(root_info),
            ctypes.c_uint32(rank),
            ctypes.byref(config),
            ctypes.byref(comm),
        )
        api = "HcclCommInitRootInfoConfig"
    else:
        ret = library.HcclCommInitRootInfo(
            ctypes.c_uint32(rank_size),
            ctypes.byref(root_info),
            ctypes.c_uint32(rank),
            ctypes.byref(comm),
        )
        api = "HcclCommInitRootInfo"
    if ret != 0 or not comm.value:
        raise RuntimeError(f"{api} failed with status {ret}")
    return int(comm.value)


def destroy_hccl_comm(hccl_comm, *, hccl_library=None):
    if not hccl_comm:
        return
    library = _load_hccl_library(hccl_library)
    _bind_hccl_root_info_apis(library)
    ret = library.HcclCommDestroy(ctypes.c_void_p(_as_pointer_value(hccl_comm)))
    if ret != 0:
        raise RuntimeError(f"HcclCommDestroy failed with status {ret}")


def hccl_op_expansion_mode_value(op_expansion_mode=None):
    if op_expansion_mode is None:
        op_expansion_mode = os.getenv("TRITON_ASCEND_GIN_HCCL_OP_EXPANSION_MODE")
    if op_expansion_mode is None or op_expansion_mode == "":
        return _HCCL_COMM_DEFAULT_OP_EXPANSION_MODE
    if isinstance(op_expansion_mode, int):
        return int(op_expansion_mode)
    text = str(op_expansion_mode).strip().lower().replace("-", "_")
    if text.startswith("raw:"):
        return int(text[4:], 0)
    try:
        return int(text, 0)
    except ValueError:
        pass
    if text not in _HCCL_OP_EXPANSION_MODES:
        supported = ", ".join(sorted(_HCCL_OP_EXPANSION_MODES))
        raise ValueError(f"unsupported HCCL op expansion mode {op_expansion_mode!r}; supported: {supported}")
    return _HCCL_OP_EXPANSION_MODES[text]


def hccl_channel_engine_value(engine=None):
    if engine is None:
        engine = os.getenv("TRITON_ASCEND_GIN_HCCL_CHANNEL_ENGINE", "aiv")
    if engine is None or engine == "":
        engine = "aiv"
    if isinstance(engine, int):
        return int(engine)
    text = str(engine).strip().lower().replace("-", "_")
    if text.startswith("raw:"):
        return int(text[4:], 0)
    try:
        return int(text, 0)
    except ValueError:
        pass
    if text not in _HCCL_CHANNEL_ENGINES:
        supported = ", ".join(sorted(_HCCL_CHANNEL_ENGINES))
        raise ValueError(f"unsupported HCCL channel engine {engine!r}; supported: {supported}")
    return _HCCL_CHANNEL_ENGINES[text]


def _hccl_channel_engine_name(value):
    value = int(value)
    for name, engine_value in _HCCL_CHANNEL_ENGINES.items():
        if engine_value == value and name not in ("host", "host_ts", "ai_cpu", "ai_cpu_ts"):
            return name
    return f"raw:{value}"


def hccl_comm_config_capability(*, hccl_library=None):
    library = _load_hccl_library(hccl_library)
    try:
        func = library.HcclGetCommConfigCapability
    except AttributeError:
        return None
    func.argtypes = []
    func.restype = ctypes.c_uint32
    return int(func())


def install_hccl_memory_allocator(*, runtime_library=None, hccl_library=None):
    """Use HCCL's VMM allocator for subsequent torch NPU tensor allocations."""

    global _ACTIVE_HCCL_ALLOCATOR

    if hccl_library is not None:
        os.environ["TRITON_ASCEND_HCCL_LIB"] = os.fsdecode(hccl_library)

    library = _load_runtime_library(runtime_library)
    runtime_path = getattr(library, "_name", None)
    if not runtime_path:
        raise RuntimeError("could not determine the GIN runtime library path")

    import torch_npu

    allocator = torch_npu.npu.NPUPluggableAllocator(
        os.fsdecode(runtime_path),
        "TritonAscendGinHcclAllocatorAlloc",
        "TritonAscendGinHcclAllocatorFree",
    )
    torch_npu.npu.change_current_allocator(allocator)
    _ACTIVE_HCCL_ALLOCATOR = allocator
    return allocator


def _encode_path(path):
    if path is None:
        return None
    return os.fsencode(path)


def _derive_hccl_alg_library(hccl_library=None):
    env_alg = os.getenv("TRITON_ASCEND_HCCL_ALG_LIB")
    if env_alg:
        return env_alg

    base = hccl_library or os.getenv("TRITON_ASCEND_HCCL_LIB")
    if base:
        base_path = Path(base)
        if base_path.name in ("libhcomm.so", "libhccl.so"):
            return str(base_path.with_name("libhccl_alg.so"))
    return None


def _as_pointer_value(value):
    if isinstance(value, ctypes.c_void_p):
        return int(value.value or 0)
    raw = getattr(value, "value", None)
    if isinstance(raw, int):
        return int(raw)
    return int(value)


def _load_runtime_library(path=None):
    candidates = []
    if path:
        candidates.append(path)
    env_path = os.getenv("TRITON_ASCEND_GIN_RUNTIME_LIB")
    if env_path:
        candidates.append(env_path)
    candidates.extend(_local_runtime_library_candidates())
    found = ctypes.util.find_library("triton_ascend_gin_runtime")
    if found:
        candidates.append(found)
    candidates.append("libtriton_ascend_gin_runtime.so")

    errors = []
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            _bind_runtime_library(library)
            return library
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("failed to load Triton Ascend GIN runtime library; " + "; ".join(errors))


def _local_runtime_library_candidates():
    name = "libtriton_ascend_gin_runtime.so"
    rel_dirs = (
        Path("backends") / "ascend" / "lib" / "gin",
        Path("backends") / "ascend" / "lib",
        Path("_C"),
        Path("backend") / "lib" / "gin",
        Path("lib") / "gin",
    )
    here = Path(__file__).resolve()
    candidates = []
    seen = set()
    for parent in here.parents:
        for rel_dir in rel_dirs:
            candidate = parent / rel_dir / name
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                candidates.append(key)
    return candidates


def _load_hccl_library(path=None):
    candidates = []
    if path:
        candidates.append(path)
    env_path = os.getenv("TRITON_ASCEND_HCCL_LIB")
    if env_path:
        candidates.append(env_path)
    for name in ("hcomm", "hccl"):
        found = ctypes.util.find_library(name)
        if found:
            candidates.append(found)
    candidates.extend(["libhcomm.so", "libhccl.so"])

    errors = []
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            library.HcclCommGetHandleWithName.argtypes = [
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            library.HcclCommGetHandleWithName.restype = ctypes.c_int
            if hasattr(library, "HcclGetRootInfo"):
                _bind_hccl_root_info_apis(library)
            _HCCL_LIBRARIES.append(library)
            return library
        except (OSError, AttributeError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("failed to load HCCL library with HcclCommGetHandleWithName; " + "; ".join(errors))


def _bind_hccl_root_info_apis(library):
    library.HcclGetRootInfo.argtypes = [ctypes.POINTER(HcclRootInfo)]
    library.HcclGetRootInfo.restype = ctypes.c_int
    library.HcclCommInitRootInfo.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(HcclRootInfo),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.HcclCommInitRootInfo.restype = ctypes.c_int
    library.HcclCommInitRootInfoConfig.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(HcclRootInfo),
        ctypes.c_uint32,
        ctypes.POINTER(HcclCommConfig),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.HcclCommInitRootInfoConfig.restype = ctypes.c_int
    library.HcclCommDestroy.argtypes = [ctypes.c_void_p]
    library.HcclCommDestroy.restype = ctypes.c_int


def _make_hccl_comm_config(rank, comm_name, sym_win_max_mem_gb, *, op_expansion_mode=None):
    config = HcclCommConfig()
    info = HcclCommConfigInfo.from_buffer(config)
    info.size = ctypes.sizeof(HcclCommConfig)
    info.magicWord = _HCCL_COMM_CONFIG_MAGIC_WORD
    info.version = _HCCL_COMM_CONFIG_VERSION
    info.reserved = 0
    config.hcclBufferSize = _HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET
    config.hcclDeterministic = _HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET
    config.hcclCommName = os.fsencode(comm_name)[: _HCCL_COMM_NAME_MAX_LENGTH - 1]
    config.hcclUdi = b""
    config.hcclOpExpansionMode = hccl_op_expansion_mode_value(op_expansion_mode)
    config.hcclRdmaTrafficClass = _HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET
    config.hcclRdmaServiceLevel = _HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET
    config.hcclWorldRankID = int(rank)
    config.hcclJobID = 0
    config.aclGraphZeroCopyEnable = 0
    config.hcclExecTimeOut = _HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET
    config.hcclAlgo = b""
    config.hcclRetryEnable = b""
    config.hcclRetryParams = b""
    config.hcclBufferName = b""
    config.hcclQos = _HCCL_COMM_QOS_CONFIG_NOT_SET
    config.hcclSymWinMaxMemSizePerRank = int(sym_win_max_mem_gb)
    return config


def _bind_runtime_library(library):
    library.TritonAscendGinCreateFromTileXR.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(TileXROptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.TritonAscendGinCreateFromTileXR.restype = ctypes.c_int
    library.TritonAscendGinCreateFromHcclPeerMem.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(HcclPeerMemOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.TritonAscendGinCreateFromHcclPeerMem.restype = ctypes.c_int
    library.TritonAscendGinCreateFromHcclChannel.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(HcclChannelOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.TritonAscendGinCreateFromHcclChannel.restype = ctypes.c_int
    library.TritonAscendGinProbeHcclChannel.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(HcclChannelOptions),
        ctypes.POINTER(HcclChannelProbe),
    ]
    library.TritonAscendGinProbeHcclChannel.restype = ctypes.c_int
    library.TritonAscendGinProbeHcclAlgBuffers.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.POINTER(HcclAlgBufferProbe),
    ]
    library.TritonAscendGinProbeHcclAlgBuffers.restype = ctypes.c_int
    library.TritonAscendGinRefreshFromTileXR.argtypes = [ctypes.c_void_p]
    library.TritonAscendGinRefreshFromTileXR.restype = ctypes.c_int
    library.TritonAscendGinGetDevComm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    library.TritonAscendGinGetDevComm.restype = ctypes.c_int
    library.TritonAscendGinGetHcclBuffer.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.TritonAscendGinGetHcclBuffer.restype = ctypes.c_int
    library.TritonAscendGinGetHcclAivOpbaseBuffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.TritonAscendGinGetHcclAivOpbaseBuffer.restype = ctypes.c_int
    library.TritonAscendGinHcclAivAllGather.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    library.TritonAscendGinHcclAivAllGather.restype = ctypes.c_int
    library.TritonAscendGinRegisterWindow.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.TritonAscendGinRegisterWindow.restype = ctypes.c_int
    library.TritonAscendGinRegisterSignalWindow.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.TritonAscendGinRegisterSignalWindow.restype = ctypes.c_int
    library.TritonAscendGinGetLastError.argtypes = []
    library.TritonAscendGinGetLastError.restype = ctypes.c_char_p
    library.TritonAscendGinDestroy.argtypes = [ctypes.c_void_p]
    library.TritonAscendGinDestroy.restype = None


def _check(library, status):
    if status == 0:
        return
    message = library.TritonAscendGinGetLastError()
    if message:
        message = message.decode("utf-8", errors="replace")
    else:
        message = "unknown error"
    raise RuntimeError(f"Triton Ascend GIN runtime failed with status {status}: {message}")


def _tensor_nbytes(tensor):
    if hasattr(tensor, "nbytes"):
        return int(tensor.nbytes)
    if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
        return int(tensor.numel()) * int(tensor.element_size())
    raise TypeError("window_handle requires nbytes=... for tensors without nbytes/numel/element_size")


def _shape_nbytes(shape, dtype):
    import torch

    if isinstance(shape, int):
        shape = (shape,)
    numel = 1
    for dim in shape:
        if dim < 0:
            raise ValueError(f"shape dimensions must be non-negative, got {shape}")
        numel *= int(dim)
    return numel * torch.empty((), dtype=dtype).element_size()


def _contiguous_stride(shape):
    if len(shape) == 0:
        return ()
    stride = [1]
    for dim in reversed(shape[1:]):
        stride.insert(0, stride[0] * int(dim))
    return tuple(stride)


__all__ = [
    "GinComm",
    "TileXROptions",
    "HcclPeerMemOptions",
    "HcclChannelOptions",
    "HcclChannelProbe",
    "HcclAlgBufferProbe",
    "create_from_tilexr",
    "create_from_hccl_peer_mem",
    "create_from_hccl_channel",
    "probe_hccl_channel",
    "probe_hccl_alg_buffers",
    "hccl_comm_handle_from_name",
    "hccl_comm_handle_from_process_group",
    "create_from_torch_hccl_process_group",
    "create_hccl_root_info_comm",
    "destroy_hccl_comm",
    "hccl_comm_config_capability",
    "hccl_channel_engine_value",
    "hccl_op_expansion_mode_value",
    "install_hccl_memory_allocator",
]
