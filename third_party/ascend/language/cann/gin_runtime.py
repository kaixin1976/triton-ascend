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
_HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_COMM_QOS_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET = -1
_HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE_GB = 16


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
        config = _make_hccl_comm_config(rank, rendezvous_id, sym_win_max_mem_gb)
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


def _make_hccl_comm_config(rank, comm_name, sym_win_max_mem_gb):
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
    config.hcclOpExpansionMode = _HCCL_COMM_DEFAULT_OP_EXPANSION_MODE
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
    library.TritonAscendGinRefreshFromTileXR.argtypes = [ctypes.c_void_p]
    library.TritonAscendGinRefreshFromTileXR.restype = ctypes.c_int
    library.TritonAscendGinGetDevComm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    library.TritonAscendGinGetDevComm.restype = ctypes.c_int
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


__all__ = [
    "GinComm",
    "TileXROptions",
    "HcclPeerMemOptions",
    "create_from_tilexr",
    "create_from_hccl_peer_mem",
    "hccl_comm_handle_from_name",
    "hccl_comm_handle_from_process_group",
    "create_from_torch_hccl_process_group",
    "create_hccl_root_info_comm",
    "destroy_hccl_comm",
    "install_hccl_memory_allocator",
]
