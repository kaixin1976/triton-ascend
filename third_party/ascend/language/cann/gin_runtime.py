import ctypes
import ctypes.util
import os


_DEFAULT_IPC_DATA_OFFSET = 2 * 1024 * 1024
_DEFAULT_WINDOW_BYTES = 100 * 1024 * 1024
_DEFAULT_SIGNAL_STRIDE = 8
_DEFAULT_SIGNAL_SLOTS = 2048


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
]
