import os
from pathlib import Path

from triton.language import core, semantic

try:
    from .extension.core import CORE as _ASCEND_CORE
    from .extension.core import MODE as _ASCEND_MODE
    from .extension.core import PIPE as _ASCEND_PIPE
    from .extension.custom_op import custom_semantic as _ascend_custom_semantic
    from .extension.custom_op import register_custom_op as _ascend_register_custom_op
except Exception:
    _ASCEND_CORE = None
    _ASCEND_MODE = None
    _ASCEND_PIPE = None
    _ascend_custom_semantic = None
    _ascend_register_custom_op = None


TEAM_WORLD = 0
TEAM_LOCAL = 1

GIN_BACKEND_HCCL_PEER_MEM = 1 << 0
GIN_BACKEND_TILEXR_UDMA = 1 << 1
GIN_BACKEND_HCOMM_TASK = 1 << 2
GIN_BACKEND_TILEXR_IPC_PEER_MEM = 1 << 3
GIN_BACKEND_UBS_CORE_UBC = 1 << 4
GIN_BACKEND_HCCL_CHANNEL = 1 << 5
GIN_BACKEND_UBS_COMM_URMA = GIN_BACKEND_UBS_CORE_UBC
GIN_BACKEND_PEER_MEM_COPY = (
    GIN_BACKEND_HCCL_PEER_MEM | GIN_BACKEND_TILEXR_IPC_PEER_MEM | GIN_BACKEND_HCCL_CHANNEL
)
GIN_BACKEND_ALL = (
    GIN_BACKEND_HCCL_PEER_MEM
    | GIN_BACKEND_TILEXR_UDMA
    | GIN_BACKEND_HCOMM_TASK
    | GIN_BACKEND_TILEXR_IPC_PEER_MEM
    | GIN_BACKEND_UBS_CORE_UBC
    | GIN_BACKEND_HCCL_CHANNEL
)

SEM_RELAXED = 0
SEM_ACQUIRE = 2
SEM_RELEASE = 3
SEM_ACQ_REL = 4
SEM_SEQ_CST = 5


_GIN_SYMBOLS = [
    "__triton_ascend_gin_rank",
    "__triton_ascend_gin_num_ranks",
    "__triton_ascend_gin_put",
    "__triton_ascend_gin_put_signal",
    "__triton_ascend_gin_put_window",
    "__triton_ascend_gin_put_signal_window",
    "__triton_ascend_gin_get",
    "__triton_ascend_gin_signal",
    "__triton_ascend_gin_wait_signal",
    "__triton_ascend_gin_read_signal",
    "__triton_ascend_gin_reset_signal",
    "__triton_ascend_gin_flush",
    "__triton_ascend_gin_barrier",
]

_GIN_CUSTOM_NAMES = {}
_GIN_EXTERN_SYMBOLS = {}


def _gin_lowering_mode():
    return os.environ.get("TRITON_ASCEND_GIN_LOWERING", "npuir").lower()


def _use_bitcode_lowering():
    return _gin_lowering_mode() in ("bitcode", "bc", "legacy")


def _use_custom_semantic():
    return os.environ.get("TRITON_ASCEND_GIN_USE_CUSTOM_SEMANTIC", "0") in ("1", "true", "TRUE", "on", "ON")


def _builtin_name(symbol):
    return symbol.replace("__triton_ascend_gin_", "__builtin_triton_ascend_gin_", 1)


def _linked_custom_name(symbol):
    return _linked_symbol(symbol)


def _linked_symbol(symbol):
    return symbol + "_op"


def _extern_elementwise_symbol(symbol):
    stem = symbol
    if stem.startswith("__triton_ascend_gin_"):
        stem = stem[len("__triton_ascend_gin_"):]
    # CCE emits AIV bitcode entry points with the ".vector" suffix.
    return "__hmf_triton_ascend_gin_" + stem + "_op.vector"


def _default_bitcode_path():
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates = [
            parent / "backends" / "ascend" / "lib" / "gin" / "triton_ascend_gin_tilexr.bc",
            parent / "backend" / "lib" / "gin" / "triton_ascend_gin_tilexr.bc",
            parent / "lib" / "gin" / "triton_ascend_gin_tilexr.bc",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return None


def _gin_bitcode_path():
    env_path = os.environ.get("TRITON_ASCEND_GIN_BITCODE")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return str(path.resolve())
    return _default_bitcode_path()


def _register_gin_custom_ops():
    if _ascend_register_custom_op is None:
        return

    bitcode = _gin_bitcode_path() if _use_bitcode_lowering() else None
    for symbol in _GIN_SYMBOLS:
        name = _linked_custom_name(symbol) if bitcode else _builtin_name(symbol)
        attrs = {
            "name": name,
            "symbol": _linked_symbol(symbol) if bitcode else symbol,
            "core": _ASCEND_CORE.VECTOR,
            "pipe": _ASCEND_PIPE.PIPE_ALL,
            "mode": _ASCEND_MODE.SIMT,
        }
        if bitcode:
            attrs["bitcode"] = bitcode
        op_class = type(
            name,
            (),
            attrs,
        )
        try:
            _ascend_register_custom_op(op_class)
        except AssertionError as exc:
            if "already used" not in str(exc):
                raise
        _GIN_CUSTOM_NAMES[symbol] = name
        if bitcode:
            _GIN_EXTERN_SYMBOLS[symbol] = _extern_elementwise_symbol(symbol)


_register_gin_custom_ops()


class DevComm:
    """Opaque device communicator descriptor passed from the host runtime."""

    def __init__(self, handle):
        self.handle = handle


class Window:
    """Opaque registered or symmetric communication window.

    ``ptr`` is optional and is used only by ordinary Triton operations through
    ``local_ptr``. Communication operations use ``handle`` and byte offsets.
    """

    def __init__(self, handle, ptr=None):
        self.handle = handle
        self.ptr = ptr


class Gin:
    """GIN-like context selected from a device communicator."""

    def __init__(self, dev_comm, backend_mask, context_id):
        self.dev_comm = dev_comm
        self.backend_mask = backend_mask
        self.context_id = context_id


def _to_tensor(x, _builder):
    return semantic.to_tensor(x, _builder)


def _as_i32(x, _builder):
    return _to_tensor(x, _builder).to(core.int32, _builder=_builder)


def _as_u32(x, _builder):
    return _to_tensor(x, _builder).to(core.uint32, _builder=_builder)


def _as_u64(x, _builder):
    return _to_tensor(x, _builder).to(core.uint64, _builder=_builder)


def _as_block(x, dtype, _builder):
    tensor = _to_tensor(x, _builder).to(dtype, _builder=_builder)
    if tensor.type.is_block():
        return tensor
    return semantic.splat(tensor, [1], _builder)


def _as_i32_block(x, _builder):
    return _as_block(x, core.int32, _builder)


def _as_u64_block(x, _builder):
    return _as_block(x, core.uint64, _builder)


def _dev_comm_handle(comm, _builder):
    if isinstance(comm, (tuple, list)) and len(comm) == 3:
        comm = comm[0]
    if isinstance(comm, Gin):
        comm = comm.dev_comm
    if isinstance(comm, DevComm):
        return _as_u64(comm.handle, _builder)
    return _as_u64(comm, _builder)


def _window_handle(window, _builder):
    if isinstance(window, (tuple, list)) and len(window) == 2:
        return _as_u64(window[0], _builder)
    if isinstance(window, Window):
        return _as_u64(window.handle, _builder)
    return _as_u64(window, _builder)


def _ptr_value(ptr, _builder):
    if isinstance(ptr, (tuple, list)) and len(ptr) == 2:
        ptr = ptr[1]
        if ptr is None:
            raise ValueError("GIN operation received a window source/destination without ptr=...")
    if isinstance(ptr, Window):
        if ptr.ptr is None:
            raise ValueError("GIN operation received a Window source/destination without ptr=...")
        ptr = ptr.ptr
    return _as_u64(ptr, _builder)


def _gin_fields(g, _builder):
    if isinstance(g, (tuple, list)) and len(g) == 3:
        return (
            _as_u64(g[0], _builder),
            _as_u32(g[1], _builder),
            _as_i32(g[2], _builder),
        )
    if isinstance(g, Gin):
        return (
            _dev_comm_handle(g.dev_comm, _builder),
            _as_u32(g.backend_mask, _builder),
            _as_i32(g.context_id, _builder),
        )
    return (
        _dev_comm_handle(g, _builder),
        _as_u32(GIN_BACKEND_ALL, _builder),
        _as_i32(0, _builder),
    )


def _extern(symbol, args, arg_types, ret_type=core.int32, is_pure=False, _builder=None):
    bitcode = _gin_bitcode_path() if _use_bitcode_lowering() else None
    if bitcode and not _use_custom_semantic():
        block_args = []
        extern_arg_types = []
        for arg in args:
            arg_tensor = _to_tensor(arg, _builder)
            if arg_tensor.dtype == core.uint64:
                block_args.append(_as_u64_block(arg_tensor, _builder))
                extern_arg_types.append(core.uint64)
            else:
                block_args.append(_as_i32_block(arg_tensor, _builder))
                extern_arg_types.append(core.int32)
        return core.extern_elementwise(
            "triton_ascend_gin",
            bitcode,
            block_args,
            {tuple(extern_arg_types): (_extern_elementwise_symbol(symbol), ret_type)},
            # The Triton-Ascend GIN converter handles these impure externs
            # before the generic Ascend extern-elementwise pure-op check.
            is_pure=False,
            _builder=_builder,
        )
    if _use_custom_semantic() and _ascend_custom_semantic is not None:
        del arg_types, is_pure
        custom_name = _GIN_CUSTOM_NAMES.get(symbol, _builtin_name(symbol))
        if ret_type == core.uint64:
            out = _as_u64_block(0, _builder)
        else:
            out = _as_i32_block(0, _builder)
        block_args = []
        for arg in args:
            arg_tensor = _to_tensor(arg, _builder)
            if arg_tensor.dtype == core.uint64:
                block_args.append(_as_u64_block(arg_tensor, _builder))
            else:
                block_args.append(_as_i32_block(arg_tensor, _builder))
        return _ascend_custom_semantic(custom_name, *block_args, out=out, _builder=_builder)
    return core.extern_elementwise(
        "triton_ascend_gin",
        "",
        args,
        {tuple(arg_types): (symbol, ret_type)},
        is_pure=is_pure,
        _builder=_builder,
    )


@core.builtin
def dev_comm(handle, _builder=None):
    """Create an opaque device communicator from a uint64-compatible handle."""

    return _as_u64(handle, _builder)


@core.builtin
def window(handle, ptr=None, _builder=None):
    """Create an opaque registered/symmetric communication window."""

    handle = _as_u64(handle, _builder)
    if ptr is None:
        return handle
    return (handle, ptr)


@core.builtin
def local_ptr(window, _builder=None):
    """Return the local pointer associated with a communication window."""

    if isinstance(window, (tuple, list)) and len(window) == 2 and window[1] is not None:
        return window[1]
    if not isinstance(window, Window) or window.ptr is None:
        raise ValueError("gin.local_ptr expects a gin.window created with ptr=...")
    return window.ptr


@core.builtin
def gin(dev_comm, backend_mask=GIN_BACKEND_ALL, context_id=0, _builder=None):
    """Create a GIN-like communication context from a device communicator."""

    if not isinstance(dev_comm, DevComm):
        return (_as_u64(dev_comm, _builder), _as_u32(backend_mask, _builder), _as_i32(context_id, _builder))
    return (_as_u64(dev_comm.handle, _builder), _as_u32(backend_mask, _builder), _as_i32(context_id, _builder))


@core.builtin
def token(_builder=None):
    """Return an empty dependency token for chaining communication operations."""

    return _as_i32_block(0, _builder)


@core.builtin
def byte_offset(ptr, base, _builder=None):
    """Compute a byte offset between two pointers as uint64."""

    return semantic.sub(_as_u64(ptr, _builder), _as_u64(base, _builder), False, _builder)


def _load_u32_field(comm, byte_offset, _builder):
    addr = semantic.add(_dev_comm_handle(comm, _builder), _as_u64(byte_offset, _builder), False, _builder)
    ptr = addr.to(core.pointer_type(core.uint32), _builder=_builder)
    return core.load(ptr, _builder=_builder).to(core.int32, _builder=_builder)


@core.builtin
def rank(comm, team=TEAM_WORLD, _builder=None):
    """Return this rank in the selected communication team."""

    args = [_dev_comm_handle(comm, _builder), _as_i32(team, _builder)]
    return _extern(
        "__triton_ascend_gin_rank",
        args,
        [core.uint64, core.int32],
        _builder=_builder,
    )


@core.builtin
def num_ranks(comm, team=TEAM_WORLD, _builder=None):
    """Return the number of ranks in the selected communication team."""

    args = [_dev_comm_handle(comm, _builder), _as_i32(team, _builder)]
    return _extern(
        "__triton_ascend_gin_num_ranks",
        args,
        [core.uint64, core.int32],
        _builder=_builder,
    )


@core.builtin
def put(g, peer, dst_window, dst_offset, src, nbytes, team=TEAM_WORLD, sem=SEM_RELEASE, token=0, _builder=None):
    """Issue a one-sided put from local GM ``src`` to a peer window."""

    comm_handle, backend_mask, context_id = _gin_fields(g, _builder)
    args = [
        comm_handle,
        backend_mask,
        context_id,
        _as_i32(team, _builder),
        _as_i32(peer, _builder),
        _window_handle(dst_window, _builder),
        _as_u64(dst_offset, _builder),
        _ptr_value(src, _builder),
        _as_u64(nbytes, _builder),
        _as_i32(sem, _builder),
        _as_i32(token, _builder),
    ]
    return _extern(
        "__triton_ascend_gin_put",
        args,
        [
            core.uint64,
            core.uint32,
            core.int32,
            core.int32,
            core.int32,
            core.uint64,
            core.uint64,
            core.uint64,
            core.uint64,
            core.int32,
            core.int32,
        ],
        _builder=_builder,
    )


@core.builtin
def put_signal(g, peer, dst_window, dst_offset, src, nbytes, signal_id, signal_value=1, team=TEAM_WORLD,
               sem=SEM_RELEASE, token=0, _builder=None):
    """Issue a put and then make a peer-visible signal available."""

    comm_handle, backend_mask, context_id = _gin_fields(g, _builder)
    args = [
        comm_handle,
        backend_mask,
        context_id,
        _as_i32(team, _builder),
        _as_i32(peer, _builder),
        _window_handle(dst_window, _builder),
        _as_u64(dst_offset, _builder),
        _ptr_value(src, _builder),
        _as_u64(nbytes, _builder),
        _as_i32(signal_id, _builder),
        _as_u64(signal_value, _builder),
        _as_i32(sem, _builder),
        _as_i32(token, _builder),
    ]
    return _extern(
        "__triton_ascend_gin_put_signal",
        args,
        [
            core.uint64,
            core.uint32,
            core.int32,
            core.int32,
            core.int32,
            core.uint64,
            core.uint64,
            core.uint64,
            core.uint64,
            core.int32,
            core.uint64,
            core.int32,
            core.int32,
        ],
        _builder=_builder,
    )


@core.builtin
def put_window(g, peer, dst_window, dst_offset, src_window, src_offset, nbytes, team=TEAM_WORLD,
               sem=SEM_RELEASE, token=0, _builder=None):
    """Issue a one-sided put from a local registered window to a peer window."""

    comm_handle, backend_mask, context_id = _gin_fields(g, _builder)
    args = [
        comm_handle,
        backend_mask,
        context_id,
        _as_i32(team, _builder),
        _as_i32(peer, _builder),
        _window_handle(dst_window, _builder),
        _as_u64(dst_offset, _builder),
        _window_handle(src_window, _builder),
        _as_u64(src_offset, _builder),
        _as_u64(nbytes, _builder),
        _as_i32(sem, _builder),
        _as_i32(token, _builder),
    ]
    return _extern(
        "__triton_ascend_gin_put_window",
        args,
        [
            core.uint64,
            core.uint32,
            core.int32,
            core.int32,
            core.int32,
            core.uint64,
            core.uint64,
            core.uint64,
            core.uint64,
            core.uint64,
            core.int32,
            core.int32,
        ],
        _builder=_builder,
    )


@core.builtin
def put_signal_window(g, peer, dst_window, dst_offset, src_window, src_offset, nbytes, signal_id,
                      signal_value=1, team=TEAM_WORLD, sem=SEM_RELEASE, token=0, _builder=None):
    """Issue a put from a local registered window and then signal the peer."""

    comm_handle, backend_mask, context_id = _gin_fields(g, _builder)
    args = [
        comm_handle,
        backend_mask,
        context_id,
        _as_i32(team, _builder),
        _as_i32(peer, _builder),
        _window_handle(dst_window, _builder),
        _as_u64(dst_offset, _builder),
        _window_handle(src_window, _builder),
        _as_u64(src_offset, _builder),
        _as_u64(nbytes, _builder),
        _as_i32(signal_id, _builder),
        _as_u64(signal_value, _builder),
        _as_i32(sem, _builder),
        _as_i32(token, _builder),
    ]
    return _extern(
        "__triton_ascend_gin_put_signal_window",
        args,
        [
            core.uint64,
            core.uint32,
            core.int32,
            core.int32,
            core.int32,
            core.uint64,
            core.uint64,
            core.uint64,
            core.uint64,
            core.uint64,
            core.int32,
            core.uint64,
            core.int32,
            core.int32,
        ],
        _builder=_builder,
    )


@core.builtin
def get(g, peer, dst, src_window, src_offset, nbytes, team=TEAM_WORLD, sem=SEM_ACQUIRE, token=0, _builder=None):
    """Issue a one-sided get from a peer window to local GM ``dst``."""

    comm_handle, backend_mask, context_id = _gin_fields(g, _builder)
    args = [
        comm_handle,
        backend_mask,
        context_id,
        _as_i32(team, _builder),
        _as_i32(peer, _builder),
        _ptr_value(dst, _builder),
        _window_handle(src_window, _builder),
        _as_u64(src_offset, _builder),
        _as_u64(nbytes, _builder),
        _as_i32(sem, _builder),
        _as_i32(token, _builder),
    ]
    return _extern(
        "__triton_ascend_gin_get",
        args,
        [
            core.uint64,
            core.uint32,
            core.int32,
            core.int32,
            core.int32,
            core.uint64,
            core.uint64,
            core.uint64,
            core.uint64,
            core.int32,
            core.int32,
        ],
        _builder=_builder,
    )


@core.builtin
def signal(g, peer, signal_id, signal_value=1, team=TEAM_WORLD, sem=SEM_RELEASE, token=0, _builder=None):
    """Signal a peer without an accompanying data transfer."""

    comm_handle, backend_mask, context_id = _gin_fields(g, _builder)
    args = [
        comm_handle,
        backend_mask,
        context_id,
        _as_i32(team, _builder),
        _as_i32(peer, _builder),
        _as_i32(signal_id, _builder),
        _as_u64(signal_value, _builder),
        _as_i32(sem, _builder),
        _as_i32(token, _builder),
    ]
    return _extern(
        "__triton_ascend_gin_signal",
        args,
        [
            core.uint64,
            core.uint32,
            core.int32,
            core.int32,
            core.int32,
            core.int32,
            core.uint64,
            core.int32,
            core.int32,
        ],
        _builder=_builder,
    )


@core.builtin
def wait_signal(g, peer, signal_id, least_value=1, bits=64, sem=SEM_ACQUIRE, token=0, _builder=None):
    """Wait until a local signal slot associated with ``peer`` reaches ``least_value``."""

    comm_handle, backend_mask, context_id = _gin_fields(g, _builder)
    args = [
        comm_handle,
        backend_mask,
        context_id,
        _as_i32(peer, _builder),
        _as_i32(signal_id, _builder),
        _as_u64(least_value, _builder),
        _as_i32(bits, _builder),
        _as_i32(sem, _builder),
        _as_i32(token, _builder),
    ]
    return _extern(
        "__triton_ascend_gin_wait_signal",
        args,
        [
            core.uint64,
            core.uint32,
            core.int32,
            core.int32,
            core.int32,
            core.uint64,
            core.int32,
            core.int32,
            core.int32,
        ],
        _builder=_builder,
    )


@core.builtin
def read_signal(g, peer, signal_id, bits=64, sem=SEM_ACQUIRE, _builder=None):
    """Read a local signal slot associated with ``peer``."""

    comm_handle, backend_mask, context_id = _gin_fields(g, _builder)
    args = [
        comm_handle,
        backend_mask,
        context_id,
        _as_i32(peer, _builder),
        _as_i32(signal_id, _builder),
        _as_i32(bits, _builder),
        _as_i32(sem, _builder),
    ]
    return _extern(
        "__triton_ascend_gin_read_signal",
        args,
        [core.uint64, core.uint32, core.int32, core.int32, core.int32, core.int32, core.int32],
        ret_type=core.uint64,
        _builder=_builder,
    )


@core.builtin
def reset_signal(g, peer, signal_id, token=0, _builder=None):
    """Reset a local signal slot associated with ``peer`` to zero."""

    comm_handle, backend_mask, context_id = _gin_fields(g, _builder)
    args = [
        comm_handle,
        backend_mask,
        context_id,
        _as_i32(peer, _builder),
        _as_i32(signal_id, _builder),
        _as_i32(token, _builder),
    ]
    return _extern(
        "__triton_ascend_gin_reset_signal",
        args,
        [core.uint64, core.uint32, core.int32, core.int32, core.int32, core.int32],
        _builder=_builder,
    )


@core.builtin
def flush(g, peer=-1, team=TEAM_WORLD, token=0, _builder=None):
    """Flush outstanding communication operations for ``peer`` or all peers."""

    comm_handle, backend_mask, context_id = _gin_fields(g, _builder)
    args = [
        comm_handle,
        backend_mask,
        context_id,
        _as_i32(team, _builder),
        _as_i32(peer, _builder),
        _as_i32(token, _builder),
    ]
    return _extern(
        "__triton_ascend_gin_flush",
        args,
        [core.uint64, core.uint32, core.int32, core.int32, core.int32, core.int32],
        _builder=_builder,
    )


@core.builtin
def barrier(g, team=TEAM_WORLD, token=0, _builder=None):
    """Run a team barrier through the communication helper."""

    comm_handle, backend_mask, context_id = _gin_fields(g, _builder)
    args = [comm_handle, backend_mask, context_id, _as_i32(team, _builder), _as_i32(token, _builder)]
    return _extern(
        "__triton_ascend_gin_barrier",
        args,
        [core.uint64, core.uint32, core.int32, core.int32, core.int32],
        _builder=_builder,
    )


__all__ = [
    "TEAM_WORLD",
    "TEAM_LOCAL",
    "GIN_BACKEND_HCCL_PEER_MEM",
    "GIN_BACKEND_TILEXR_UDMA",
    "GIN_BACKEND_HCOMM_TASK",
    "GIN_BACKEND_TILEXR_IPC_PEER_MEM",
    "GIN_BACKEND_UBS_CORE_UBC",
    "GIN_BACKEND_HCCL_CHANNEL",
    "GIN_BACKEND_UBS_COMM_URMA",
    "GIN_BACKEND_PEER_MEM_COPY",
    "GIN_BACKEND_ALL",
    "SEM_RELAXED",
    "SEM_ACQUIRE",
    "SEM_RELEASE",
    "SEM_ACQ_REL",
    "SEM_SEQ_CST",
    "DevComm",
    "Window",
    "Gin",
    "dev_comm",
    "window",
    "local_ptr",
    "gin",
    "token",
    "byte_offset",
    "rank",
    "num_ranks",
    "put",
    "put_signal",
    "put_window",
    "put_signal_window",
    "get",
    "signal",
    "wait_signal",
    "read_signal",
    "reset_signal",
    "flush",
    "barrier",
]
