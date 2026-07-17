import argparse
import ctypes
import os
import pathlib
import time

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.gin as tgin
import triton.language.extra.cann.gin_runtime as gin_runtime


TILEXR_LIB = os.getenv("TRITON_ASCEND_TILEXR_LIB", "/home/kaixin/TileXR/install/lib64/libtile-comm.so")
HCCL_LIB = os.getenv(
    "TRITON_ASCEND_HCCL_LIB",
    "/home/kaixin/Ascend/cann-9.1.0-beta.1/aarch64-linux/lib64/libhcomm.so",
)
RUNTIME_LIB = os.getenv("TRITON_ASCEND_GIN_RUNTIME_LIB")
SIGNAL_SLOTS = 128
SLOT_DIRECT_PUT = 0
SLOT_PUT_SIGNAL = 1
SLOT_PUT_WINDOW = 2
SLOT_GET_SOURCE = 3
SLOT_PUT_SIGNAL_WINDOW = 4
SLOT_DIRECT_PUT_OFFSET = 5
SLOT_PUT_WINDOW_INLINE = 6
SLOTS = 7


@triton.jit
def reset_gin_signals_kernel(
    comm_h,
    MAX_RANKS_C: tl.constexpr,
    BLOCKS: tl.constexpr,
    BACKEND_MASK: tl.constexpr,
):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=BACKEND_MASK)
    nranks = tgin.num_ranks(dev)
    token = tgin.token()
    for peer in tl.static_range(0, MAX_RANKS_C):
        if peer < nranks:
            token = tgin.reset_signal(gin, peer, signal_id=8, token=token)
            token = tgin.reset_signal(gin, peer, signal_id=48, token=token)
            for pid in tl.static_range(0, BLOCKS):
                token = tgin.reset_signal(gin, peer, signal_id=24 + pid, token=token)
    token = tgin.flush(gin, token=token)


@triton.jit
def direct_put_kernel(x, y, comm_h, win_h, n_elements: tl.constexpr,
                      BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                      RANK_SIZE_C: tl.constexpr,
                      BACKEND_MASK: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=BACKEND_MASK)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    peer = tl.where(rank + 1 == RANK_SIZE_C, 0, rank + 1)
    token = tgin.token()

    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        vals = tl.load(x + offs)
        base = rank * n_elements
        tl.store(y + base + offs, vals)

    base = rank * n_elements
    token = tgin.put(gin, peer, win, base * 4, x, n_elements * 4, token=token)
    token = tgin.flush(gin, peer=peer, token=token)


@triton.jit
def direct_put_offset_kernel(x, y, comm_h, win_h, n_elements: tl.constexpr,
                             BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                             SLOT: tl.constexpr, RANK_SIZE_C: tl.constexpr,
                             BACKEND_MASK: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=BACKEND_MASK)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    peer = tl.where(rank + 1 == RANK_SIZE_C, 0, rank + 1)
    token = tgin.token()
    tile_bytes = BLOCK * 4

    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        dst_base = (SLOT * RANK_SIZE_C + rank) * n_elements
        vals = tl.load(x + n_elements + offs)
        tl.store(y + dst_base + offs, vals)
        dst_byte_off = (dst_base + pid * BLOCK) * 4
        src_ptr = x + n_elements + pid * BLOCK
        token = tgin.put(gin, peer, win, dst_byte_off, src_ptr,
                          tile_bytes, token=token)
    token = tgin.flush(gin, peer=peer, token=token)


@triton.jit
def put_signal_kernel(x, y, comm_h, win_h, n_elements: tl.constexpr,
                      BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                      SLOT: tl.constexpr, RANK_SIZE_C: tl.constexpr,
                      SIGNAL_BASE: tl.constexpr, BACKEND_MASK: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=BACKEND_MASK)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    peer = tl.where(rank + 1 == RANK_SIZE_C, 0, rank + 1)
    wait_peer = tl.where(rank == 0, RANK_SIZE_C - 1, rank - 1)
    token = tgin.token()

    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        vals = tl.load(x + offs)
        base = (SLOT * RANK_SIZE_C + rank) * n_elements
        tl.store(y + base + offs, vals)

    base = (SLOT * RANK_SIZE_C + rank) * n_elements
    transfer_bytes = n_elements * 4
    token = tgin.put_signal(
        gin, peer, win, base * 4, x, transfer_bytes,
        signal_id=SIGNAL_BASE, signal_value=transfer_bytes, token=token)
    token = tgin.flush(gin, peer=peer, token=token)
    token = tgin.wait_signal(gin, wait_peer, signal_id=SIGNAL_BASE,
                              least_value=transfer_bytes, token=token)


@triton.jit
def prepare_put_window_source_kernel(y, comm_h, n_elements: tl.constexpr,
                                     BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                                     SLOT: tl.constexpr, RANK_SIZE_C: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    rank = tgin.rank(dev)
    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        base = (SLOT * RANK_SIZE_C + rank) * n_elements
        vals = offs.to(tl.float32) + rank.to(tl.float32) * 2000.0
        tl.store(y + base + offs, vals)


@triton.jit
def put_window_kernel(y, comm_h, win_h, n_elements: tl.constexpr,
                      BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                      SLOT: tl.constexpr, RANK_SIZE_C: tl.constexpr,
                      BACKEND_MASK: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=BACKEND_MASK)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    peer = tl.where(rank + 1 == RANK_SIZE_C, 0, rank + 1)
    token = tgin.token()
    tile_bytes = BLOCK * 4

    for pid in tl.static_range(0, BLOCKS):
        base = (SLOT * RANK_SIZE_C + rank) * n_elements
        byte_off = (base + pid * BLOCK) * 4
        token = tgin.put_window(gin, peer, win, byte_off, win, byte_off,
                                 tile_bytes, token=token)
    token = tgin.flush(gin, peer=peer, token=token)


@triton.jit
def put_window_inline_kernel(y, comm_h, win_h, n_elements: tl.constexpr,
                             BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                             SLOT: tl.constexpr, RANK_SIZE_C: tl.constexpr,
                             BACKEND_MASK: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=BACKEND_MASK)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    peer = tl.where(rank + 1 == RANK_SIZE_C, 0, rank + 1)
    token = tgin.token()
    tile_bytes = BLOCK * 4

    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        base = (SLOT * RANK_SIZE_C + rank) * n_elements
        vals = offs.to(tl.float32) + rank.to(tl.float32) * 6000.0
        tl.store(y + base + offs, vals)

    tl.debug_barrier()

    for pid in tl.static_range(0, BLOCKS):
        base = (SLOT * RANK_SIZE_C + rank) * n_elements
        byte_off = (base + pid * BLOCK) * 4
        token = tgin.put_window(gin, peer, win, byte_off, win, byte_off,
                                 tile_bytes, token=token)
    token = tgin.flush(gin, peer=peer, token=token)


@triton.jit
def put_signal_window_kernel(y, comm_h, win_h, n_elements: tl.constexpr,
                             BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                             SLOT: tl.constexpr, RANK_SIZE_C: tl.constexpr,
                             SIGNAL_BASE: tl.constexpr,
                             BACKEND_MASK: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=BACKEND_MASK)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    peer = tl.where(rank + 1 == RANK_SIZE_C, 0, rank + 1)
    wait_peer = tl.where(rank == 0, RANK_SIZE_C - 1, rank - 1)
    token = tgin.token()
    tile_bytes = BLOCK * 4

    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        base = (SLOT * RANK_SIZE_C + rank) * n_elements
        vals = offs.to(tl.float32) + rank.to(tl.float32) * 4000.0
        tl.store(y + base + offs, vals)
        byte_off = (base + pid * BLOCK) * 4
        token = tgin.put_signal_window(
            gin, peer, win, byte_off, win, byte_off, tile_bytes,
            signal_id=SIGNAL_BASE + pid, signal_value=tile_bytes, token=token)

    token = tgin.flush(gin, peer=peer, token=token)
    for pid in tl.static_range(0, BLOCKS):
        token = tgin.wait_signal(gin, wait_peer, signal_id=SIGNAL_BASE + pid,
                                  least_value=tile_bytes, token=token)


@triton.jit
def prepare_get_source_kernel(y, comm_h, win_h, n_elements: tl.constexpr,
                              BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                              SLOT: tl.constexpr, RANK_SIZE_C: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    rank = tgin.rank(dev)
    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        base = (SLOT * RANK_SIZE_C + rank) * n_elements
        vals = offs.to(tl.float32) + rank.to(tl.float32) * 3000.0
        tl.store(y + base + offs, vals)


@triton.jit
def get_kernel(z, comm_h, win_h, n_elements: tl.constexpr,
               BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
               SLOT: tl.constexpr, RANK_SIZE_C: tl.constexpr,
               BACKEND_MASK: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=BACKEND_MASK)
    win = tgin.window(win_h)
    rank = tgin.rank(dev)
    peer = tl.where(rank + 1 == RANK_SIZE_C, 0, rank + 1)
    token = tgin.token()

    src_base = (SLOT * RANK_SIZE_C + peer) * n_elements
    token = tgin.get(gin, peer, z, win, src_base * 4, n_elements * 4,
                      token=token)
    token = tgin.flush(gin, peer=peer, token=token)


@triton.jit
def get_offset_kernel(z, comm_h, win_h, n_elements: tl.constexpr,
                      BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                      SLOT: tl.constexpr, RANK_SIZE_C: tl.constexpr,
                      BACKEND_MASK: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=BACKEND_MASK)
    win = tgin.window(win_h)
    rank = tgin.rank(dev)
    peer = tl.where(rank + 1 == RANK_SIZE_C, 0, rank + 1)
    token = tgin.token()
    tile_bytes = BLOCK * 4
    src_base = (SLOT * RANK_SIZE_C + peer) * n_elements

    for pid in tl.static_range(0, BLOCKS):
        token = tgin.get(gin, peer, z + n_elements + pid * BLOCK, win,
                          (src_base + pid * BLOCK) * 4, tile_bytes,
                          token=token)
    token = tgin.flush(gin, peer=peer, token=token)


@triton.jit
def signal_read_barrier_kernel(meta, comm_h, SIGNAL_ID: tl.constexpr,
                               RANK_SIZE_C: tl.constexpr,
                               BACKEND_MASK: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=BACKEND_MASK)
    rank = tgin.rank(dev)
    nranks = tgin.num_ranks(dev)
    peer = tl.where(rank + 1 == RANK_SIZE_C, 0, rank + 1)
    wait_peer = tl.where(rank == 0, RANK_SIZE_C - 1, rank - 1)
    token = tgin.token()

    token = tgin.signal(gin, peer, signal_id=SIGNAL_ID,
                         signal_value=rank + 101, token=token)
    token = tgin.flush(gin, peer=peer, token=token)
    token = tgin.barrier(gin, token=token)
    token = tgin.wait_signal(gin, wait_peer, signal_id=SIGNAL_ID,
                              least_value=wait_peer + 101, token=token)
    value = tgin.read_signal(gin, wait_peer, signal_id=SIGNAL_ID)
    idx = tl.arange(0, 1)
    tl.store(meta + idx, value.to(tl.int64))
    tl.store(meta + 1 + idx, rank.to(tl.int64))
    tl.store(meta + 2 + idx, nranks.to(tl.int64))
    tl.store(meta + 3 + idx, token.to(tl.int64))
    token = tgin.reset_signal(gin, wait_peer, signal_id=SIGNAL_ID, token=token)
    token = tgin.flush(gin, peer=wait_peer, token=token)


def bind_tilexr(tilexr_lib):
    lib = ctypes.CDLL(tilexr_lib, mode=ctypes.RTLD_GLOBAL)
    lib.TileXRCommInitRankLocal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    lib.TileXRCommInitRankLocal.restype = ctypes.c_int
    lib.TileXRCommDestroy.argtypes = [ctypes.c_void_p]
    lib.TileXRCommDestroy.restype = ctypes.c_int
    return lib


def backend_mask(backend):
    if backend == "tilexr":
        return tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM
    if backend == "hccl_peer_mem":
        return tgin.GIN_BACKEND_HCCL_PEER_MEM
    if backend == "hccl_channel":
        return tgin.GIN_BACKEND_HCCL_CHANNEL
    raise ValueError(f"unsupported backend {backend!r}")


def create_tilexr_gin(args):
    tile = bind_tilexr(args.tilexr_lib)
    tile_comm = ctypes.c_void_p()
    ret = tile.TileXRCommInitRankLocal(args.rank_size, args.rank, ctypes.byref(tile_comm))
    if ret != 0 or not tile_comm.value:
        raise RuntimeError(f"TileXRCommInitRankLocal failed rank={args.rank} ret={ret}")
    gin = gin_runtime.create_from_tilexr(
        tile_comm,
        runtime_library=RUNTIME_LIB,
        tilexr_library=args.tilexr_lib,
    )
    return gin, tile, tile_comm, None


def create_hccl_gin(args):
    mode_value = gin_runtime.hccl_op_expansion_mode_value(args.hccl_op_expansion_mode)
    channel_engine_value = gin_runtime.hccl_channel_engine_value(args.hccl_channel_engine)
    capability = gin_runtime.hccl_comm_config_capability(hccl_library=args.hccl_lib)
    capability_text = "none" if capability is None else f"0x{capability:x}"
    print(
        f"HCCL GIN root-info config: rank={args.rank} "
        f"op_expansion_mode={args.hccl_op_expansion_mode or 'default'} "
        f"value={mode_value} channel_engine={args.hccl_channel_engine} "
        f"channel_engine_value={channel_engine_value} capability={capability_text}",
        flush=True,
    )
    if args.backend == "hccl_channel":
        os.environ.setdefault("HCCL_INDEPENDENT_OP", "1")
    hccl_comm = gin_runtime.create_hccl_root_info_comm(
        rank=args.rank,
        rank_size=args.rank_size,
        rendezvous_id=args.tag,
        hccl_library=args.hccl_lib,
        use_config=True,
        op_expansion_mode=args.hccl_op_expansion_mode,
    )
    if args.backend == "hccl_channel":
        gin = gin_runtime.create_from_hccl_channel(
            hccl_comm,
            runtime_library=RUNTIME_LIB,
            hccl_library=args.hccl_lib,
            signal_slots=SIGNAL_SLOTS,
            engine=args.hccl_channel_engine,
        )
    else:
        gin = gin_runtime.create_from_hccl_peer_mem(
            hccl_comm,
            runtime_library=RUNTIME_LIB,
            hccl_library=args.hccl_lib,
            signal_slots=SIGNAL_SLOTS,
        )
    return gin, None, ctypes.c_void_p(hccl_comm), None


def hccl_signal_bytes(args):
    return args.rank_size * SIGNAL_SLOTS * 8


def hccl_window_numel(args, data_numel):
    if args.backend not in ("hccl_peer_mem", "hccl_channel"):
        return data_numel
    data_bytes = data_numel * 4
    signal_offset = ((data_bytes + 7) // 8) * 8
    total_bytes = signal_offset + hccl_signal_bytes(args)
    return (total_bytes + 3) // 4


def allocate_window_tensor(args, gin, data_numel, fill_value):
    if args.backend == "hccl_channel":
        tensor = gin.hccl_buffer_tensor(
            (hccl_window_numel(args, data_numel),),
            torch.float32,
            device=f"npu:{args.device}",
        )
        tensor.fill_(fill_value)
        return tensor
    return torch.full(
        (hccl_window_numel(args, data_numel),),
        fill_value,
        dtype=torch.float32,
        device="npu",
    )


def register_window(gin, args, tensor, data_numel, rendezvous_id):
    if args.backend in ("hccl_peer_mem", "hccl_channel"):
        return gin.window_handle(tensor, nbytes=data_numel * 4, rendezvous_id=rendezvous_id)
    return gin.window_handle(tensor, rendezvous_id=rendezvous_id)


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


def expected_slot(n_elements, rank_size, scale):
    chunks = []
    for source_rank in range(rank_size):
        chunks.append(torch.arange(n_elements, dtype=torch.float32) + source_rank * scale)
    return torch.cat(chunks)


def expected_slot_offset(n_elements, rank_size, scale, start):
    chunks = []
    for source_rank in range(rank_size):
        chunks.append(torch.arange(start, start + n_elements, dtype=torch.float32) + source_rank * scale)
    return torch.cat(chunks)


def expected_ring_slot(n_elements, rank_size, rank, scale, fill_value=-777.0, start=0):
    expected = torch.full((rank_size * n_elements,), fill_value, dtype=torch.float32)
    prev = (rank - 1 + rank_size) % rank_size
    for source_rank in sorted({rank, prev}):
        begin = source_rank * n_elements
        expected[begin:begin + n_elements] = (
            torch.arange(start, start + n_elements, dtype=torch.float32)
            + source_rank * scale
        )
    return expected


def check_slot(name, y_cpu, slot, n_elements, rank_size, expected):
    start = slot * rank_size * n_elements
    got = y_cpu[start:start + rank_size * n_elements]
    diff = (got - expected).abs()
    maxerr = float(diff.max())
    if maxerr != 0.0:
        bad = int(diff.argmax())
        ctx_start = max(0, bad - 4)
        ctx_end = min(int(got.numel()), bad + 5)
        raise AssertionError(
            f"{name} failed maxerr={maxerr} bad={bad} got={float(got[bad])} "
            f"expected={float(expected[bad])} "
            f"got_ctx={got[ctx_start:ctx_end].tolist()} "
            f"expected_ctx={expected[ctx_start:ctx_end].tolist()}"
        )
    print(f"{name}: PASS maxerr=0.0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rank-size", type=int, default=2)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--backend", choices=("tilexr", "hccl_peer_mem", "hccl_channel"), default="tilexr")
    parser.add_argument("--tilexr-lib", default=TILEXR_LIB)
    parser.add_argument("--hccl-lib", default=HCCL_LIB)
    parser.add_argument("--hccl-op-expansion-mode")
    parser.add_argument("--hccl-channel-engine", default="aiv")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--block", type=int, default=32)
    args = parser.parse_args()

    if args.rank_size < 1:
        raise ValueError("rank-size must be positive")
    if args.n % args.block != 0:
        raise ValueError("n must be a multiple of block")

    if args.backend == "hccl_peer_mem":
        gin_runtime.install_hccl_memory_allocator(
            runtime_library=RUNTIME_LIB,
            hccl_library=args.hccl_lib,
        )
    torch.npu.set_device(args.device)
    mask = backend_mask(args.backend)

    gin = None
    tile = None
    tile_comm = ctypes.c_void_p()
    try:
        if args.backend == "tilexr":
            gin, tile, tile_comm, _signal_window = create_tilexr_gin(args)
        else:
            gin, tile, tile_comm, _signal_window = create_hccl_gin(args)
        comm_h = gin.dev_comm
        x = torch.arange(args.n, dtype=torch.float32, device="npu") + args.rank * 1000.0
        x_offset = torch.arange(args.n * 2, dtype=torch.float32, device="npu") + args.rank * 5000.0
        y_numel = SLOTS * args.rank_size * args.n
        y = allocate_window_tensor(args, gin, y_numel, -777.0)
        z = torch.full((args.n,), -777.0, dtype=torch.float32, device="npu")
        z_offset = torch.full((args.n * 2,), -777.0, dtype=torch.float32, device="npu")
        meta = torch.full((4,), -1, dtype=torch.int64, device="npu")
        win_h = register_window(gin, args, y, y_numel, args.tag)

        blocks = triton.cdiv(args.n, args.block)
        grid = (1,)
        prefix = f"/tmp/triton_gin_{args.backend}_op_cov_{args.tag}"

        reset_gin_signals_kernel[grid](
            comm_h, args.rank_size, blocks, BACKEND_MASK=mask
        )
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "reset")

        direct_put_kernel[grid](
            x, y, comm_h, win_h, args.n, BLOCK=args.block, BLOCKS=blocks,
            RANK_SIZE_C=args.rank_size, BACKEND_MASK=mask,
        )
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "direct_put")

        direct_put_offset_kernel[grid](x_offset, y, comm_h, win_h, args.n,
                                       BLOCK=args.block, BLOCKS=blocks,
                                       SLOT=SLOT_DIRECT_PUT_OFFSET,
                                       RANK_SIZE_C=args.rank_size,
                                       BACKEND_MASK=mask)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "direct_put_offset")

        put_signal_kernel[grid](x, y, comm_h, win_h, args.n, BLOCK=args.block,
                                BLOCKS=blocks, SLOT=SLOT_PUT_SIGNAL,
                                RANK_SIZE_C=args.rank_size, SIGNAL_BASE=8,
                                BACKEND_MASK=mask)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "put_signal")

        prepare_put_window_source_kernel[grid](
            y, comm_h, args.n, BLOCK=args.block, BLOCKS=blocks,
            SLOT=SLOT_PUT_WINDOW, RANK_SIZE_C=args.rank_size,
        )
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "put_window_source_ready")

        put_window_kernel[grid](
            y, comm_h, win_h, args.n, BLOCK=args.block, BLOCKS=blocks,
            SLOT=SLOT_PUT_WINDOW, RANK_SIZE_C=args.rank_size,
            BACKEND_MASK=mask,
        )
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "put_window")

        put_window_inline_kernel[grid](y, comm_h, win_h, args.n,
                                       BLOCK=args.block, BLOCKS=blocks,
                                       SLOT=SLOT_PUT_WINDOW_INLINE,
                                       RANK_SIZE_C=args.rank_size,
                                       BACKEND_MASK=mask)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "put_window_inline")

        put_signal_window_kernel[grid](y, comm_h, win_h, args.n, BLOCK=args.block,
                                       BLOCKS=blocks, SLOT=SLOT_PUT_SIGNAL_WINDOW,
                                       RANK_SIZE_C=args.rank_size, SIGNAL_BASE=24,
                                       BACKEND_MASK=mask)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "put_signal_window")

        prepare_get_source_kernel[grid](
            y, comm_h, win_h, args.n, BLOCK=args.block, BLOCKS=blocks,
            SLOT=SLOT_GET_SOURCE, RANK_SIZE_C=args.rank_size,
        )
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "get_source_ready")

        get_kernel[grid](
            z, comm_h, win_h, args.n, BLOCK=args.block, BLOCKS=blocks,
            SLOT=SLOT_GET_SOURCE, RANK_SIZE_C=args.rank_size,
            BACKEND_MASK=mask,
        )
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "get_done")

        get_offset_kernel[grid](z_offset, comm_h, win_h, args.n,
                                BLOCK=args.block, BLOCKS=blocks,
                                SLOT=SLOT_GET_SOURCE,
                                RANK_SIZE_C=args.rank_size,
                                BACKEND_MASK=mask)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "get_offset_done")

        signal_read_barrier_kernel[grid](
            meta, comm_h, SIGNAL_ID=48, RANK_SIZE_C=args.rank_size,
            BACKEND_MASK=mask
        )
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "signal_done")

        y_cpu = y.cpu()
        z_cpu = z.cpu()
        z_offset_cpu = z_offset.cpu()
        meta_cpu = meta.cpu()
        check_slot("gin_put", y_cpu, SLOT_DIRECT_PUT, args.n, args.rank_size,
                   expected_ring_slot(args.n, args.rank_size, args.rank, 1000.0))
        check_slot("gin_put_offset", y_cpu, SLOT_DIRECT_PUT_OFFSET, args.n,
                   args.rank_size,
                   expected_ring_slot(args.n, args.rank_size, args.rank, 5000.0,
                                      start=args.n))
        check_slot("gin_put_signal", y_cpu, SLOT_PUT_SIGNAL, args.n, args.rank_size,
                   expected_ring_slot(args.n, args.rank_size, args.rank, 1000.0))
        check_slot("gin_put_window", y_cpu, SLOT_PUT_WINDOW, args.n, args.rank_size,
                   expected_ring_slot(args.n, args.rank_size, args.rank, 2000.0))
        check_slot("gin_put_window_inline", y_cpu, SLOT_PUT_WINDOW_INLINE,
                   args.n, args.rank_size,
                   expected_ring_slot(args.n, args.rank_size, args.rank, 6000.0))
        check_slot("gin_put_signal_window", y_cpu, SLOT_PUT_SIGNAL_WINDOW, args.n, args.rank_size,
                   expected_ring_slot(args.n, args.rank_size, args.rank, 4000.0))

        peer = (args.rank + 1) % args.rank_size
        expected_get = torch.arange(args.n, dtype=torch.float32) + peer * 3000.0
        get_diff = (z_cpu - expected_get).abs()
        get_maxerr = float(get_diff.max())
        if get_maxerr != 0.0:
            bad = int(get_diff.argmax())
            raise AssertionError(
                f"gin_get failed maxerr={get_maxerr} bad={bad} got={float(z_cpu[bad])} "
                f"expected={float(expected_get[bad])}"
            )
        print("gin_get: PASS maxerr=0.0")

        get_offset_window = z_offset_cpu[args.n:args.n * 2]
        get_offset_diff = (get_offset_window - expected_get).abs()
        get_offset_maxerr = float(get_offset_diff.max())
        if get_offset_maxerr != 0.0:
            bad = int(get_offset_diff.argmax())
            raise AssertionError(
                f"gin_get_offset failed maxerr={get_offset_maxerr} bad={bad} "
                f"got={float(get_offset_window[bad])} expected={float(expected_get[bad])}"
            )
        print("gin_get_offset: PASS maxerr=0.0")

        expected_signal = (args.rank - 1 + args.rank_size) % args.rank_size + 101
        if int(meta_cpu[0]) != expected_signal or int(meta_cpu[1]) != args.rank or int(meta_cpu[2]) != args.rank_size:
            raise AssertionError(f"signal/read/barrier metadata mismatch: {meta_cpu.tolist()}")
        print(f"gin_signal/read_signal/barrier: PASS meta={meta_cpu.tolist()}")
        print(f"GIN OP coverage runtime: PASS rank={args.rank} backend={args.backend}")
    finally:
        if gin is not None:
            gin.close()
        if args.backend in ("hccl_peer_mem", "hccl_channel") and tile_comm.value:
            gin_runtime.destroy_hccl_comm(tile_comm, hccl_library=args.hccl_lib)
        if tile is not None and tile_comm.value:
            tile.TileXRCommDestroy(tile_comm)


if __name__ == "__main__":
    main()



