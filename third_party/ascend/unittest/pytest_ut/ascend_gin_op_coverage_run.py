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


TILEXR_LIB = "/home/kaixin/TileXR/install/lib64/libtile-comm.so"
RUNTIME_LIB = os.getenv("TRITON_ASCEND_GIN_RUNTIME_LIB")
MAX_RANKS = 2
SLOT_DIRECT_PUT = 0
SLOT_PUT_SIGNAL = 1
SLOT_PUT_WINDOW = 2
SLOT_GET_SOURCE = 3
SLOT_PUT_SIGNAL_WINDOW = 4
SLOT_DIRECT_PUT_OFFSET = 5
SLOT_PUT_WINDOW_INLINE = 6
SLOTS = 7


@triton.jit
def reset_gin_signals_kernel(comm_h, MAX_RANKS_C: tl.constexpr, SIGNALS: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    nranks = tgin.num_ranks(dev)
    token = tgin.token()
    for signal_id in tl.static_range(0, SIGNALS):
        for peer in tl.static_range(0, MAX_RANKS_C):
            if peer < nranks:
                token = tgin.reset_signal(gin, peer, signal_id=signal_id, token=token)
    token = tgin.flush(gin, token=token)


@triton.jit
def direct_put_kernel(x, y, comm_h, win_h, n_elements: tl.constexpr,
                      BLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    peer = 1 - rank
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
                             SLOT: tl.constexpr, RANK_SIZE_C: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    peer = 1 - rank
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
                      SIGNAL_BASE: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    peer = 1 - rank
    token = tgin.token()

    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        vals = tl.load(x + offs)
        base = (2 + rank) * n_elements
        tl.store(y + base + offs, vals)

    base = (2 + rank) * n_elements
    transfer_bytes = n_elements * 4
    token = tgin.put_signal(
        gin, peer, win, base * 4, x, transfer_bytes,
        signal_id=SIGNAL_BASE, signal_value=transfer_bytes, token=token)
    token = tgin.flush(gin, peer=peer, token=token)
    token = tgin.wait_signal(gin, peer, signal_id=SIGNAL_BASE,
                              least_value=transfer_bytes, token=token)


@triton.jit
def prepare_put_window_source_kernel(y, comm_h, n_elements: tl.constexpr,
                                     BLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    rank = tgin.rank(dev)
    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        base = (4 + rank) * n_elements
        vals = offs.to(tl.float32) + rank.to(tl.float32) * 2000.0
        tl.store(y + base + offs, vals)


@triton.jit
def put_window_kernel(y, comm_h, win_h, n_elements: tl.constexpr,
                      BLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    peer = 1 - rank
    token = tgin.token()
    tile_bytes = BLOCK * 4

    for pid in tl.static_range(0, BLOCKS):
        base = (4 + rank) * n_elements
        byte_off = (base + pid * BLOCK) * 4
        token = tgin.put_window(gin, peer, win, byte_off, win, byte_off,
                                 tile_bytes, token=token)
    token = tgin.flush(gin, peer=peer, token=token)


@triton.jit
def put_window_inline_kernel(y, comm_h, win_h, n_elements: tl.constexpr,
                             BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                             SLOT: tl.constexpr, RANK_SIZE_C: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    peer = 1 - rank
    token = tgin.token()
    tile_bytes = BLOCK * 4

    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        base = (SLOT * RANK_SIZE_C + rank) * n_elements
        vals = offs.to(tl.float32) + rank.to(tl.float32) * 6000.0
        tl.store(y + base + offs, vals)
        byte_off = (base + pid * BLOCK) * 4
        token = tgin.put_window(gin, peer, win, byte_off, win, byte_off,
                                 tile_bytes, token=token)
    token = tgin.flush(gin, peer=peer, token=token)


@triton.jit
def put_signal_window_kernel(y, comm_h, win_h, n_elements: tl.constexpr,
                             BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                             SIGNAL_BASE: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    peer = 1 - rank
    token = tgin.token()
    tile_bytes = BLOCK * 4

    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        base = (8 + rank) * n_elements
        vals = offs.to(tl.float32) + rank.to(tl.float32) * 4000.0
        tl.store(y + base + offs, vals)
        byte_off = (base + pid * BLOCK) * 4
        token = tgin.put_signal_window(
            gin, peer, win, byte_off, win, byte_off, tile_bytes,
            signal_id=SIGNAL_BASE + pid, signal_value=tile_bytes, token=token)

    token = tgin.flush(gin, peer=peer, token=token)
    for pid in tl.static_range(0, BLOCKS):
        token = tgin.wait_signal(gin, peer, signal_id=SIGNAL_BASE + pid,
                                  least_value=tile_bytes, token=token)


@triton.jit
def prepare_get_source_kernel(y, comm_h, win_h, n_elements: tl.constexpr,
                              BLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    rank = tgin.rank(dev)
    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        base = (6 + rank) * n_elements
        vals = offs.to(tl.float32) + rank.to(tl.float32) * 3000.0
        tl.store(y + base + offs, vals)


@triton.jit
def get_kernel(z, comm_h, win_h, n_elements: tl.constexpr,
               BLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    win = tgin.window(win_h)
    rank = tgin.rank(dev)
    peer = 1 - rank
    token = tgin.token()

    src_base = (6 + peer) * n_elements
    token = tgin.get(gin, peer, z, win, src_base * 4, n_elements * 4,
                      token=token)
    token = tgin.flush(gin, peer=peer, token=token)


@triton.jit
def get_offset_kernel(z, comm_h, win_h, n_elements: tl.constexpr,
                      BLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    win = tgin.window(win_h)
    rank = tgin.rank(dev)
    peer = 1 - rank
    token = tgin.token()
    tile_bytes = BLOCK * 4
    src_base = (6 + peer) * n_elements

    for pid in tl.static_range(0, BLOCKS):
        token = tgin.get(gin, peer, z + n_elements + pid * BLOCK, win,
                          (src_base + pid * BLOCK) * 4, tile_bytes,
                          token=token)
    token = tgin.flush(gin, peer=peer, token=token)


@triton.jit
def signal_read_barrier_kernel(meta, comm_h, SIGNAL_ID: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    rank = tgin.rank(dev)
    nranks = tgin.num_ranks(dev)
    peer = 1 - rank
    token = tgin.token()

    token = tgin.signal(gin, peer, signal_id=SIGNAL_ID,
                         signal_value=rank + 101, token=token)
    token = tgin.flush(gin, peer=peer, token=token)
    token = tgin.barrier(gin, token=token)
    token = tgin.wait_signal(gin, peer, signal_id=SIGNAL_ID,
                              least_value=peer + 101, token=token)
    value = tgin.read_signal(gin, peer, signal_id=SIGNAL_ID)
    idx = tl.arange(0, 1)
    tl.store(meta + idx, value.to(tl.int64))
    tl.store(meta + 1 + idx, rank.to(tl.int64))
    tl.store(meta + 2 + idx, nranks.to(tl.int64))
    tl.store(meta + 3 + idx, token.to(tl.int64))
    token = tgin.reset_signal(gin, peer, signal_id=SIGNAL_ID, token=token)
    token = tgin.flush(gin, peer=peer, token=token)


def bind_tilexr():
    lib = ctypes.CDLL(TILEXR_LIB, mode=ctypes.RTLD_GLOBAL)
    lib.TileXRCommInitRankLocal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    lib.TileXRCommInitRankLocal.restype = ctypes.c_int
    lib.TileXRCommDestroy.argtypes = [ctypes.c_void_p]
    lib.TileXRCommDestroy.restype = ctypes.c_int
    return lib


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
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--block", type=int, default=32)
    args = parser.parse_args()

    if args.rank_size != 2:
        raise ValueError("this coverage runner currently validates the two-rank path")
    if args.n % args.block != 0:
        raise ValueError("n must be a multiple of block")

    torch.npu.set_device(args.device)
    tile = bind_tilexr()
    tile_comm = ctypes.c_void_p()
    ret = tile.TileXRCommInitRankLocal(args.rank_size, args.rank, ctypes.byref(tile_comm))
    if ret != 0 or not tile_comm.value:
        raise RuntimeError(f"TileXRCommInitRankLocal failed rank={args.rank} ret={ret}")

    gin = None
    try:
        gin = gin_runtime.create_from_tilexr(
            tile_comm,
            runtime_library=RUNTIME_LIB,
            tilexr_library=TILEXR_LIB,
        )
        comm_h = gin.dev_comm
        x = torch.arange(args.n, dtype=torch.float32, device="npu") + args.rank * 1000.0
        x_offset = torch.arange(args.n * 2, dtype=torch.float32, device="npu") + args.rank * 5000.0
        y = torch.full((SLOTS * args.rank_size * args.n,), -777.0, dtype=torch.float32, device="npu")
        z = torch.full((args.n,), -777.0, dtype=torch.float32, device="npu")
        z_offset = torch.full((args.n * 2,), -777.0, dtype=torch.float32, device="npu")
        meta = torch.full((4,), -1, dtype=torch.int64, device="npu")
        win_h = gin.window_handle(y, rendezvous_id=args.tag)

        blocks = triton.cdiv(args.n, args.block)
        grid = (1,)
        prefix = f"/tmp/triton_comm_gin_cov_{args.tag}"

        reset_gin_signals_kernel[grid](comm_h, args.rank_size, 64)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "reset")

        direct_put_kernel[grid](x, y, comm_h, win_h, args.n, BLOCK=args.block, BLOCKS=blocks)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "direct_put")

        direct_put_offset_kernel[grid](x_offset, y, comm_h, win_h, args.n,
                                       BLOCK=args.block, BLOCKS=blocks,
                                       SLOT=SLOT_DIRECT_PUT_OFFSET,
                                       RANK_SIZE_C=args.rank_size)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "direct_put_offset")

        put_signal_kernel[grid](x, y, comm_h, win_h, args.n, BLOCK=args.block,
                                BLOCKS=blocks, SIGNAL_BASE=8)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "put_signal")

        prepare_put_window_source_kernel[grid](y, comm_h, args.n, BLOCK=args.block, BLOCKS=blocks)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "put_window_source_ready")

        put_window_kernel[grid](y, comm_h, win_h, args.n, BLOCK=args.block, BLOCKS=blocks)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "put_window")

        put_window_inline_kernel[grid](y, comm_h, win_h, args.n,
                                       BLOCK=args.block, BLOCKS=blocks,
                                       SLOT=SLOT_PUT_WINDOW_INLINE,
                                       RANK_SIZE_C=args.rank_size)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "put_window_inline")

        put_signal_window_kernel[grid](y, comm_h, win_h, args.n, BLOCK=args.block,
                                       BLOCKS=blocks, SIGNAL_BASE=24)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "put_signal_window")

        prepare_get_source_kernel[grid](y, comm_h, win_h, args.n, BLOCK=args.block, BLOCKS=blocks)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "get_source_ready")

        get_kernel[grid](z, comm_h, win_h, args.n, BLOCK=args.block, BLOCKS=blocks)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "get_done")

        get_offset_kernel[grid](z_offset, comm_h, win_h, args.n,
                                BLOCK=args.block, BLOCKS=blocks)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "get_offset_done")

        signal_read_barrier_kernel[grid](meta, comm_h, SIGNAL_ID=48)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "signal_done")

        y_cpu = y.cpu()
        z_cpu = z.cpu()
        z_offset_cpu = z_offset.cpu()
        meta_cpu = meta.cpu()
        check_slot("gin_put", y_cpu, SLOT_DIRECT_PUT, args.n, args.rank_size,
                   expected_slot(args.n, args.rank_size, 1000.0))
        check_slot("gin_put_offset", y_cpu, SLOT_DIRECT_PUT_OFFSET, args.n,
                   args.rank_size, expected_slot_offset(args.n, args.rank_size, 5000.0, args.n))
        check_slot("gin_put_signal", y_cpu, SLOT_PUT_SIGNAL, args.n, args.rank_size,
                   expected_slot(args.n, args.rank_size, 1000.0))
        check_slot("gin_put_window", y_cpu, SLOT_PUT_WINDOW, args.n, args.rank_size,
                   expected_slot(args.n, args.rank_size, 2000.0))
        check_slot("gin_put_window_inline", y_cpu, SLOT_PUT_WINDOW_INLINE,
                   args.n, args.rank_size,
                   expected_slot(args.n, args.rank_size, 6000.0))
        check_slot("gin_put_signal_window", y_cpu, SLOT_PUT_SIGNAL_WINDOW, args.n, args.rank_size,
                   expected_slot(args.n, args.rank_size, 4000.0))

        peer = 1 - args.rank
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

        expected_signal = peer + 101
        if int(meta_cpu[0]) != expected_signal or int(meta_cpu[1]) != args.rank or int(meta_cpu[2]) != args.rank_size:
            raise AssertionError(f"signal/read/barrier metadata mismatch: {meta_cpu.tolist()}")
        print(f"gin_signal/read_signal/barrier: PASS meta={meta_cpu.tolist()}")
        print(f"GIN OP coverage runtime: PASS rank={args.rank}")
    finally:
        if gin is not None:
            gin.close()
        if tile_comm.value:
            tile.TileXRCommDestroy(tile_comm)


if __name__ == "__main__":
    main()



