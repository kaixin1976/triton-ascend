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
SIGNAL_BASE_ALLREDUCE = 0
SIGNAL_BASE_ALL2ALL = 64
SIGNAL_SLOTS = 128


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
def gin_allreduce_sum_kernel(x, staging, out, comm_h, win_h,
                             n_elements: tl.constexpr, BLOCK: tl.constexpr,
                             BLOCKS: tl.constexpr, SIGNAL_BASE: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    win = tgin.window(win_h, ptr=staging)
    rank = tgin.rank(dev)
    nranks = tgin.num_ranks(dev)
    peer = 1 - rank
    token = tgin.token()
    tile_bytes = BLOCK * 4

    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        vals = tl.load(x + offs)
        local_base = rank * n_elements
        tl.store(staging + local_base + offs, vals)

        dst_byte_off = (local_base + pid * BLOCK) * 4
        if peer < nranks:
            token = tgin.put_signal(
                gin,
                peer,
                win,
                dst_byte_off,
                x + pid * BLOCK,
                tile_bytes,
                signal_id=SIGNAL_BASE + pid,
                signal_value=tile_bytes,
                token=token,
            )

    token = tgin.flush(gin, peer=peer, token=token)

    for pid in tl.static_range(0, BLOCKS):
        if peer < nranks:
            token = tgin.wait_signal(
                gin,
                peer,
                signal_id=SIGNAL_BASE + pid,
                least_value=tile_bytes,
                token=token,
            )

    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        acc = tl.load(staging + offs)
        acc += tl.load(staging + n_elements + offs)
        tl.store(out + offs, acc)


@triton.jit
def gin_all2all_kernel(x, y, comm_h, win_h, n_elements: tl.constexpr,
                       BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
                       SIGNAL_BASE: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    win = tgin.window(win_h, ptr=y)
    rank = tgin.rank(dev)
    nranks = tgin.num_ranks(dev)
    peer = 1 - rank
    token = tgin.token()
    tile_bytes = BLOCK * 4

    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        self_base = rank * n_elements
        vals = tl.load(x + self_base + offs)
        tl.store(y + self_base + offs, vals)

        dst_base = rank * n_elements
        src_base = peer * n_elements
        dst_byte_off = (dst_base + pid * BLOCK) * 4
        if peer < nranks:
            token = tgin.put_signal(
                gin,
                peer,
                win,
                dst_byte_off,
                x + src_base + pid * BLOCK,
                tile_bytes,
                signal_id=SIGNAL_BASE + pid,
                signal_value=tile_bytes,
                token=token,
            )

    token = tgin.flush(gin, peer=peer, token=token)

    for pid in tl.static_range(0, BLOCKS):
        if peer < nranks:
            token = tgin.wait_signal(
                gin,
                peer,
                signal_id=SIGNAL_BASE + pid,
                least_value=tile_bytes,
                token=token,
            )


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


def check_exact(name, got, expected):
    diff = (got - expected).abs()
    maxerr = float(diff.max()) if diff.numel() else 0.0
    if maxerr != 0.0:
        bad = int(diff.argmax())
        ctx_start = max(0, bad - 4)
        ctx_end = min(int(got.numel()), bad + 5)
        raise AssertionError(
            f"{name} failed maxerr={maxerr} bad={bad} "
            f"got={float(got[bad])} expected={float(expected[bad])} "
            f"got_ctx={got[ctx_start:ctx_end].tolist()} "
            f"expected_ctx={expected[ctx_start:ctx_end].tolist()}"
        )
    print(f"{name}: PASS maxerr=0.0")


def run_allreduce(args, gin, comm_h):
    x = torch.arange(args.n, dtype=torch.float32, device="npu") + args.rank * 1000.0
    staging = torch.full((args.rank_size * args.n,), -777.0, dtype=torch.float32, device="npu")
    out = torch.full((args.n,), -777.0, dtype=torch.float32, device="npu")
    win_h = gin.window_handle(staging, rendezvous_id=f"{args.tag}_allreduce")
    blocks = triton.cdiv(args.n, args.block)
    grid = (1,)

    gin_allreduce_sum_kernel[grid](
        x,
        staging,
        out,
        comm_h,
        win_h,
        args.n,
        BLOCK=args.block,
        BLOCKS=blocks,
        SIGNAL_BASE=SIGNAL_BASE_ALLREDUCE,
    )
    torch.npu.synchronize()

    expected = torch.arange(args.n, dtype=torch.float32)
    expected = expected + (torch.arange(args.n, dtype=torch.float32) + 1000.0)
    check_exact("gin_allreduce_sum", out.cpu(), expected)


def run_all2all(args, gin, comm_h):
    x_chunks = []
    for dst_rank in range(args.rank_size):
        x_chunks.append(torch.arange(args.n, dtype=torch.float32) + args.rank * 1000.0 + dst_rank * 100.0)
    x = torch.cat(x_chunks).to("npu")
    y = torch.full((args.rank_size * args.n,), -777.0, dtype=torch.float32, device="npu")
    win_h = gin.window_handle(y, rendezvous_id=f"{args.tag}_all2all")
    blocks = triton.cdiv(args.n, args.block)
    grid = (1,)

    gin_all2all_kernel[grid](
        x,
        y,
        comm_h,
        win_h,
        args.n,
        BLOCK=args.block,
        BLOCKS=blocks,
        SIGNAL_BASE=SIGNAL_BASE_ALL2ALL,
    )
    torch.npu.synchronize()

    expected_chunks = []
    for source_rank in range(args.rank_size):
        expected_chunks.append(torch.arange(args.n, dtype=torch.float32) + source_rank * 1000.0 + args.rank * 100.0)
    expected = torch.cat(expected_chunks)
    check_exact("gin_all2all", y.cpu(), expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rank-size", type=int, default=2)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--case", choices=("allreduce", "all2all", "both"), default="both")
    args = parser.parse_args()

    if args.rank_size != 2:
        raise ValueError("this runner currently validates the two-rank path")
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
        blocks = triton.cdiv(args.n, args.block)
        grid = (1,)
        reset_gin_signals_kernel[grid](comm_h, args.rank_size, SIGNAL_SLOTS)
        torch.npu.synchronize()

        prefix = f"/tmp/triton_comm_gin_collective_{args.tag}"
        wait_files(prefix, args.rank_size, args.rank, "reset")

        if args.case in ("allreduce", "both"):
            run_allreduce(args, gin, comm_h)
            wait_files(prefix, args.rank_size, args.rank, "allreduce_done")

        if args.case in ("all2all", "both"):
            run_all2all(args, gin, comm_h)
            wait_files(prefix, args.rank_size, args.rank, "all2all_done")

        print(f"GIN collective patterns runtime: PASS rank={args.rank} case={args.case}")
    finally:
        if gin is not None:
            gin.close()
        if tile_comm.value:
            tile.TileXRCommDestroy(tile_comm)


if __name__ == "__main__":
    main()
