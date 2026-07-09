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

from ascend_gin_fused_all_gather import fused_all_gather_scale_kernel


TILEXR_LIB = "/home/kaixin/TileXR/install/lib64/libtile-comm.so"
RUNTIME_LIB = os.getenv("TRITON_ASCEND_GIN_RUNTIME_LIB")


@triton.jit
def reset_gin_signals_kernel(comm_h, MAX_RANKS: tl.constexpr, BLOCKS: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    nranks = tgin.num_ranks(dev)
    token = tgin.token()
    for pid in tl.static_range(0, BLOCKS):
        for peer in tl.static_range(0, MAX_RANKS):
            if peer < nranks:
                token = tgin.reset_signal(gin, peer, signal_id=pid, token=token)
    token = tgin.flush(gin, token=token)


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
        raise ValueError("this runner currently validates the two-rank path")
    if args.n % args.block != 0:
        raise ValueError("n must be a multiple of block for the current byte-copy smoke")

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
        x = torch.arange(args.n, dtype=torch.float32, device="npu") + args.rank * 100.0
        y = torch.full((args.rank_size * args.n,), -777.0, dtype=torch.float32, device="npu")
        win_h = gin.window_handle(y, rendezvous_id=args.tag)

        blocks = triton.cdiv(args.n, args.block)
        grid = (1,)
        reset_gin_signals_kernel[grid](comm_h, args.rank_size, blocks)
        torch.npu.synchronize()

        prefix = f"/tmp/triton_comm_allgather_{args.tag}"
        wait_files(prefix, args.rank_size, args.rank, "reset")

        fused_all_gather_scale_kernel[grid](
            x,
            y,
            comm_h,
            win_h,
            args.n,
            args.rank_size,
            BLOCK=args.block,
            BLOCKS=blocks,
        )
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "kernel_done")

        expected_chunks = []
        for source_rank in range(args.rank_size):
            expected_chunks.append((torch.arange(args.n, dtype=torch.float32) + source_rank * 100.0) * 0.5)
        expected = torch.cat(expected_chunks)
        got = y.cpu()
        diff = (got - expected).abs()
        maxerr = float(diff.max())
        bad_idx = int(diff.argmax()) if diff.numel() else -1
        print(f"rank={args.rank} device={args.device} comm_h=0x{comm_h:x} win_h={win_h} maxerr={maxerr}")
        print("got_head=", got[: min(8, got.numel())].tolist())
        print("got_peer_head=", got[args.n : args.n + min(8, args.n)].tolist())
        if bad_idx >= 0:
            print("bad_idx=", bad_idx, "got=", float(got[bad_idx]), "expected=", float(expected[bad_idx]))
            print("got_tail=", got[max(0, got.numel() - 8) :].tolist())
        if maxerr != 0.0:
            raise AssertionError(f"fused all-gather check failed maxerr={maxerr}")
        wait_files(prefix, args.rank_size, args.rank, "checked")
        print(f"fused all-gather user-window runtime: PASS rank={args.rank}")
    finally:
        if gin is not None:
            gin.close()
        if tile_comm.value:
            tile.TileXRCommDestroy(tile_comm)


if __name__ == "__main__":
    main()




