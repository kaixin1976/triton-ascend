import argparse
import ctypes
import datetime
import os
import pathlib
import time

import torch
import torch_npu
import torch.distributed as dist
import triton
import triton.language as tl
import triton.language.extra.cann.gin as tgin
import triton.language.extra.cann.gin_runtime as gin_runtime

from ascend_gin_fused_all_gather import fused_all_gather_scale_kernel


TILEXR_LIB = os.getenv("TRITON_ASCEND_TILEXR_LIB", "/home/kaixin/TileXR/install/lib64/libtile-comm.so")
HCCL_LIB = os.getenv(
    "TRITON_ASCEND_HCCL_LIB",
    "/home/kaixin/Ascend/cann-9.1.0-beta.1/aarch64-linux/lib64/libhcomm.so",
)
RUNTIME_LIB = os.getenv("TRITON_ASCEND_GIN_RUNTIME_LIB")
SIGNAL_SLOTS = 128


@triton.jit
def reset_gin_signals_kernel(
    comm_h,
    MAX_RANKS: tl.constexpr,
    BLOCKS: tl.constexpr,
    BACKEND_MASK: tl.constexpr,
):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=BACKEND_MASK)
    nranks = tgin.num_ranks(dev)
    token = tgin.token()
    for pid in tl.static_range(0, BLOCKS):
        for peer in tl.static_range(0, MAX_RANKS):
            if peer < nranks:
                token = tgin.reset_signal(gin, peer, signal_id=pid, token=token)
    token = tgin.flush(gin, token=token)


def bind_tilexr(tilexr_lib):
    lib = ctypes.CDLL(tilexr_lib, mode=ctypes.RTLD_GLOBAL)
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


def backend_mask(backend):
    if backend == "tilexr":
        return tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM
    if backend == "hccl_peer_mem":
        return tgin.GIN_BACKEND_HCCL_PEER_MEM
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
    hccl_comm = gin_runtime.create_hccl_root_info_comm(
        rank=args.rank,
        rank_size=args.rank_size,
        rendezvous_id=args.tag,
        hccl_library=args.hccl_lib,
        use_config=False,
    )
    gin = gin_runtime.create_from_hccl_peer_mem(
        hccl_comm,
        runtime_library=RUNTIME_LIB,
        hccl_library=args.hccl_lib,
        signal_slots=SIGNAL_SLOTS,
    )
    signal_window = torch.zeros((args.rank_size * SIGNAL_SLOTS,), dtype=torch.int64, device="npu")
    gin.signal_window_handle(signal_window, rendezvous_id=args.tag + "_signal")
    return gin, None, ctypes.c_void_p(hccl_comm), signal_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rank-size", type=int, default=2)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--backend", choices=("tilexr", "hccl_peer_mem"), default="tilexr")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port")
    parser.add_argument("--tilexr-lib", default=TILEXR_LIB)
    parser.add_argument("--hccl-lib", default=HCCL_LIB)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--block", type=int, default=32)
    args = parser.parse_args()

    if args.rank_size < 2:
        raise ValueError("rank-size must be at least 2")
    if args.rank < 0 or args.rank >= args.rank_size:
        raise ValueError("rank must be in [0, rank-size)")
    if args.n % args.block != 0:
        raise ValueError("n must be a multiple of block for the current byte-copy smoke")

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
        x = torch.arange(args.n, dtype=torch.float32, device="npu") + args.rank * 100.0
        y = torch.full((args.rank_size * args.n,), -777.0, dtype=torch.float32, device="npu")
        win_h = gin.window_handle(y, rendezvous_id=args.tag)

        blocks = triton.cdiv(args.n, args.block)
        grid = (1,)
        reset_gin_signals_kernel[grid](comm_h, args.rank_size, blocks, BACKEND_MASK=mask)
        torch.npu.synchronize()

        prefix = f"/tmp/triton_gin_{args.backend}_allgather_{args.tag}"
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
            BACKEND_MASK=mask,
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
        print(
            f"rank={args.rank} device={args.device} backend={args.backend} "
            f"comm_h=0x{comm_h:x} win_h={win_h} maxerr={maxerr}"
        )
        print("got_head=", got[: min(8, got.numel())].tolist())
        print("got_peer_head=", got[args.n : args.n + min(8, args.n)].tolist())
        if bad_idx >= 0:
            print("bad_idx=", bad_idx, "got=", float(got[bad_idx]), "expected=", float(expected[bad_idx]))
            print("got_tail=", got[max(0, got.numel() - 8) :].tolist())
        if maxerr != 0.0:
            raise AssertionError(f"fused all-gather check failed maxerr={maxerr}")
        wait_files(prefix, args.rank_size, args.rank, "checked")
        print(f"fused all-gather user-window runtime: PASS rank={args.rank} backend={args.backend}")
    finally:
        if gin is not None:
            gin.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        if args.backend == "hccl_peer_mem" and tile_comm.value:
            gin_runtime.destroy_hccl_comm(tile_comm, hccl_library=args.hccl_lib)
        if tile is not None and tile_comm.value:
            tile.TileXRCommDestroy(tile_comm)


if __name__ == "__main__":
    main()
