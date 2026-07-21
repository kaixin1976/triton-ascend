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
RUNTIME_LIB = os.getenv("TRITON_ASCEND_GIN_RUNTIME_LIB")

SIGNAL_BASE_DISPATCH = 0
SIGNAL_BASE_COMBINE = 64
SIGNAL_SLOTS = 128


@triton.jit
def reset_gin_signals_kernel(
    comm_h,
    MAX_RANKS: tl.constexpr,
    SIGNAL_BASE: tl.constexpr,
    SIGNALS: tl.constexpr,
):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    nranks = tgin.num_ranks(dev)
    token = tgin.token()
    for signal_idx in tl.static_range(0, SIGNALS):
        signal_id = SIGNAL_BASE + signal_idx
        for peer in tl.static_range(0, MAX_RANKS):
            if peer < nranks:
                token = tgin.reset_signal(gin, peer, signal_id=signal_id, token=token)
    token = tgin.flush(gin, token=token)


@triton.jit
def moe_dispatch_expert_combine_kernel(
    x,
    arena,
    comm_h,
    win_h,
    n_elements: tl.constexpr,
    MAX_RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCKS: tl.constexpr,
    DISPATCH_BASE: tl.constexpr,
    COMBINE_BASE: tl.constexpr,
    SEND_SCRATCH_BASE: tl.constexpr,
    RETURN_SCRATCH_BASE: tl.constexpr,
    SIGNAL_DISPATCH: tl.constexpr,
    SIGNAL_COMBINE: tl.constexpr,
):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    win = tgin.window(win_h, ptr=arena)
    rank = tgin.rank(dev)
    nranks = tgin.num_ranks(dev)
    token = tgin.token()
    tile_bytes = BLOCK * 4
    expert_scale = rank.to(tl.float32) + 1.0

    # Stage 1: origin rank pushes one token block per expert rank.
    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        for dst in tl.static_range(0, MAX_RANKS):
            if dst < nranks:
                vals = tl.load(x + dst * n_elements + offs)
                send_base = SEND_SCRATCH_BASE + dst * n_elements
                tl.store(arena + send_base + offs, vals)

                if dst == rank:
                    local_inbox_base = DISPATCH_BASE + rank * n_elements
                    tl.store(arena + local_inbox_base + offs, vals)
                else:
                    tl.debug_barrier()
                    src_byte_off = (send_base + pid * BLOCK) * 4
                    dst_byte_off = (DISPATCH_BASE + rank * n_elements + pid * BLOCK) * 4
                    token = tgin.put_signal_window(
                        gin,
                        dst,
                        win,
                        dst_byte_off,
                        win,
                        src_byte_off,
                        tile_bytes,
                        signal_id=SIGNAL_DISPATCH + pid,
                        signal_value=tile_bytes,
                        token=token,
                    )

    token = tgin.flush(gin, token=token)

    # Stage 2: expert rank waits for its inbound token blocks.
    for pid in tl.static_range(0, BLOCKS):
        for src in tl.static_range(0, MAX_RANKS):
            if src < nranks and src != rank:
                token = tgin.wait_signal(
                    gin,
                    src,
                    signal_id=SIGNAL_DISPATCH + pid,
                    least_value=tile_bytes,
                    token=token,
                )

    # Stage 3: expert compute is fused here, then results are pushed back.
    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        for src in tl.static_range(0, MAX_RANKS):
            if src < nranks:
                inbox_base = DISPATCH_BASE + src * n_elements
                vals = tl.load(arena + inbox_base + offs)
                result = vals * expert_scale + src * 0.25
                return_base = RETURN_SCRATCH_BASE + src * n_elements
                tl.store(arena + return_base + offs, result)

                if src == rank:
                    local_out_base = COMBINE_BASE + rank * n_elements
                    tl.store(arena + local_out_base + offs, result)
                else:
                    tl.debug_barrier()
                    src_byte_off = (return_base + pid * BLOCK) * 4
                    dst_byte_off = (COMBINE_BASE + rank * n_elements + pid * BLOCK) * 4
                    token = tgin.put_signal_window(
                        gin,
                        src,
                        win,
                        dst_byte_off,
                        win,
                        src_byte_off,
                        tile_bytes,
                        signal_id=SIGNAL_COMBINE + pid,
                        signal_value=tile_bytes,
                        token=token,
                    )

    token = tgin.flush(gin, token=token)

    # Stage 4: origin rank waits for all expert outputs in its combine window.
    for pid in tl.static_range(0, BLOCKS):
        for expert in tl.static_range(0, MAX_RANKS):
            if expert < nranks and expert != rank:
                token = tgin.wait_signal(
                    gin,
                    expert,
                    signal_id=SIGNAL_COMBINE + pid,
                    least_value=tile_bytes,
                    token=token,
                )


def bind_tilexr(tilexr_lib):
    lib = ctypes.CDLL(tilexr_lib, mode=ctypes.RTLD_GLOBAL)
    lib.TileXRCommInitRankLocal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    lib.TileXRCommInitRankLocal.restype = ctypes.c_int
    lib.TileXRCommDestroy.argtypes = [ctypes.c_void_p]
    lib.TileXRCommDestroy.restype = ctypes.c_int
    return lib


def wait_files(prefix, rank_size, rank, phase, timeout_s=180):
    gin_runtime.rendezvous_barrier(prefix, rank_size, rank, phase, timeout_s)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rank-size", type=int, default=2)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tilexr-lib", default=TILEXR_LIB)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--block", type=int, default=32)
    args = parser.parse_args()

    if args.rank_size < 2:
        raise ValueError("rank-size must be at least 2")
    if args.rank < 0 or args.rank >= args.rank_size:
        raise ValueError("rank must be in [0, rank-size)")
    if args.n % args.block != 0:
        raise ValueError("n must be a multiple of block")
    blocks = triton.cdiv(args.n, args.block)
    if SIGNAL_BASE_COMBINE + blocks > SIGNAL_SLOTS:
        raise ValueError("not enough signal slots for the selected n/block")

    torch.npu.set_device(args.device)
    tile = bind_tilexr(args.tilexr_lib)
    tile_comm = ctypes.c_void_p()
    gin = None

    try:
        ret = tile.TileXRCommInitRankLocal(args.rank_size, args.rank, ctypes.byref(tile_comm))
        if ret != 0 or not tile_comm.value:
            raise RuntimeError(f"TileXRCommInitRankLocal failed rank={args.rank} ret={ret}")
        gin = gin_runtime.create_from_tilexr(
            tile_comm,
            runtime_library=RUNTIME_LIB,
            tilexr_library=args.tilexr_lib,
            signal_slots=SIGNAL_SLOTS,
        )
        comm_h = gin.dev_comm

        segment_numel = args.rank_size * args.n
        dispatch_base = 0
        combine_base = segment_numel
        send_scratch_base = 2 * segment_numel
        return_scratch_base = 3 * segment_numel
        arena_numel = 4 * segment_numel

        x_chunks = [
            torch.arange(args.n, dtype=torch.float32, device="npu") + args.rank * 1000.0 + expert * 100.0
            for expert in range(args.rank_size)
        ]
        x = torch.cat(x_chunks)
        arena = torch.full((arena_numel,), -777.0, dtype=torch.float32, device="npu")
        win_h = gin.window_handle(arena, rendezvous_id=f"{args.tag}_moe")

        prefix = f"/tmp/triton_gin_moe_{args.tag}"
        grid = (1,)
        reset_gin_signals_kernel[grid](
            comm_h,
            args.rank_size,
            SIGNAL_BASE_DISPATCH,
            blocks,
        )
        reset_gin_signals_kernel[grid](
            comm_h,
            args.rank_size,
            SIGNAL_BASE_COMBINE,
            blocks,
        )
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "reset")

        moe_dispatch_expert_combine_kernel[grid](
            x,
            arena,
            comm_h,
            win_h,
            args.n,
            args.rank_size,
            BLOCK=args.block,
            BLOCKS=blocks,
            DISPATCH_BASE=dispatch_base,
            COMBINE_BASE=combine_base,
            SEND_SCRATCH_BASE=send_scratch_base,
            RETURN_SCRATCH_BASE=return_scratch_base,
            SIGNAL_DISPATCH=SIGNAL_BASE_DISPATCH,
            SIGNAL_COMBINE=SIGNAL_BASE_COMBINE,
        )
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "kernel_done")

        expected_chunks = []
        base = torch.arange(args.n, dtype=torch.float32)
        for expert in range(args.rank_size):
            vals = base + args.rank * 1000.0 + expert * 100.0
            expected_chunks.append(vals * float(expert + 1) + args.rank * 0.25)
        expected = torch.cat(expected_chunks)
        got = arena[combine_base : combine_base + segment_numel].cpu()
        check_exact("gin_moe_dispatch_combine", got, expected)
        wait_files(prefix, args.rank_size, args.rank, "checked")
        print(
            f"GIN MoE dispatch/combine: PASS rank={args.rank} "
            f"rank_size={args.rank_size} n={args.n} block={args.block}",
            flush=True,
        )
    finally:
        if gin is not None:
            gin.close()
        if tile_comm.value:
            tile.TileXRCommDestroy(tile_comm)


if __name__ == "__main__":
    main()
