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

SIGNAL_BASE_AG = 0
SIGNAL_BASE_AR = 16
SIGNAL_BASE_RS = 32
SIGNAL_SLOTS = 64


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
def allgather_gemm_kernel(
    a_local,
    b,
    c,
    a_arena,
    comm_h,
    a_win_h,
    MAX_RANKS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SIGNAL_BASE: tl.constexpr,
):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    a_win = tgin.window(a_win_h, ptr=a_arena)
    rank = tgin.rank(dev)
    nranks = tgin.num_ranks(dev)
    token = tgin.token()
    a_bytes = M * K * 2

    rows = tl.arange(0, M)
    cols_k = tl.arange(0, K)
    cols_n = tl.arange(0, N)

    a_vals = tl.load(a_local + rows[:, None] * K + cols_k[None, :])
    local_a_off = rank * M * K
    tl.store(a_arena + local_a_off + rows[:, None] * K + cols_k[None, :], a_vals)
    tl.debug_barrier()

    local_byte_off = local_a_off * 2
    for peer in tl.static_range(0, MAX_RANKS):
        if peer < nranks and peer != rank:
            token = tgin.put_signal_window(
                gin,
                peer,
                a_win,
                local_byte_off,
                a_win,
                local_byte_off,
                a_bytes,
                signal_id=SIGNAL_BASE,
                signal_value=a_bytes,
                token=token,
            )
    token = tgin.flush(gin, token=token)

    for peer in tl.static_range(0, MAX_RANKS):
        if peer < nranks and peer != rank:
            token = tgin.wait_signal(
                gin,
                peer,
                signal_id=SIGNAL_BASE,
                least_value=a_bytes,
                token=token,
            )

    for src in tl.static_range(0, MAX_RANKS):
        if src < nranks:
            src_a_off = src * M * K
            acc = tl.zeros((M, N), dtype=tl.float32)
            for kk in tl.static_range(0, K):
                a_col = tl.load(a_arena + src_a_off + rows * K + kk).to(tl.float32)
                b_row = tl.load(b + kk * N + cols_n).to(tl.float32)
                acc += a_col[:, None] * b_row[None, :]
            tl.store(c + (src * M + rows)[:, None] * N + cols_n[None, :], acc)


@triton.jit
def gemm_allreduce_kernel(
    a_local,
    b_local,
    c_out,
    c_arena,
    comm_h,
    c_win_h,
    MAX_RANKS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SIGNAL_BASE: tl.constexpr,
):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    c_win = tgin.window(c_win_h, ptr=c_arena)
    rank = tgin.rank(dev)
    nranks = tgin.num_ranks(dev)
    token = tgin.token()
    c_bytes = M * N * 4

    rows = tl.arange(0, M)
    cols_k = tl.arange(0, K)
    cols_n = tl.arange(0, N)
    c_vals = tl.zeros((M, N), dtype=tl.float32)
    for kk in tl.static_range(0, K):
        a_col = tl.load(a_local + rows * K + kk).to(tl.float32)
        b_row = tl.load(b_local + kk * N + cols_n).to(tl.float32)
        c_vals += a_col[:, None] * b_row[None, :]

    local_c_off = rank * M * N
    tl.store(c_arena + local_c_off + rows[:, None] * N + cols_n[None, :], c_vals)
    tl.debug_barrier()

    local_byte_off = local_c_off * 4
    for peer in tl.static_range(0, MAX_RANKS):
        if peer < nranks and peer != rank:
            token = tgin.put_signal_window(
                gin,
                peer,
                c_win,
                local_byte_off,
                c_win,
                local_byte_off,
                c_bytes,
                signal_id=SIGNAL_BASE,
                signal_value=c_bytes,
                token=token,
            )
    token = tgin.flush(gin, token=token)

    for peer in tl.static_range(0, MAX_RANKS):
        if peer < nranks and peer != rank:
            token = tgin.wait_signal(
                gin,
                peer,
                signal_id=SIGNAL_BASE,
                least_value=c_bytes,
                token=token,
            )

    acc = tl.zeros((M, N), dtype=tl.float32)
    for src in tl.static_range(0, MAX_RANKS):
        if src < nranks:
            src_c_off = src * M * N
            acc += tl.load(c_arena + src_c_off + rows[:, None] * N + cols_n[None, :])
    tl.store(c_out + rows[:, None] * N + cols_n[None, :], acc)


@triton.jit
def gemm_reduce_scatter_kernel(
    a_local,
    b_local,
    c_out,
    c_arena,
    comm_h,
    c_win_h,
    MAX_RANKS: tl.constexpr,
    M_PER_RANK: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SIGNAL_BASE: tl.constexpr,
):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    c_win = tgin.window(c_win_h, ptr=c_arena)
    rank = tgin.rank(dev)
    nranks = tgin.num_ranks(dev)
    token = tgin.token()
    seg_elems = M_PER_RANK * N
    seg_bytes = seg_elems * 4

    rows = tl.arange(0, M_PER_RANK)
    cols_n = tl.arange(0, N)

    for dst in tl.static_range(0, MAX_RANKS):
        if dst < nranks:
            a_row_base = dst * M_PER_RANK
            c_vals = tl.zeros((M_PER_RANK, N), dtype=tl.float32)
            for kk in tl.static_range(0, K):
                a_col = tl.load(a_local + (a_row_base + rows) * K + kk).to(tl.float32)
                b_row = tl.load(b_local + kk * N + cols_n).to(tl.float32)
                c_vals += a_col[:, None] * b_row[None, :]
            scratch_off = nranks * seg_elems + dst * seg_elems
            tl.store(c_arena + scratch_off + rows[:, None] * N + cols_n[None, :], c_vals)

            if dst == rank:
                local_slot = rank * seg_elems
                tl.store(c_arena + local_slot + rows[:, None] * N + cols_n[None, :], c_vals)
            else:
                tl.debug_barrier()
                src_byte_off = scratch_off * 4
                dst_byte_off = rank * seg_elems * 4
                token = tgin.put_signal_window(
                    gin,
                    dst,
                    c_win,
                    dst_byte_off,
                    c_win,
                    src_byte_off,
                    seg_bytes,
                    signal_id=SIGNAL_BASE,
                    signal_value=seg_bytes,
                    token=token,
                )
    token = tgin.flush(gin, token=token)

    for src in tl.static_range(0, MAX_RANKS):
        if src < nranks and src != rank:
            token = tgin.wait_signal(
                gin,
                src,
                signal_id=SIGNAL_BASE,
                least_value=seg_bytes,
                token=token,
            )

    acc = tl.zeros((M_PER_RANK, N), dtype=tl.float32)
    for src in tl.static_range(0, MAX_RANKS):
        if src < nranks:
            acc += tl.load(c_arena + src * seg_elems + rows[:, None] * N + cols_n[None, :])
    tl.store(c_out + rows[:, None] * N + cols_n[None, :], acc)


def bind_tilexr(tilexr_lib):
    lib = ctypes.CDLL(tilexr_lib, mode=ctypes.RTLD_GLOBAL)
    lib.TileXRCommInitRankLocal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    lib.TileXRCommInitRankLocal.restype = ctypes.c_int
    lib.TileXRCommDestroy.argtypes = [ctypes.c_void_p]
    lib.TileXRCommDestroy.restype = ctypes.c_int
    return lib


def wait_files(prefix, rank_size, rank, phase, timeout_s=180):
    gin_runtime.rendezvous_barrier(prefix, rank_size, rank, phase, timeout_s)


def check_close(name, got, expected, atol=2e-2, rtol=1e-3):
    diff = (got - expected).abs()
    maxerr = float(diff.max()) if diff.numel() else 0.0
    limit = atol + rtol * expected.abs()
    bad_mask = diff > limit
    if bool(bad_mask.any()):
        bad = int(diff.argmax())
        raise AssertionError(
            f"{name} failed maxerr={maxerr} atol={atol} rtol={rtol} bad={bad} "
            f"got={float(got.flatten()[bad])} expected={float(expected.flatten()[bad])} "
            f"limit={float(limit.flatten()[bad])}"
        )
    print(f"{name}: PASS maxerr={maxerr} atol={atol} rtol={rtol}")


def reference_gemm(a, b):
    a = a.float()
    b = b.float()
    out = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    for kk in range(a.shape[1]):
        out += a[:, kk : kk + 1] * b[kk : kk + 1, :]
    return out


def make_a(rank, rows, k, device):
    return (
        torch.arange(rows * k, dtype=torch.float16, device=device).reshape(rows, k) * 0.01
        + rank
    )


def make_b(rank, k, n, device):
    return (
        torch.arange(k * n, dtype=torch.float16, device=device).reshape(k, n) * 0.005
        + 1.0
        + rank * 0.1
    )


def reset_all(comm_h, rank_size, rank, tag):
    grid = (1,)
    for signal_base in (SIGNAL_BASE_AG, SIGNAL_BASE_AR, SIGNAL_BASE_RS):
        reset_gin_signals_kernel[grid](comm_h, rank_size, signal_base, 1)
    torch.npu.synchronize()
    wait_files(f"/tmp/triton_gin_gemm_{tag}", rank_size, rank, "reset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rank-size", type=int, default=2)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tilexr-lib", default=TILEXR_LIB)
    parser.add_argument("--case", choices=("allgather_gemm", "gemm_allreduce", "gemm_reduce_scatter", "all"), default="all")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()

    if args.rank_size < 2:
        raise ValueError("rank-size must be at least 2")
    if args.m > 32 or args.n > 32 or args.k > 32:
        raise ValueError("this smoke runner intentionally keeps one small tile per case")

    torch.npu.set_device(args.device)
    device = "npu"
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
        prefix = f"/tmp/triton_gin_gemm_{args.tag}"
        reset_all(comm_h, args.rank_size, args.rank, args.tag)

        grid = (1,)

        if args.case in ("allgather_gemm", "all"):
            a = make_a(args.rank, args.m, args.k, device)
            b = make_b(0, args.k, args.n, device)
            c = torch.full((args.rank_size * args.m, args.n), -777.0, dtype=torch.float32, device=device)
            a_arena = torch.full((args.rank_size * args.m * args.k,), -777.0, dtype=torch.float16, device=device)
            a_win_h = gin.window_handle(a_arena, rendezvous_id=f"{args.tag}_ag_a")
            allgather_gemm_kernel[grid](
                a,
                b,
                c,
                a_arena,
                comm_h,
                a_win_h,
                args.rank_size,
                M=args.m,
                N=args.n,
                K=args.k,
                SIGNAL_BASE=SIGNAL_BASE_AG,
            )
            torch.npu.synchronize()
            wait_files(prefix, args.rank_size, args.rank, "allgather_gemm")
            expected = torch.cat(
                [reference_gemm(make_a(src, args.m, args.k, "cpu"), make_b(0, args.k, args.n, "cpu"))
                 for src in range(args.rank_size)],
                dim=0,
            )
            check_close("gin_allgather_gemm", c.cpu(), expected)

        if args.case in ("gemm_allreduce", "all"):
            a = make_a(args.rank, args.m, args.k, device)
            b = make_b(args.rank, args.k, args.n, device)
            c_out = torch.full((args.m, args.n), -777.0, dtype=torch.float32, device=device)
            c_arena = torch.full((args.rank_size * args.m * args.n,), -777.0, dtype=torch.float32, device=device)
            c_win_h = gin.window_handle(c_arena, rendezvous_id=f"{args.tag}_ar_c")
            gemm_allreduce_kernel[grid](
                a,
                b,
                c_out,
                c_arena,
                comm_h,
                c_win_h,
                args.rank_size,
                M=args.m,
                N=args.n,
                K=args.k,
                SIGNAL_BASE=SIGNAL_BASE_AR,
            )
            torch.npu.synchronize()
            wait_files(prefix, args.rank_size, args.rank, "gemm_allreduce")
            expected = sum(
                reference_gemm(make_a(src, args.m, args.k, "cpu"), make_b(src, args.k, args.n, "cpu"))
                for src in range(args.rank_size)
            )
            check_close("gin_gemm_allreduce", c_out.cpu(), expected)

        if args.case in ("gemm_reduce_scatter", "all"):
            total_m = args.rank_size * args.m
            a = make_a(args.rank, total_m, args.k, device)
            b = make_b(args.rank, args.k, args.n, device)
            c_out = torch.full((args.m, args.n), -777.0, dtype=torch.float32, device=device)
            seg_elems = args.m * args.n
            c_arena = torch.full((2 * args.rank_size * seg_elems,), -777.0, dtype=torch.float32, device=device)
            c_win_h = gin.window_handle(c_arena, rendezvous_id=f"{args.tag}_rs_c")
            gemm_reduce_scatter_kernel[grid](
                a,
                b,
                c_out,
                c_arena,
                comm_h,
                c_win_h,
                args.rank_size,
                M_PER_RANK=args.m,
                N=args.n,
                K=args.k,
                SIGNAL_BASE=SIGNAL_BASE_RS,
            )
            torch.npu.synchronize()
            wait_files(prefix, args.rank_size, args.rank, "gemm_reduce_scatter")
            expected = sum(
                reference_gemm(make_a(src, total_m, args.k, "cpu"), make_b(src, args.k, args.n, "cpu"))
                for src in range(args.rank_size)
            )
            expected = expected[args.rank * args.m : (args.rank + 1) * args.m, :]
            check_close("gin_gemm_reduce_scatter", c_out.cpu(), expected)

        wait_files(prefix, args.rank_size, args.rank, "checked")
        print(f"GIN GEMM fusion patterns: PASS rank={args.rank} case={args.case}")
    finally:
        if gin is not None:
            gin.close()
        if tile_comm.value:
            tile.TileXRCommDestroy(tile_comm)


if __name__ == "__main__":
    main()
