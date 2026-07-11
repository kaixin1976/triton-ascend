import argparse
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


RUNTIME_LIB = os.getenv("TRITON_ASCEND_GIN_RUNTIME_LIB")
HCCL_LIB = os.getenv(
    "TRITON_ASCEND_HCCL_LIB",
    "/home/kaixin/Ascend/cann-9.1.0-beta.1/aarch64-linux/lib64/libhcomm.so",
)
SIGNAL_SLOTS = 64


@triton.jit
def reset_hccl_gin_signals_kernel(comm_h, MAX_RANKS: tl.constexpr, SIGNALS: tl.constexpr):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_HCCL_PEER_MEM)
    nranks = tgin.num_ranks(dev)
    token = tgin.token()
    for signal_id in tl.static_range(0, SIGNALS):
        for peer in tl.static_range(0, MAX_RANKS):
            if peer < nranks:
                token = tgin.reset_signal(gin, peer, signal_id=signal_id, token=token)
    token = tgin.flush(gin, token=token)


@triton.jit
def hccl_fused_all_gather_scale_kernel(
    x,
    y,
    comm_h,
    win_h,
    n_elements: tl.constexpr,
    MAX_RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    dev = tgin.dev_comm(comm_h)
    gin = tgin.gin(dev, backend_mask=tgin.GIN_BACKEND_HCCL_PEER_MEM)
    win = tgin.window(win_h, ptr=y)

    rank = tgin.rank(dev)
    nranks = tgin.num_ranks(dev)
    token = tgin.token()
    tile_bytes = BLOCK * 4

    for pid in tl.static_range(0, BLOCKS):
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        local_tile = y + rank * n_elements + pid * BLOCK
        values = tl.load(x + offs, mask=mask, other=0.0) * 0.5
        tl.store(local_tile + tl.arange(0, BLOCK), values, mask=mask)

        dst_byte_off = (rank * n_elements + pid * BLOCK) * 4
        for peer in tl.static_range(0, MAX_RANKS):
            if peer < nranks and peer != rank:
                token = tgin.put_signal_window(
                    gin,
                    peer,
                    win,
                    dst_byte_off,
                    win,
                    dst_byte_off,
                    tile_bytes,
                    signal_id=pid,
                    signal_value=tile_bytes,
                    token=token,
                )

    token = tgin.flush(gin, token=token)

    for pid in tl.static_range(0, BLOCKS):
        for peer in tl.static_range(0, MAX_RANKS):
            if peer < nranks and peer != rank:
                token = tgin.wait_signal(
                    gin,
                    peer,
                    signal_id=pid,
                    least_value=tile_bytes,
                    token=token,
                )


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
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--block", type=int, default=32)
    args = parser.parse_args()

    if args.rank_size != 2:
        raise ValueError("this HCCL smoke runner currently validates the two-rank path")
    if args.n % args.block != 0:
        raise ValueError("n must be a multiple of block for the current byte-copy smoke")

    torch.npu.set_device(args.device)
    dist.init_process_group(
        backend="hccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=args.rank_size,
        timeout=datetime.timedelta(seconds=180),
    )

    gin = None
    try:
        # Force the process group to materialize its HCCL communicator before
        # resolving the comm name to HcclComm*.
        probe = torch.tensor([float(args.rank + 1)], dtype=torch.float32, device="npu")
        dist.all_reduce(probe, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()
        if float(probe.cpu().item()) != 3.0:
            raise AssertionError(f"HCCL all_reduce probe failed: {float(probe.cpu().item())}")

        gin = gin_runtime.create_from_torch_hccl_process_group(
            rank=args.rank,
            runtime_library=RUNTIME_LIB,
            hccl_library=HCCL_LIB,
            signal_slots=SIGNAL_SLOTS,
        )
        comm_h = gin.dev_comm
        x = torch.arange(args.n, dtype=torch.float32, device="npu") + args.rank * 100.0
        y = torch.full((args.rank_size * args.n,), -777.0, dtype=torch.float32, device="npu")
        signal_window = torch.zeros((args.rank_size * SIGNAL_SLOTS,), dtype=torch.int64, device="npu")
        win_h = gin.window_handle(y, rendezvous_id=args.tag)
        sig_h = gin.signal_window_handle(signal_window, rendezvous_id=args.tag + "_signal")

        blocks = triton.cdiv(args.n, args.block)
        grid = (1,)
        prefix = f"/tmp/triton_gin_hccl_{args.tag}"

        reset_hccl_gin_signals_kernel[grid](comm_h, args.rank_size, blocks)
        torch.npu.synchronize()
        wait_files(prefix, args.rank_size, args.rank, "reset")

        hccl_fused_all_gather_scale_kernel[grid](
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

        expected = torch.cat(
            [
                (torch.arange(args.n, dtype=torch.float32) + source_rank * 100.0) * 0.5
                for source_rank in range(args.rank_size)
            ]
        )
        got = y.cpu()
        diff = (got - expected).abs()
        maxerr = float(diff.max())
        bad_idx = int(diff.argmax()) if diff.numel() else -1
        print(
            f"rank={args.rank} device={args.device} comm_h=0x{comm_h:x} "
            f"win_h={win_h} sig_h={sig_h} maxerr={maxerr}"
        )
        print("got_head=", got[: min(8, got.numel())].tolist())
        print("got_peer_head=", got[args.n: args.n + min(8, args.n)].tolist())
        if maxerr != 0.0:
            raise AssertionError(
                f"HCCL GIN fused all-gather failed maxerr={maxerr} bad_idx={bad_idx} "
                f"got={float(got[bad_idx])} expected={float(expected[bad_idx])}"
            )
        wait_files(prefix, args.rank_size, args.rank, "checked")
        print(f"HCCL GIN fused all-gather: PASS rank={args.rank}")
    finally:
        if gin is not None:
            gin.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
