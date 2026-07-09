import triton
import triton.language as tl
import triton.language.extra.cann.gin as tgin


@triton.jit
def fused_all_gather_scale_kernel(x, y, comm_h, win_h, n_elements: tl.constexpr,
                                  MAX_RANKS: tl.constexpr, BLOCK: tl.constexpr,
                                  BLOCKS: tl.constexpr):
    comm = tgin.dev_comm(comm_h)
    gin = tgin.gin(comm, backend_mask=tgin.GIN_BACKEND_TILEXR_IPC_PEER_MEM)
    win = tgin.window(win_h, ptr=y)

    rank = tgin.rank(comm)
    nranks = tgin.num_ranks(comm)

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
                token = tgin.wait_signal(gin, peer, signal_id=pid, least_value=tile_bytes, token=token)


__all__ = ["fused_all_gather_scale_kernel"]

