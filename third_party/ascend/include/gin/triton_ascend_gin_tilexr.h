#ifndef TRITON_ASCEND_GIN_TILEXR_H_
#define TRITON_ASCEND_GIN_TILEXR_H_

#include "triton_ascend_gin_abi.h"

#if defined(__NPU_ARCH__) || defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__CCE__)
#include "kernel_operator.h"
#endif

#if defined(TRITON_ASCEND_GIN_ENABLE_TILEXR)
#include "tilexr_udma.h"
#endif

#if defined(__NPU_ARCH__) || defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__CCE_IS_AICORE__) || defined(__CCE__)
#define TRITON_ASCEND_GIN_DEVICE __aicore__
#define TRITON_ASCEND_GIN_GM __gm__
#else
#define TRITON_ASCEND_GIN_DEVICE
#define TRITON_ASCEND_GIN_GM
#endif

#ifndef TRITON_ASCEND_GIN_COPY_UB_OFFSET
#define TRITON_ASCEND_GIN_COPY_UB_OFFSET 8192
#endif

#ifndef TRITON_ASCEND_GIN_COPY_CHUNK_BYTES
#define TRITON_ASCEND_GIN_COPY_CHUNK_BYTES 1024
#endif

#ifdef __cplusplus
extern "C" {
#endif

TRITON_ASCEND_GIN_DEVICE inline const TRITON_ASCEND_GIN_GM TritonAscendGinDev *
__triton_ascend_gin_as_dev(uint64_t dev_comm) {
  return reinterpret_cast<const TRITON_ASCEND_GIN_GM TritonAscendGinDev *>(dev_comm);
}

TRITON_ASCEND_GIN_DEVICE inline uint64_t
__triton_ascend_gin_signal_slot(const TRITON_ASCEND_GIN_GM TritonAscendGinDev *comm,
                                 int32_t source_rank, int32_t signal_id) {
  if (comm == nullptr || source_rank < 0 || source_rank >= static_cast<int32_t>(comm->nranks) ||
      signal_id < 0) {
    return 0;
  }
  uint64_t slots = comm->signal_slots != 0 ? comm->signal_slots : TRITON_ASCEND_GIN_DEFAULT_SIGNAL_SLOTS;
  return static_cast<uint64_t>(source_rank) * slots + static_cast<uint64_t>(signal_id);
}

TRITON_ASCEND_GIN_DEVICE inline uint64_t
__triton_ascend_gin_signal_offset(const TRITON_ASCEND_GIN_GM TritonAscendGinDev *comm,
                                   int32_t source_rank, int32_t signal_id) {
  uint64_t stride = comm->signal_stride != 0 ? comm->signal_stride : sizeof(uint64_t);
  return comm->window_bytes + __triton_ascend_gin_signal_slot(comm, source_rank, signal_id) * stride;
}

TRITON_ASCEND_GIN_DEVICE inline bool
__triton_ascend_gin_use_peer_mem(const TRITON_ASCEND_GIN_GM TritonAscendGinDev *comm, uint32_t backend_mask) {
  if (comm == nullptr) {
    return false;
  }
  if ((backend_mask & TRITON_ASCEND_GIN_BACKEND_TILEXR_IPC_PEER_MEM) != 0 &&
      comm->backend_kind == TRITON_ASCEND_GIN_BACKEND_KIND_TILEXR_IPC_PEER_MEM) {
    return true;
  }
  return (backend_mask & TRITON_ASCEND_GIN_BACKEND_HCCL_PEER_MEM) != 0 &&
         comm->backend_kind == TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_PEER_MEM;
}

TRITON_ASCEND_GIN_DEVICE inline uint64_t
__triton_ascend_gin_peer_window_addr(const TRITON_ASCEND_GIN_GM TritonAscendGinDev *comm,
                                      int32_t peer, uint64_t offset) {
  if (comm == nullptr || peer < 0 || peer >= static_cast<int32_t>(comm->nranks)) {
    return 0;
  }
  uint64_t base = comm->peer_window_base[peer];
  return base == 0 ? 0 : base + offset;
}

TRITON_ASCEND_GIN_DEVICE inline uint64_t
__triton_ascend_gin_peer_signal_addr(const TRITON_ASCEND_GIN_GM TritonAscendGinDev *comm,
                                      int32_t target_peer, int32_t source_rank, int32_t signal_id) {
  if (comm == nullptr || target_peer < 0 || target_peer >= static_cast<int32_t>(comm->nranks)) {
    return 0;
  }
  uint64_t stride = comm->signal_stride != 0 ? comm->signal_stride : sizeof(uint64_t);
  uint64_t base = comm->peer_signal_base[target_peer];
  return base == 0 ? 0 : base + __triton_ascend_gin_signal_slot(comm, source_rank, signal_id) * stride;
}

TRITON_ASCEND_GIN_DEVICE inline void
__triton_ascend_gin_store_u64(uint64_t addr, uint64_t value) {
  if (addr == 0) {
    return;
  }
#if defined(__NPU_ARCH__) || defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__CCE_IS_AICORE__) || defined(__CCE__)
  st_dev(value, reinterpret_cast<TRITON_ASCEND_GIN_GM uint64_t *>(addr), 0);
  pipe_barrier(PIPE_ALL);
#else
  *reinterpret_cast<uint64_t *>(addr) = value;
#endif
}

#if defined(__NPU_ARCH__) || defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__CCE_IS_AICORE__) || defined(__CCE__)
TRITON_ASCEND_GIN_DEVICE inline void
__triton_ascend_gin_cp_gm_to_ub(uint64_t ub_addr, uint64_t gm_addr, uint32_t size) {
  AscendC::LocalTensor<uint8_t> ub_tensor;
  AscendC::TBuffAddr ub_tensor_addr{};
  ub_tensor_addr.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
  ub_tensor_addr.bufferAddr = ub_addr;
  ub_tensor.SetAddr(ub_tensor_addr);

  AscendC::GlobalTensor<uint8_t> gm_tensor;
  gm_tensor.SetGlobalBuffer(reinterpret_cast<TRITON_ASCEND_GIN_GM uint8_t *>(gm_addr));

  AscendC::DataCopyExtParams copy_params(1, size, 0, 0, 0);
  AscendC::DataCopyPadExtParams<uint8_t> pad_params;
  AscendC::DataCopyPad(ub_tensor, gm_tensor, copy_params, pad_params);
}

TRITON_ASCEND_GIN_DEVICE inline void
__triton_ascend_gin_cp_ub_to_gm(uint64_t gm_addr, uint64_t ub_addr, uint32_t size) {
  AscendC::LocalTensor<uint8_t> ub_tensor;
  AscendC::TBuffAddr ub_tensor_addr{};
  ub_tensor_addr.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
  ub_tensor_addr.bufferAddr = ub_addr;
  ub_tensor.SetAddr(ub_tensor_addr);

  AscendC::GlobalTensor<uint8_t> gm_tensor;
  gm_tensor.SetGlobalBuffer(reinterpret_cast<TRITON_ASCEND_GIN_GM uint8_t *>(gm_addr));

  AscendC::DataCopyExtParams copy_params(1, size, 0, 0, 0);
  AscendC::DataCopyPad(gm_tensor, ub_tensor, copy_params);
}
#endif

TRITON_ASCEND_GIN_DEVICE inline void
__triton_ascend_gin_copy_bytes(uint64_t dst_addr, uint64_t src_addr, uint64_t nbytes) {
  if (dst_addr == 0 || src_addr == 0 || nbytes == 0) {
    return;
  }
#if defined(__NPU_ARCH__) || defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__CCE_IS_AICORE__) || defined(__CCE__)
  // Current extern helper ABI has no UB scratch operand. Use a conservative
  // private UB slice for the prototype GM->UB->GM copy path.
  constexpr uint64_t kCopyUbOffset = TRITON_ASCEND_GIN_COPY_UB_OFFSET;
  constexpr uint64_t kCopyChunkBytes = TRITON_ASCEND_GIN_COPY_CHUNK_BYTES;
  uint64_t copied = 0;
  while (copied < nbytes) {
    uint64_t remaining = nbytes - copied;
    uint32_t chunk = static_cast<uint32_t>(remaining > kCopyChunkBytes ? kCopyChunkBytes : remaining);
    __triton_ascend_gin_cp_gm_to_ub(kCopyUbOffset, src_addr + copied, chunk);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    __triton_ascend_gin_cp_ub_to_gm(dst_addr + copied, kCopyUbOffset, chunk);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    copied += chunk;
  }
  AscendC::PipeBarrier<PIPE_ALL>();
#else
  auto dst = reinterpret_cast<uint8_t *>(dst_addr);
  auto src = reinterpret_cast<const uint8_t *>(src_addr);
  for (uint64_t i = 0; i < nbytes; ++i) {
    dst[i] = src[i];
  }
#endif
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_rank(uint64_t dev_comm, int32_t team) {
  (void)team;
  auto comm = __triton_ascend_gin_as_dev(dev_comm);
  return comm == nullptr ? -1 : static_cast<int32_t>(comm->rank);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_num_ranks(uint64_t dev_comm, int32_t team) {
  (void)team;
  auto comm = __triton_ascend_gin_as_dev(dev_comm);
  return comm == nullptr ? -1 : static_cast<int32_t>(comm->nranks);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_put(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_ptr, uint64_t nbytes, int32_t sem, int32_t token) {
  (void)context_id;
  (void)team;
  (void)dst_window;
  (void)sem;
  auto comm = __triton_ascend_gin_as_dev(dev_comm);
  if (comm == nullptr) {
    return token;
  }
  if (__triton_ascend_gin_use_peer_mem(comm, backend_mask)) {
    uint64_t dst_addr = __triton_ascend_gin_peer_window_addr(comm, peer, dst_offset);
    __triton_ascend_gin_copy_bytes(dst_addr, src_ptr, nbytes);
    return token + 1;
  }
#if defined(TRITON_ASCEND_GIN_ENABLE_TILEXR)
  if ((backend_mask & TRITON_ASCEND_GIN_BACKEND_TILEXR_UDMA) != 0 &&
      comm->backend_kind == TRITON_ASCEND_GIN_BACKEND_KIND_TILEXR_UDMA) {
    auto args = reinterpret_cast<const TRITON_ASCEND_GIN_GM TileXR::CommArgs *>(comm->tilexr_comm_args_dev);
    auto src = reinterpret_cast<const TRITON_ASCEND_GIN_GM uint8_t *>(src_ptr);
    TileXR::UDMAPutNbi(args, peer, src, dst_offset, static_cast<uint32_t>(nbytes));
  }
#else
  (void)backend_mask;
  (void)peer;
  (void)dst_offset;
  (void)src_ptr;
  (void)nbytes;
#endif
  return token + 1;
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_put_signal(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_ptr, uint64_t nbytes,
    int32_t signal_id, uint64_t signal_value, int32_t sem, int32_t token) {
  (void)context_id;
  (void)team;
  (void)dst_window;
  (void)sem;
  auto comm = __triton_ascend_gin_as_dev(dev_comm);
  if (comm == nullptr) {
    return token;
  }
  if (__triton_ascend_gin_use_peer_mem(comm, backend_mask)) {
    uint64_t dst_addr = __triton_ascend_gin_peer_window_addr(comm, peer, dst_offset);
    uint64_t signal_addr = __triton_ascend_gin_peer_signal_addr(
        comm, peer, static_cast<int32_t>(comm->rank), signal_id);
    __triton_ascend_gin_copy_bytes(dst_addr, src_ptr, nbytes);
    __triton_ascend_gin_store_u64(signal_addr, signal_value);
    return token + 1;
  }
#if defined(TRITON_ASCEND_GIN_ENABLE_TILEXR)
  if ((backend_mask & TRITON_ASCEND_GIN_BACKEND_TILEXR_UDMA) != 0 &&
      comm->backend_kind == TRITON_ASCEND_GIN_BACKEND_KIND_TILEXR_UDMA) {
    auto args = reinterpret_cast<const TRITON_ASCEND_GIN_GM TileXR::CommArgs *>(comm->tilexr_comm_args_dev);
    auto src = reinterpret_cast<const TRITON_ASCEND_GIN_GM uint8_t *>(src_ptr);
    uint64_t signal_offset = __triton_ascend_gin_signal_offset(
        comm, static_cast<int32_t>(comm->rank), signal_id);
    TileXR::UDMAPutSignalNbi(args, peer, src, dst_offset, static_cast<uint32_t>(nbytes),
                             signal_offset, signal_value);
  }
#else
  (void)backend_mask;
  (void)peer;
  (void)dst_offset;
  (void)src_ptr;
  (void)nbytes;
  (void)signal_id;
  (void)signal_value;
#endif
  return token + 1;
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_put_window(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_window, uint64_t src_offset,
    uint64_t nbytes, int32_t sem, int32_t token) {
  (void)src_window;
  auto comm = __triton_ascend_gin_as_dev(dev_comm);
  if (comm == nullptr) {
    return token;
  }
  uint64_t src_ptr = __triton_ascend_gin_peer_window_addr(
      comm, static_cast<int32_t>(comm->rank), src_offset);
  return __triton_ascend_gin_put(dev_comm, backend_mask, context_id, team, peer, dst_window,
                                  dst_offset, src_ptr, nbytes, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_put_signal_window(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_window, uint64_t src_offset,
    uint64_t nbytes, int32_t signal_id, uint64_t signal_value, int32_t sem, int32_t token) {
  (void)src_window;
  auto comm = __triton_ascend_gin_as_dev(dev_comm);
  if (comm == nullptr) {
    return token;
  }
  uint64_t src_ptr = __triton_ascend_gin_peer_window_addr(
      comm, static_cast<int32_t>(comm->rank), src_offset);
  return __triton_ascend_gin_put_signal(dev_comm, backend_mask, context_id, team, peer,
                                         dst_window, dst_offset, src_ptr, nbytes, signal_id,
                                         signal_value, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_get(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_ptr, uint64_t src_window, uint64_t src_offset, uint64_t nbytes, int32_t sem, int32_t token) {
  (void)context_id;
  (void)team;
  (void)src_window;
  (void)sem;
  auto comm = __triton_ascend_gin_as_dev(dev_comm);
  if (comm == nullptr) {
    return token;
  }
  if (__triton_ascend_gin_use_peer_mem(comm, backend_mask)) {
    uint64_t src_addr = __triton_ascend_gin_peer_window_addr(comm, peer, src_offset);
    __triton_ascend_gin_copy_bytes(dst_ptr, src_addr, nbytes);
    return token + 1;
  }
#if defined(TRITON_ASCEND_GIN_ENABLE_TILEXR)
  if ((backend_mask & TRITON_ASCEND_GIN_BACKEND_TILEXR_UDMA) != 0 &&
      comm->backend_kind == TRITON_ASCEND_GIN_BACKEND_KIND_TILEXR_UDMA) {
    auto args = reinterpret_cast<const TRITON_ASCEND_GIN_GM TileXR::CommArgs *>(comm->tilexr_comm_args_dev);
    auto dst = reinterpret_cast<TRITON_ASCEND_GIN_GM uint8_t *>(dst_ptr);
    TileXR::UDMAGetNbi(args, peer, dst, src_offset, static_cast<uint32_t>(nbytes));
  }
#else
  (void)backend_mask;
  (void)peer;
  (void)dst_ptr;
  (void)src_offset;
  (void)nbytes;
#endif
  return token + 1;
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_signal(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    int32_t signal_id, uint64_t signal_value, int32_t sem, int32_t token) {
  (void)context_id;
  (void)team;
  (void)sem;
  auto comm = __triton_ascend_gin_as_dev(dev_comm);
  if (comm == nullptr) {
    return token;
  }
  if (__triton_ascend_gin_use_peer_mem(comm, backend_mask)) {
    uint64_t signal_addr = __triton_ascend_gin_peer_signal_addr(
        comm, peer, static_cast<int32_t>(comm->rank), signal_id);
    __triton_ascend_gin_store_u64(signal_addr, signal_value);
    return token + 1;
  }
#if defined(TRITON_ASCEND_GIN_ENABLE_TILEXR)
  if ((backend_mask & TRITON_ASCEND_GIN_BACKEND_TILEXR_UDMA) != 0 &&
      comm->backend_kind == TRITON_ASCEND_GIN_BACKEND_KIND_TILEXR_UDMA) {
    auto args = reinterpret_cast<const TRITON_ASCEND_GIN_GM TileXR::CommArgs *>(comm->tilexr_comm_args_dev);
    uint64_t signal_offset = __triton_ascend_gin_signal_offset(
        comm, static_cast<int32_t>(comm->rank), signal_id);
    TileXR::UDMAPutSignalNbi(args, peer, reinterpret_cast<const TRITON_ASCEND_GIN_GM uint8_t *>(0),
                             signal_offset, 0, signal_offset, signal_value);
  }
#else
  (void)backend_mask;
  (void)peer;
  (void)signal_id;
  (void)signal_value;
#endif
  return token + 1;
}

TRITON_ASCEND_GIN_DEVICE uint64_t __triton_ascend_gin_read_signal(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t peer,
    int32_t signal_id, int32_t bits, int32_t sem) {
  (void)backend_mask;
  (void)context_id;
  (void)bits;
  (void)sem;
  auto comm = __triton_ascend_gin_as_dev(dev_comm);
  if (comm == nullptr || peer < 0 || peer >= static_cast<int32_t>(comm->nranks)) {
    return 0;
  }
  uint64_t addr = __triton_ascend_gin_peer_signal_addr(
      comm, static_cast<int32_t>(comm->rank), peer, signal_id);
  if (addr == 0) {
    return 0;
  }
#if defined(__NPU_ARCH__) || defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__CCE_IS_AICORE__) || defined(__CCE__)
  return ld_dev(reinterpret_cast<TRITON_ASCEND_GIN_GM uint64_t *>(addr), 0);
#else
  return *reinterpret_cast<uint64_t *>(addr);
#endif
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_wait_signal(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t peer,
    int32_t signal_id, uint64_t least_value, int32_t bits, int32_t sem, int32_t token) {
  while (__triton_ascend_gin_read_signal(dev_comm, backend_mask, context_id, peer, signal_id, bits, sem) <
         least_value) {
#if defined(__NPU_ARCH__) || defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__CCE_IS_AICORE__) || defined(__CCE__)
    pipe_barrier(PIPE_ALL);
#endif
  }
  return token + 1;
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_reset_signal(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t peer,
    int32_t signal_id, int32_t token) {
  (void)backend_mask;
  (void)context_id;
  auto comm = __triton_ascend_gin_as_dev(dev_comm);
  if (comm == nullptr || peer < 0 || peer >= static_cast<int32_t>(comm->nranks)) {
    return token;
  }
  uint64_t addr = __triton_ascend_gin_peer_signal_addr(
      comm, static_cast<int32_t>(comm->rank), peer, signal_id);
  if (addr == 0) {
    return token;
  }
#if defined(__NPU_ARCH__) || defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__CCE_IS_AICORE__) || defined(__CCE__)
  st_dev(static_cast<uint64_t>(0), reinterpret_cast<TRITON_ASCEND_GIN_GM uint64_t *>(addr), 0);
#else
  *reinterpret_cast<uint64_t *>(addr) = 0;
#endif
  return token + 1;
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_flush(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer, int32_t token) {
  (void)context_id;
  (void)team;
  auto comm = __triton_ascend_gin_as_dev(dev_comm);
  if (comm == nullptr) {
    return token;
  }
  if (__triton_ascend_gin_use_peer_mem(comm, backend_mask)) {
#if defined(__NPU_ARCH__) || defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__CCE_IS_AICORE__) || defined(__CCE__)
    pipe_barrier(PIPE_ALL);
#endif
    return token + 1;
  }
#if defined(TRITON_ASCEND_GIN_ENABLE_TILEXR)
  if ((backend_mask & TRITON_ASCEND_GIN_BACKEND_TILEXR_UDMA) != 0 &&
      comm->backend_kind == TRITON_ASCEND_GIN_BACKEND_KIND_TILEXR_UDMA) {
    auto args = reinterpret_cast<const TRITON_ASCEND_GIN_GM TileXR::CommArgs *>(comm->tilexr_comm_args_dev);
    if (peer >= 0) {
      TileXR::UDMAQuiet(args, peer);
    } else {
      for (uint32_t rank = 0; rank < comm->nranks; ++rank) {
        if (rank != comm->rank) {
          TileXR::UDMAQuiet(args, static_cast<int>(rank));
        }
      }
    }
  }
#else
  (void)backend_mask;
  (void)peer;
#endif
  return token + 1;
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_barrier(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t token) {
  (void)dev_comm;
  (void)backend_mask;
  (void)context_id;
  (void)team;
  return token + 1;
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_rank_op(
    uint64_t dev_comm, int32_t team) {
  return __triton_ascend_gin_rank(dev_comm, team);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_num_ranks_op(
    uint64_t dev_comm, int32_t team) {
  return __triton_ascend_gin_num_ranks(dev_comm, team);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_put_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_ptr, uint64_t nbytes, int32_t sem,
    int32_t token) {
  return __triton_ascend_gin_put(dev_comm, backend_mask, context_id, team, peer, dst_window,
                                  dst_offset, src_ptr, nbytes, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_put_signal_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_ptr, uint64_t nbytes,
    int32_t signal_id, uint64_t signal_value, int32_t sem, int32_t token) {
  return __triton_ascend_gin_put_signal(dev_comm, backend_mask, context_id, team, peer,
                                         dst_window, dst_offset, src_ptr, nbytes, signal_id,
                                         signal_value, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_put_window_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_window, uint64_t src_offset,
    uint64_t nbytes, int32_t sem, int32_t token) {
  return __triton_ascend_gin_put_window(dev_comm, backend_mask, context_id, team, peer,
                                         dst_window, dst_offset, src_window, src_offset,
                                         nbytes, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_put_signal_window_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_window, uint64_t src_offset,
    uint64_t nbytes, int32_t signal_id, uint64_t signal_value, int32_t sem, int32_t token) {
  return __triton_ascend_gin_put_signal_window(dev_comm, backend_mask, context_id, team, peer,
                                                dst_window, dst_offset, src_window, src_offset,
                                                nbytes, signal_id, signal_value, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_get_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_ptr, uint64_t src_window, uint64_t src_offset, uint64_t nbytes, int32_t sem,
    int32_t token) {
  return __triton_ascend_gin_get(dev_comm, backend_mask, context_id, team, peer, dst_ptr,
                                  src_window, src_offset, nbytes, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_signal_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    int32_t signal_id, uint64_t signal_value, int32_t sem, int32_t token) {
  return __triton_ascend_gin_signal(dev_comm, backend_mask, context_id, team, peer, signal_id,
                                     signal_value, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_wait_signal_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t peer,
    int32_t signal_id, uint64_t least_value, int32_t bits, int32_t sem, int32_t token) {
  return __triton_ascend_gin_wait_signal(dev_comm, backend_mask, context_id, peer, signal_id,
                                          least_value, bits, sem, token);
}

TRITON_ASCEND_GIN_DEVICE uint64_t __triton_ascend_gin_read_signal_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t peer,
    int32_t signal_id, int32_t bits, int32_t sem) {
  return __triton_ascend_gin_read_signal(dev_comm, backend_mask, context_id, peer, signal_id,
                                          bits, sem);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_reset_signal_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t peer,
    int32_t signal_id, int32_t token) {
  return __triton_ascend_gin_reset_signal(dev_comm, backend_mask, context_id, peer, signal_id,
                                           token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_flush_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    int32_t token) {
  return __triton_ascend_gin_flush(dev_comm, backend_mask, context_id, team, peer, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_barrier_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t token) {
  return __triton_ascend_gin_barrier(dev_comm, backend_mask, context_id, team, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __hmf_triton_ascend_gin_rank_op(
    uint64_t dev_comm, int32_t team) {
  return __triton_ascend_gin_rank_op(dev_comm, team);
}

TRITON_ASCEND_GIN_DEVICE int32_t __hmf_triton_ascend_gin_num_ranks_op(
    uint64_t dev_comm, int32_t team) {
  return __triton_ascend_gin_num_ranks_op(dev_comm, team);
}

TRITON_ASCEND_GIN_DEVICE int32_t __hmf_triton_ascend_gin_put_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_ptr, uint64_t nbytes, int32_t sem,
    int32_t token) {
  return __triton_ascend_gin_put_op(dev_comm, backend_mask, context_id, team, peer, dst_window,
                                     dst_offset, src_ptr, nbytes, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __hmf_triton_ascend_gin_put_signal_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_ptr, uint64_t nbytes,
    int32_t signal_id, uint64_t signal_value, int32_t sem, int32_t token) {
  return __triton_ascend_gin_put_signal_op(dev_comm, backend_mask, context_id, team, peer,
                                            dst_window, dst_offset, src_ptr, nbytes,
                                            signal_id, signal_value, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __hmf_triton_ascend_gin_put_window_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_window, uint64_t src_offset,
    uint64_t nbytes, int32_t sem, int32_t token) {
  return __triton_ascend_gin_put_window_op(dev_comm, backend_mask, context_id, team, peer,
                                            dst_window, dst_offset, src_window, src_offset,
                                            nbytes, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __hmf_triton_ascend_gin_put_signal_window_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_window, uint64_t src_offset,
    uint64_t nbytes, int32_t signal_id, uint64_t signal_value, int32_t sem, int32_t token) {
  return __triton_ascend_gin_put_signal_window_op(dev_comm, backend_mask, context_id, team, peer,
                                                   dst_window, dst_offset, src_window, src_offset,
                                                   nbytes, signal_id, signal_value, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __hmf_triton_ascend_gin_get_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_ptr, uint64_t src_window, uint64_t src_offset, uint64_t nbytes, int32_t sem,
    int32_t token) {
  return __triton_ascend_gin_get_op(dev_comm, backend_mask, context_id, team, peer, dst_ptr,
                                     src_window, src_offset, nbytes, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __hmf_triton_ascend_gin_signal_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    int32_t signal_id, uint64_t signal_value, int32_t sem, int32_t token) {
  return __triton_ascend_gin_signal_op(dev_comm, backend_mask, context_id, team, peer,
                                        signal_id, signal_value, sem, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __hmf_triton_ascend_gin_wait_signal_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t peer,
    int32_t signal_id, uint64_t least_value, int32_t bits, int32_t sem, int32_t token) {
  return __triton_ascend_gin_wait_signal_op(dev_comm, backend_mask, context_id, peer,
                                             signal_id, least_value, bits, sem, token);
}

TRITON_ASCEND_GIN_DEVICE uint64_t __hmf_triton_ascend_gin_read_signal_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t peer,
    int32_t signal_id, int32_t bits, int32_t sem) {
  return __triton_ascend_gin_read_signal_op(dev_comm, backend_mask, context_id, peer,
                                             signal_id, bits, sem);
}

TRITON_ASCEND_GIN_DEVICE int32_t __hmf_triton_ascend_gin_reset_signal_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t peer,
    int32_t signal_id, int32_t token) {
  return __triton_ascend_gin_reset_signal_op(dev_comm, backend_mask, context_id, peer,
                                              signal_id, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __hmf_triton_ascend_gin_flush_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    int32_t token) {
  return __triton_ascend_gin_flush_op(dev_comm, backend_mask, context_id, team, peer, token);
}

TRITON_ASCEND_GIN_DEVICE int32_t __hmf_triton_ascend_gin_barrier_op(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t token) {
  return __triton_ascend_gin_barrier_op(dev_comm, backend_mask, context_id, team, token);
}

#ifdef __cplusplus
}
#endif

#undef TRITON_ASCEND_GIN_DEVICE
#undef TRITON_ASCEND_GIN_GM

#endif  // TRITON_ASCEND_GIN_TILEXR_H_
