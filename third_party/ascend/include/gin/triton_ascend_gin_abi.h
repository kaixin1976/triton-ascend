#ifndef TRITON_ASCEND_GIN_ABI_H_
#define TRITON_ASCEND_GIN_ABI_H_

#include <stdint.h>

#ifndef TRITON_ASCEND_GIN_MAX_RANKS
#define TRITON_ASCEND_GIN_MAX_RANKS 128
#endif

#ifndef TRITON_ASCEND_GIN_DEFAULT_SIGNAL_SLOTS
#define TRITON_ASCEND_GIN_DEFAULT_SIGNAL_SLOTS 2048
#endif

#if defined(__NPU_ARCH__) || defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__CCE_IS_AICORE__) || defined(__CCE__)
#define TRITON_ASCEND_GIN_DEVICE __aicore__
#define TRITON_ASCEND_GIN_GM __gm__
#else
#define TRITON_ASCEND_GIN_DEVICE
#define TRITON_ASCEND_GIN_GM
#endif

enum TritonAscendGinBackendMask {
  TRITON_ASCEND_GIN_BACKEND_HCCL_PEER_MEM = 1u << 0,
  TRITON_ASCEND_GIN_BACKEND_TILEXR_UDMA = 1u << 1,
  TRITON_ASCEND_GIN_BACKEND_HCOMM_TASK = 1u << 2,
  TRITON_ASCEND_GIN_BACKEND_TILEXR_IPC_PEER_MEM = 1u << 3,
  TRITON_ASCEND_GIN_BACKEND_UBS_CORE_UBC = 1u << 4,
  TRITON_ASCEND_GIN_BACKEND_HCCL_CHANNEL = 1u << 5,
  TRITON_ASCEND_GIN_BACKEND_UBS_COMM_URMA =
      TRITON_ASCEND_GIN_BACKEND_UBS_CORE_UBC,
  TRITON_ASCEND_GIN_BACKEND_PEER_MEM_COPY =
      TRITON_ASCEND_GIN_BACKEND_HCCL_PEER_MEM |
      TRITON_ASCEND_GIN_BACKEND_TILEXR_IPC_PEER_MEM |
      TRITON_ASCEND_GIN_BACKEND_HCCL_CHANNEL,
  TRITON_ASCEND_GIN_BACKEND_ALL =
      TRITON_ASCEND_GIN_BACKEND_HCCL_PEER_MEM |
      TRITON_ASCEND_GIN_BACKEND_TILEXR_UDMA |
      TRITON_ASCEND_GIN_BACKEND_HCOMM_TASK |
      TRITON_ASCEND_GIN_BACKEND_TILEXR_IPC_PEER_MEM |
      TRITON_ASCEND_GIN_BACKEND_UBS_CORE_UBC |
      TRITON_ASCEND_GIN_BACKEND_HCCL_CHANNEL,
};

enum TritonAscendGinBackendKind {
  TRITON_ASCEND_GIN_BACKEND_KIND_NONE = 0,
  TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_PEER_MEM = 1,
  TRITON_ASCEND_GIN_BACKEND_KIND_TILEXR_UDMA = 2,
  TRITON_ASCEND_GIN_BACKEND_KIND_HCOMM_TASK = 3,
  TRITON_ASCEND_GIN_BACKEND_KIND_TILEXR_IPC_PEER_MEM = 4,
  TRITON_ASCEND_GIN_BACKEND_KIND_UBS_CORE_UBC = 5,
  TRITON_ASCEND_GIN_BACKEND_KIND_UBS_COMM_URMA =
      TRITON_ASCEND_GIN_BACKEND_KIND_UBS_CORE_UBC,
  TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_CHANNEL = 6,
};

enum TritonAscendGinTeam {
  TRITON_ASCEND_GIN_TEAM_WORLD = 0,
  TRITON_ASCEND_GIN_TEAM_LOCAL = 1,
};

enum TritonAscendGinSem {
  TRITON_ASCEND_GIN_SEM_RELAXED = 0,
  TRITON_ASCEND_GIN_SEM_ACQUIRE = 2,
  TRITON_ASCEND_GIN_SEM_RELEASE = 3,
  TRITON_ASCEND_GIN_SEM_ACQ_REL = 4,
  TRITON_ASCEND_GIN_SEM_SEQ_CST = 5,
};

struct TritonAscendGinDev {
  uint32_t version;
  uint32_t backend_kind;
  uint32_t rank;
  uint32_t nranks;
  uint64_t flags;

  uint64_t peer_window_base[TRITON_ASCEND_GIN_MAX_RANKS];
  uint64_t peer_signal_base[TRITON_ASCEND_GIN_MAX_RANKS];
  uint64_t window_bytes;
  uint64_t signal_stride;
  uint32_t signal_slots;
  uint32_t reserved_signal;

  uint64_t tilexr_comm_args_dev;
  uint32_t tilexr_udma_mem_handle;
  uint32_t reserved0;

  uint64_t hccl_channel_handle[TRITON_ASCEND_GIN_MAX_RANKS];
  uint64_t hccl_thread_handle;
};

#ifdef __cplusplus
extern "C" {
#endif

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_rank(uint64_t dev_comm, int32_t team);
TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_num_ranks(uint64_t dev_comm, int32_t team);

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_put(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_ptr, uint64_t nbytes, int32_t sem, int32_t token);

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_put_signal(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_ptr, uint64_t nbytes,
    int32_t signal_id, uint64_t signal_value, int32_t sem, int32_t token);

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_put_window(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_window, uint64_t src_offset,
    uint64_t nbytes, int32_t sem, int32_t token);

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_put_signal_window(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_window, uint64_t dst_offset, uint64_t src_window, uint64_t src_offset,
    uint64_t nbytes, int32_t signal_id, uint64_t signal_value, int32_t sem, int32_t token);

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_get(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    uint64_t dst_ptr, uint64_t src_window, uint64_t src_offset, uint64_t nbytes, int32_t sem, int32_t token);

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_signal(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer,
    int32_t signal_id, uint64_t signal_value, int32_t sem, int32_t token);

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_wait_signal(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t peer,
    int32_t signal_id, uint64_t least_value, int32_t bits, int32_t sem, int32_t token);

TRITON_ASCEND_GIN_DEVICE uint64_t __triton_ascend_gin_read_signal(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t peer,
    int32_t signal_id, int32_t bits, int32_t sem);

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_reset_signal(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t peer,
    int32_t signal_id, int32_t token);

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_flush(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t peer, int32_t token);

TRITON_ASCEND_GIN_DEVICE int32_t __triton_ascend_gin_barrier(
    uint64_t dev_comm, uint32_t backend_mask, int32_t context_id, int32_t team, int32_t token);

#ifdef __cplusplus
}
#endif

#undef TRITON_ASCEND_GIN_DEVICE
#undef TRITON_ASCEND_GIN_GM

#endif  // TRITON_ASCEND_GIN_ABI_H_
