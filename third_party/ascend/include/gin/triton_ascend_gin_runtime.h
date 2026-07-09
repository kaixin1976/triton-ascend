#ifndef TRITON_ASCEND_GIN_RUNTIME_H_
#define TRITON_ASCEND_GIN_RUNTIME_H_

#include <stdint.h>

#include "triton_ascend_gin_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *TritonAscendGinHandle;
typedef void *TritonAscendGinTileXRHandle;
typedef void *TritonAscendGinHcclHandle;

enum TritonAscendGinRuntimeStatus {
  TRITON_ASCEND_GIN_RUNTIME_SUCCESS = 0,
  TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE = 1,
  TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND = 2,
  TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR = 3,
  TRITON_ASCEND_GIN_RUNTIME_TILEXR_ERROR = 4,
  TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED = 5,
  TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR = 6,
};

struct TritonAscendGinTileXROptions {
  uint64_t ipc_data_offset;
  uint64_t window_bytes;
  uint64_t signal_stride;
  uint32_t signal_slots;
  uint32_t reserved0;
  const char *tilexr_library_path;
};

struct TritonAscendGinHcclPeerMemOptions {
  uint64_t window_bytes;
  uint64_t signal_stride;
  uint32_t signal_slots;
  uint32_t reserved0;
  const char *hccl_library_path;
};

int TritonAscendGinCreateFromTileXR(TritonAscendGinTileXRHandle tilexr_comm,
                                     const TritonAscendGinTileXROptions *options,
                                     TritonAscendGinHandle *handle);

int TritonAscendGinRefreshFromTileXR(TritonAscendGinHandle handle);

int TritonAscendGinCreateFromHcclPeerMem(TritonAscendGinHcclHandle hccl_comm,
                                          const TritonAscendGinHcclPeerMemOptions *options,
                                          TritonAscendGinHandle *handle);

int TritonAscendGinGetDevComm(TritonAscendGinHandle handle, uint64_t *dev_comm);

int TritonAscendGinGetHostComm(TritonAscendGinHandle handle, TritonAscendGinDev *host_comm);

int TritonAscendGinRegisterWindow(TritonAscendGinHandle handle, void *local_ptr, uint64_t bytes,
                                   const char *rendezvous_id, uint64_t *window_handle);

int TritonAscendGinRegisterSignalWindow(TritonAscendGinHandle handle, void *local_ptr, uint64_t bytes,
                                         const char *rendezvous_id, uint64_t *window_handle);

const char *TritonAscendGinGetLastError(void);

void TritonAscendGinDestroy(TritonAscendGinHandle handle);

#ifdef __cplusplus
}
#endif

#endif  // TRITON_ASCEND_GIN_RUNTIME_H_
