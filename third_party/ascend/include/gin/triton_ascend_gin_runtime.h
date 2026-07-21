#ifndef TRITON_ASCEND_GIN_RUNTIME_H_
#define TRITON_ASCEND_GIN_RUNTIME_H_

#include <stdint.h>

#include "triton_ascend_gin_abi.h"

#if defined(_WIN32)
#if defined(TRITON_ASCEND_GIN_RUNTIME_BUILD)
#define TRITON_ASCEND_GIN_RUNTIME_API __declspec(dllexport)
#else
#define TRITON_ASCEND_GIN_RUNTIME_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define TRITON_ASCEND_GIN_RUNTIME_API __attribute__((visibility("default")))
#else
#define TRITON_ASCEND_GIN_RUNTIME_API
#endif

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

struct TritonAscendGinHcclChannelOptions {
  uint64_t window_bytes;
  uint64_t signal_stride;
  uint32_t signal_slots;
  uint32_t engine;
  const char *hccl_library_path;
};

struct TritonAscendGinHcclChannelProbe {
  uint32_t struct_size;
  uint32_t rank;
  uint32_t nranks;
  uint32_t engine;
  uint32_t layer_count;
  uint32_t first_error_peer;
  uint64_t peer_mask;
  uint64_t protocol_mask;
  uint64_t hccs_peer_mask;
  uint64_t roce_peer_mask;
  uint64_t pcie_peer_mask;
  uint64_t ubc_ctp_peer_mask;
  uint64_t ubc_tp_peer_mask;
  uint64_t ub_mem_peer_mask;
  uint64_t acquire_peer_mask;
  uint64_t ccl_buffer_ptr;
  uint64_t ccl_buffer_bytes;
  uint64_t aiv_comm_info_ptr;
  uint64_t aiv_comm_info_bytes;
  uint64_t aiv_comm_info_mem_handle;
  int32_t ccl_buffer_status;
  int32_t aiv_comm_info_status;
  int32_t first_error_status;
  uint32_t reserved0;
  int32_t first_acquire_ret;
  uint32_t first_acquire_peer;
  uint32_t first_acquire_engine;
  uint32_t first_acquire_protocol;
  uint32_t first_acquire_channel_count;
  uint32_t first_acquire_mem_handle_count;
  uint64_t first_acquire_channel;
  uint32_t exchange_info_enabled;
  uint32_t exchange_info_symbol_available;
  int32_t exchange_info_status;
  uint32_t exchange_info_op_execute_config;
  uint64_t exchange_info_ccl_buffer_bytes;
  uint64_t exchange_info_count;
  uint32_t exchange_info_data_type;
  uint32_t exchange_info_aiv_core_limit;
  int32_t aiv_comm_info_layout_status;
  uint32_t reserved1;
  uint64_t aiv_comm_info_gm_in_offset;
  uint64_t aiv_comm_info_gm_out_offset;
  uint64_t aiv_comm_info_local_gm_in;
  uint64_t aiv_comm_info_local_gm_out;
  uint64_t aiv_comm_info_local_flag_base;
  uint32_t aiv_descriptor_via_cpu_enabled;
  int32_t aiv_descriptor_via_cpu_status;
  int32_t aiv_descriptor_via_cpu_first_ret;
  uint32_t aiv_descriptor_via_cpu_first_peer;
  uint32_t aiv_descriptor_via_cpu_first_protocol;
  uint32_t reserved2;
  uint64_t aiv_descriptor_via_cpu_first_channel;
  uint64_t aiv_descriptor_via_cpu_acquire_peer_mask;
  uint64_t aiv_descriptor_via_cpu_hccl_buffer_peer_mask;
  uint64_t aiv_descriptor_via_cpu_remote_mem_peer_mask;
  uint64_t aiv_descriptor_via_cpu_readback_gm_in_mask;
  uint64_t aiv_descriptor_via_cpu_readback_gm_out_mask;
  uint64_t aiv_descriptor_via_cpu_symmetric_gm_out_peer_mask;
};

struct TritonAscendGinHcclAlgBufferProbe {
  uint32_t struct_size;
  uint32_t device;
  uint64_t manager_storage_bytes;
  uint64_t ccl_buffer_ptr_before;
  uint64_t ccl_buffer_bytes_before;
  uint64_t ccl_buffer_ptr_after;
  uint64_t ccl_buffer_bytes_after;
  uint64_t in_aiv_opbase_devmem_ptr;
  uint64_t in_aiv_opbase_ptr;
  uint64_t in_aiv_opbase_bytes;
  uint64_t out_aiv_opbase_devmem_ptr;
  uint64_t out_aiv_opbase_ptr;
  uint64_t out_aiv_opbase_bytes;
  uint64_t aiv_comm_info_devmem_ptr;
  uint64_t aiv_comm_info_ptr;
  uint64_t aiv_comm_info_bytes;
  int32_t acl_set_device_status;
  int32_t get_independent_ccl_before_status;
  int32_t create_comm_aiv_buffer_status;
  int32_t create_comm_info_aiv_buffer_status;
  int32_t get_independent_ccl_after_status;
  int32_t clear_comm_aiv_buffer_status;
  int32_t release_comm_aiv_buffer_status;
  int32_t first_error_status;
};

struct TritonAscendGinHcclRdmaP2pProbe {
  uint32_t struct_size;
  uint32_t rank;
  uint32_t nranks;
  uint32_t src_rank;
  uint32_t dst_rank;
  uint32_t peer;
  uint32_t engine;
  uint32_t thread_engine;
  uint32_t protocol;
  uint32_t reserved0;
  uint64_t bytes;
  uint64_t channel;
  uint64_t thread;
  uint64_t local_ccl_buffer_ptr;
  uint64_t local_ccl_buffer_bytes;
  uint64_t remote_ccl_buffer_ptr;
  uint64_t remote_ccl_buffer_bytes;
  int32_t get_hccl_buffer_ret;
  int32_t channel_acquire_ret;
  int32_t channel_get_hccl_buffer_ret;
  int32_t thread_acquire_ret;
  int32_t local_copy_ret;
  int32_t write_ret;
  int32_t notify_ready_ret;
  int32_t wait_ready_ret;
  int32_t read_ret;
  int32_t notify_done_ret;
  int32_t wait_done_ret;
  int32_t thread_sync_ret;
  int32_t first_error_status;
};

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinCreateFromTileXR(
    TritonAscendGinTileXRHandle tilexr_comm, const TritonAscendGinTileXROptions *options,
    TritonAscendGinHandle *handle);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinRefreshFromTileXR(TritonAscendGinHandle handle);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinCreateFromHcclPeerMem(
    TritonAscendGinHcclHandle hccl_comm, const TritonAscendGinHcclPeerMemOptions *options,
    TritonAscendGinHandle *handle);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinCreateFromHcclChannel(
    TritonAscendGinHcclHandle hccl_comm, const TritonAscendGinHcclChannelOptions *options,
    TritonAscendGinHandle *handle);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinProbeHcclChannel(
    TritonAscendGinHcclHandle hccl_comm, const TritonAscendGinHcclChannelOptions *options,
    TritonAscendGinHcclChannelProbe *probe);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinProbeHcclAlgBuffers(
    const char *hccl_alg_library_path, uint32_t device,
    TritonAscendGinHcclAlgBufferProbe *probe);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinProbeHcclRdmaP2p(
    TritonAscendGinHcclHandle hccl_comm, const TritonAscendGinHcclChannelOptions *options,
    void *send_buf, void *recv_buf, uint64_t bytes, uint32_t src_rank, uint32_t dst_rank,
    TritonAscendGinHcclRdmaP2pProbe *probe);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinGetDevComm(TritonAscendGinHandle handle,
                                                            uint64_t *dev_comm);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinGetHostComm(TritonAscendGinHandle handle,
                                                             TritonAscendGinDev *host_comm);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinGetHcclBuffer(
    TritonAscendGinHandle handle, uint64_t *ptr, uint64_t *bytes);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinGetHcclAivOpbaseBuffer(
    TritonAscendGinHandle handle, uint32_t which, uint64_t *ptr, uint64_t *bytes);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinHcclAivAllGather(
    TritonAscendGinHandle handle, void *send_buf, void *recv_buf,
    uint64_t send_count, uint32_t data_type, void *stream,
    const char *rendezvous_id);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinRegisterWindow(
    TritonAscendGinHandle handle, void *local_ptr, uint64_t bytes, const char *rendezvous_id,
    uint64_t *window_handle);

TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinRegisterSignalWindow(
    TritonAscendGinHandle handle, void *local_ptr, uint64_t bytes, const char *rendezvous_id,
    uint64_t *window_handle);

TRITON_ASCEND_GIN_RUNTIME_API void *TritonAscendGinHcclAllocatorAlloc(
    int64_t size, int device, void *stream);

TRITON_ASCEND_GIN_RUNTIME_API void TritonAscendGinHcclAllocatorFree(
    void *ptr, uint64_t size, void *stream);

TRITON_ASCEND_GIN_RUNTIME_API const char *TritonAscendGinGetLastError(void);

TRITON_ASCEND_GIN_RUNTIME_API void TritonAscendGinDestroy(TritonAscendGinHandle handle);

#ifdef __cplusplus
}
#endif

#endif  // TRITON_ASCEND_GIN_RUNTIME_H_
