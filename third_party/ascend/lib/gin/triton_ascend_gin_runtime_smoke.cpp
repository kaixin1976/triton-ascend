#include "gin/triton_ascend_gin_runtime.h"

#include <acl/acl.h>

#include <cstdlib>
#include <iostream>

#include "tilexr_api.h"

namespace {

int GetEnvInt(const char *name, int fallback) {
  const char *value = std::getenv(name);
  return value == nullptr ? fallback : std::atoi(value);
}

bool Check(const char *step, int status) {
  if (status == 0) {
    std::cout << step << " ok" << std::endl;
    return true;
  }
  std::cerr << step << " failed: " << status << std::endl;
  return false;
}

}  // namespace

int main() {
  int rank = GetEnvInt("RANK", 0);
  int rankSize = GetEnvInt("RANK_SIZE", 1);
  int device = GetEnvInt("DEVICE_ID", rank);

  if (!Check("aclInit", aclInit(nullptr))) {
    return 1;
  }
  if (!Check("aclrtSetDevice", aclrtSetDevice(device))) {
    aclFinalize();
    return 1;
  }

  TileXRCommPtr tilexrComm = nullptr;
  if (!Check("TileXRCommInitRankLocal", TileXRCommInitRankLocal(rankSize, rank, &tilexrComm)) ||
      tilexrComm == nullptr) {
    aclrtResetDevice(device);
    aclFinalize();
    return 1;
  }

  TritonAscendGinHandle gin = nullptr;
  int status = TritonAscendGinCreateFromTileXR(tilexrComm, nullptr, &gin);
  if (status != TRITON_ASCEND_GIN_RUNTIME_SUCCESS || gin == nullptr) {
    std::cerr << "TritonAscendGinCreateFromTileXR failed: " << status
              << " " << TritonAscendGinGetLastError() << std::endl;
    TileXRCommDestroy(tilexrComm);
    aclrtResetDevice(device);
    aclFinalize();
    return 1;
  }

  TritonAscendGinDev hostDesc = {};
  uint64_t devComm = 0;
  bool ok = Check("TritonAscendGinGetHostComm", TritonAscendGinGetHostComm(gin, &hostDesc)) &&
            Check("TritonAscendGinGetDevComm", TritonAscendGinGetDevComm(gin, &devComm));
  ok = ok && hostDesc.backend_kind == TRITON_ASCEND_GIN_BACKEND_KIND_TILEXR_IPC_PEER_MEM;
  ok = ok && hostDesc.rank == static_cast<uint32_t>(rank);
  ok = ok && hostDesc.nranks == static_cast<uint32_t>(rankSize);
  ok = ok && devComm != 0;
  ok = ok && hostDesc.tilexr_comm_args_dev != 0;
  for (int peer = 0; peer < rankSize && peer < static_cast<int>(TRITON_ASCEND_GIN_MAX_RANKS); ++peer) {
    ok = ok && hostDesc.peer_window_base[peer] != 0;
    ok = ok && hostDesc.peer_signal_base[peer] != 0;
  }

  std::cout << "gin dev_comm=0x" << std::hex << devComm
            << " tilexr_args_dev=0x" << hostDesc.tilexr_comm_args_dev
            << " local_window=0x" << hostDesc.peer_window_base[rank]
            << " local_signal=0x" << hostDesc.peer_signal_base[rank]
            << std::dec << " rank=" << hostDesc.rank
            << " nranks=" << hostDesc.nranks << std::endl;
  for (int peer = 0; peer < rankSize && peer < static_cast<int>(TRITON_ASCEND_GIN_MAX_RANKS); ++peer) {
    std::cout << "peer[" << peer << "] window=0x" << std::hex << hostDesc.peer_window_base[peer]
              << " signal=0x" << hostDesc.peer_signal_base[peer] << std::dec << std::endl;
  }

  TritonAscendGinDestroy(gin);
  TileXRCommDestroy(tilexrComm);
  aclrtResetDevice(device);
  aclFinalize();
  return ok ? 0 : 1;
}
