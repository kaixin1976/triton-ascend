#include "gin/triton_ascend_gin_runtime.h"

#include <acl/acl_rt.h>

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <thread>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace {

constexpr uint32_t kTileXRMaxRanks = TRITON_ASCEND_GIN_MAX_RANKS;
constexpr uint64_t kDefaultTileXRIpcDataOffset = 2ull * 1024ull * 1024ull;
constexpr uint64_t kDefaultTileXRWindowBytes = 100ull * 1024ull * 1024ull;
constexpr uint64_t kDefaultSignalStride = sizeof(uint64_t);
constexpr uint32_t kDefaultSignalSlots = TRITON_ASCEND_GIN_DEFAULT_SIGNAL_SLOTS;
constexpr uint32_t kDescriptorVersion = 1;
constexpr size_t kAclIpcKeyBytes = 65;
constexpr uint32_t kWindowHandleDefault = 1;
constexpr uint64_t kMaxWindowBaseSearchOffset = 2ull * 1024ull * 1024ull;
constexpr uint64_t kWindowExportGranularity = 4096;

thread_local std::string g_lastError;

void SetLastError(const std::string &message) {
  g_lastError = message;
}

void SetAclError(const char *what, aclError err) {
  g_lastError = std::string(what) + " failed with acl error " + std::to_string(static_cast<int>(err));
}

bool DebugEnabled() {
  const char *value = std::getenv("TRITON_ASCEND_GIN_DEBUG");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

struct TileXRCommArgsCompat {
  int32_t rank;
  int32_t localRank;
  int32_t rankSize;
  int32_t localRankSize;
  uint32_t extraFlag;
  void *peerMems[kTileXRMaxRanks];
};

static_assert(offsetof(TileXRCommArgsCompat, peerMems) == 24,
              "TileXR CommArgs compatible prefix layout changed");

using TileXRGetCommArgsHostFn = int (*)(void *, TileXRCommArgsCompat **);
using TileXRGetCommArgsDevFn = int (*)(void *, void **);

struct TileXRSymbols {
  TileXRGetCommArgsHostFn getHostArgs = nullptr;
  TileXRGetCommArgsDevFn getDevArgs = nullptr;
  void *library = nullptr;
  bool ownsLibrary = false;
};

using HcclGetRankIdFn = int (*)(void *, uint32_t *);
using HcclGetRankSizeFn = int (*)(void *, uint32_t *);
using HcclCommSymWinGetFn = int (*)(void *, void *, size_t, void **, size_t *);
using HcclSymWinGetPeerPointerFn = int (*)(void *, size_t, uint32_t, void **);

struct HcclPeerMemSymbols {
  HcclGetRankIdFn getRankId = nullptr;
  HcclGetRankSizeFn getRankSize = nullptr;
  HcclCommSymWinGetFn commSymWinGet = nullptr;
  HcclSymWinGetPeerPointerFn symWinGetPeerPointer = nullptr;
  void *libraries[4] = {};
  uint32_t libraryCount = 0;
};

struct TritonAscendGinRuntimeHandle {
  void *tilexrComm = nullptr;
  TileXRSymbols tilexr;
  void *hcclComm = nullptr;
  HcclPeerMemSymbols hccl;
  TritonAscendGinDev hostDesc = {};
  void *devDesc = nullptr;
  uint64_t ipcDataOffset = kDefaultTileXRIpcDataOffset;
  uint64_t windowBytes = kDefaultTileXRWindowBytes;
  uint64_t signalStride = kDefaultSignalStride;
  uint32_t signalSlots = kDefaultSignalSlots;
  bool userWindowRegistered = false;
  uint64_t userWindowBytes = 0;
  std::string userWindowKeys[kTileXRMaxRanks];
  void *userWindowPtrs[kTileXRMaxRanks] = {};
  bool userWindowImported[kTileXRMaxRanks] = {};
};

struct WindowRendezvousInfo {
  std::string key;
  uint64_t windowOffset = 0;
  uint64_t bytes = 0;
};

#if defined(__linux__)
void *ResolveDefaultSymbol(const char *name) {
  return dlsym(RTLD_DEFAULT, name);
}

template <typename Fn>
void ResolveOne(Fn *slot, void *library, const char *name) {
  if (*slot != nullptr || library == nullptr) {
    return;
  }
  *slot = reinterpret_cast<Fn>(dlsym(library, name));
}

bool HcclSymbolsComplete(const HcclPeerMemSymbols &symbols) {
  return symbols.getRankId != nullptr && symbols.getRankSize != nullptr &&
         symbols.commSymWinGet != nullptr && symbols.symWinGetPeerPointer != nullptr;
}

void ResolveHcclSymbolsFromHandle(HcclPeerMemSymbols *symbols, void *library) {
  ResolveOne(&symbols->getRankId, library, "HcclGetRankId");
  ResolveOne(&symbols->getRankSize, library, "HcclGetRankSize");
  ResolveOne(&symbols->commSymWinGet, library, "HcclCommSymWinGet");
  ResolveOne(&symbols->symWinGetPeerPointer, library, "HcclSymWinGetPeerPointer");
}

void ResolveHcclSymbolsFromDefault(HcclPeerMemSymbols *symbols) {
  if (symbols == nullptr) {
    return;
  }
  symbols->getRankId = reinterpret_cast<HcclGetRankIdFn>(ResolveDefaultSymbol("HcclGetRankId"));
  symbols->getRankSize = reinterpret_cast<HcclGetRankSizeFn>(ResolveDefaultSymbol("HcclGetRankSize"));
  symbols->commSymWinGet =
      reinterpret_cast<HcclCommSymWinGetFn>(ResolveDefaultSymbol("HcclCommSymWinGet"));
  symbols->symWinGetPeerPointer =
      reinterpret_cast<HcclSymWinGetPeerPointerFn>(ResolveDefaultSymbol("HcclSymWinGetPeerPointer"));
}

int LoadHcclPeerMemSymbols(const char *libraryPath, HcclPeerMemSymbols *symbols) {
  if (symbols == nullptr) {
    SetLastError("HCCL symbol output is null");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  ResolveHcclSymbolsFromDefault(symbols);
  if (HcclSymbolsComplete(*symbols)) {
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }

  const char *envPath = std::getenv("TRITON_ASCEND_HCCL_LIB");
  const char *candidates[4] = {
      libraryPath,
      envPath,
      "libhccl.so",
      "libhcomm.so",
  };

  std::string dlErrors;
  for (const char *candidate : candidates) {
    if (candidate == nullptr || candidate[0] == '\0') {
      continue;
    }
    void *library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      const char *err = dlerror();
      if (err != nullptr) {
        dlErrors += std::string(candidate) + ": " + err + "; ";
      }
      continue;
    }
    if (symbols->libraryCount < 2) {
      symbols->libraries[symbols->libraryCount++] = library;
    }
    ResolveHcclSymbolsFromHandle(symbols, library);
    if (HcclSymbolsComplete(*symbols)) {
      return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
    }
  }

  SetLastError("failed to resolve HCCL peer-memory symbols "
               "(HcclGetRankId, HcclGetRankSize, HcclCommSymWinGet, "
               "HcclSymWinGetPeerPointer). " +
               dlErrors);
  return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
}

void CloseHcclPeerMemSymbols(HcclPeerMemSymbols *symbols) {
  if (symbols == nullptr) {
    return;
  }
  for (uint32_t i = 0; i < symbols->libraryCount; ++i) {
    if (symbols->libraries[i] != nullptr) {
      dlclose(symbols->libraries[i]);
      symbols->libraries[i] = nullptr;
    }
  }
  symbols->libraryCount = 0;
}

int LoadTileXRSymbols(const char *libraryPath, TileXRSymbols *symbols) {
  if (symbols == nullptr) {
    SetLastError("TileXR symbol output is null");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  symbols->getHostArgs = reinterpret_cast<TileXRGetCommArgsHostFn>(
      ResolveDefaultSymbol("TileXRGetCommArgsHost"));
  symbols->getDevArgs = reinterpret_cast<TileXRGetCommArgsDevFn>(
      ResolveDefaultSymbol("TileXRGetCommArgsDev"));
  if (symbols->getHostArgs != nullptr) {
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }

  const char *path = libraryPath;
  if (path == nullptr || path[0] == '\0') {
    path = std::getenv("TRITON_ASCEND_TILEXR_LIB");
  }
  if (path == nullptr || path[0] == '\0') {
    path = "libtile-comm.so";
  }

  void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    SetLastError(std::string("dlopen TileXR library failed: ") + dlerror());
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }

  symbols->library = library;
  symbols->ownsLibrary = true;
  symbols->getHostArgs = reinterpret_cast<TileXRGetCommArgsHostFn>(
      dlsym(library, "TileXRGetCommArgsHost"));
  symbols->getDevArgs = reinterpret_cast<TileXRGetCommArgsDevFn>(
      dlsym(library, "TileXRGetCommArgsDev"));
  if (symbols->getHostArgs == nullptr) {
    SetLastError("TileXRGetCommArgsHost symbol was not found");
    dlclose(library);
    symbols->library = nullptr;
    symbols->ownsLibrary = false;
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

void CloseTileXRSymbols(TileXRSymbols *symbols) {
  if (symbols != nullptr && symbols->ownsLibrary && symbols->library != nullptr) {
    dlclose(symbols->library);
    symbols->library = nullptr;
    symbols->ownsLibrary = false;
  }
}
#else
int LoadHcclPeerMemSymbols(const char *, HcclPeerMemSymbols *) {
  SetLastError("Triton Ascend HCCL peer-memory bridge is only implemented for Linux hosts");
  return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
}

void CloseHcclPeerMemSymbols(HcclPeerMemSymbols *) {}

int LoadTileXRSymbols(const char *, TileXRSymbols *) {
  SetLastError("Triton Ascend TileXR runtime bridge is only implemented for Linux hosts");
  return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
}

void CloseTileXRSymbols(TileXRSymbols *) {}
#endif

uint64_t PtrToU64(const void *ptr) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}

std::string EnvOrDefault(const char *name, const char *fallback) {
  const char *value = std::getenv(name);
  if (value != nullptr && value[0] != '\0') {
    return std::string(value);
  }
  return fallback == nullptr ? std::string() : std::string(fallback);
}

std::string SanitizeRendezvousId(const char *raw) {
  const char *fallback = std::getenv("TRITON_ASCEND_GIN_WINDOW_ID");
  if (fallback == nullptr || fallback[0] == '\0') {
    fallback = std::getenv("TRITON_ASCEND_GIN_WINDOW_ID");
  }
  if (fallback == nullptr || fallback[0] == '\0') {
    fallback = "default";
  }
  std::string id = raw != nullptr && raw[0] != '\0' ? raw : fallback;
  if (id.empty()) {
    id = "default";
  }
  for (char &ch : id) {
    bool keep = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.';
    if (!keep) {
      ch = '_';
    }
  }
  return id;
}

int GetEnvInt(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return std::atoi(value);
}

std::string WindowKeyPath(const std::string &dir, const std::string &id, uint32_t rank) {
  std::ostringstream os;
  os << dir << "/triton_ascend_gin_window_" << id << "_rank" << rank << ".key";
  return os.str();
}

std::string WindowReadyPath(const std::string &dir, const std::string &id, uint32_t rank) {
  std::ostringstream os;
  os << dir << "/triton_ascend_gin_window_" << id << "_rank" << rank << ".ready";
  return os.str();
}

bool WriteTextFile(const std::string &path, const std::string &value) {
  const std::string tmpPath = path + ".tmp";
  (void)std::remove(tmpPath.c_str());
  std::ofstream out(tmpPath, std::ios::out | std::ios::trunc);
  if (!out) {
    return false;
  }
  out << value << "\n";
  out.close();
  if (!out) {
    (void)std::remove(tmpPath.c_str());
    return false;
  }
  if (std::rename(tmpPath.c_str(), path.c_str()) != 0) {
    (void)std::remove(tmpPath.c_str());
    return false;
  }
  return true;
}

bool ReadTextFile(const std::string &path, std::string *value) {
  if (value == nullptr) {
    return false;
  }
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  std::getline(in, *value);
  return !value->empty();
}

bool ParseWindowRendezvousInfo(const std::string &raw, WindowRendezvousInfo *info) {
  if (info == nullptr) {
    return false;
  }
  std::istringstream in(raw);
  WindowRendezvousInfo next = {};
  if (!(in >> next.key)) {
    return false;
  }
  if (next.key.size() < kAclIpcKeyBytes - 1) {
    return false;
  }
  if (!(in >> next.windowOffset)) {
    next.windowOffset = 0;
  }
  if (!(in >> next.bytes)) {
    next.bytes = 0;
  }
  *info = next;
  return true;
}

uint64_t RoundUp(uint64_t value, uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  uint64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return value + (alignment - remainder);
}

aclError ExportUserWindowKey(void *localPtr, uint64_t bytes, char *key, size_t keyBytes,
                             uint64_t *windowOffset, uint64_t *exportBytes) {
  if (localPtr == nullptr || key == nullptr || keyBytes == 0 || windowOffset == nullptr ||
      exportBytes == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }

  const uintptr_t ptr = reinterpret_cast<uintptr_t>(localPtr);
  const uint64_t alignments[] = {0, 4096, 65536, 2ull * 1024ull * 1024ull};
  uintptr_t tried[sizeof(alignments) / sizeof(alignments[0])] = {};
  size_t triedCount = 0;
  aclError lastErr = ACL_ERROR_INVALID_PARAM;

  for (uint64_t alignment : alignments) {
    uintptr_t base = ptr;
    if (alignment != 0) {
      base = ptr & ~(static_cast<uintptr_t>(alignment) - 1);
    }
    bool seen = false;
    for (size_t i = 0; i < triedCount; ++i) {
      if (tried[i] == base) {
        seen = true;
        break;
      }
    }
    if (seen || base == 0 || base > ptr) {
      continue;
    }
    tried[triedCount++] = base;

    uint64_t offset = static_cast<uint64_t>(ptr - base);
    if (offset > kMaxWindowBaseSearchOffset ||
        bytes > std::numeric_limits<uint64_t>::max() - offset) {
      continue;
    }

    uint64_t neededBytes = bytes + offset;
    uint64_t roundedBytes = RoundUp(neededBytes, kWindowExportGranularity);
    uint64_t byteCandidates[] = {roundedBytes, neededBytes};
    for (uint64_t candidateBytes : byteCandidates) {
      if (candidateBytes < neededBytes) {
        continue;
      }
      char candidateKey[kAclIpcKeyBytes] = {};
      lastErr = aclrtIpcMemGetExportKey(reinterpret_cast<void *>(base),
                                        static_cast<size_t>(candidateBytes),
                                        candidateKey, keyBytes,
                                        ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION);
      if (lastErr == ACL_SUCCESS) {
        std::memset(key, 0, keyBytes);
        std::memcpy(key, candidateKey, keyBytes < kAclIpcKeyBytes ? keyBytes : kAclIpcKeyBytes);
        *windowOffset = offset;
        *exportBytes = candidateBytes;
        return ACL_SUCCESS;
      }
      if (candidateBytes == neededBytes) {
        break;
      }
    }
  }

  return lastErr;
}

void ClearRegisteredUserWindow(TritonAscendGinRuntimeHandle *handle) {
  if (handle == nullptr) {
    return;
  }
  for (uint32_t rank = 0; rank < handle->hostDesc.nranks && rank < kTileXRMaxRanks; ++rank) {
    if (handle->userWindowImported[rank] && !handle->userWindowKeys[rank].empty()) {
      (void)aclrtIpcMemClose(handle->userWindowKeys[rank].c_str());
    }
    handle->userWindowKeys[rank].clear();
    handle->userWindowPtrs[rank] = nullptr;
    handle->userWindowImported[rank] = false;
  }
  handle->userWindowRegistered = false;
  handle->userWindowBytes = 0;
}

int CopyDescriptorToDevice(TritonAscendGinRuntimeHandle *handle) {
  if (handle == nullptr) {
    SetLastError("runtime handle is null");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (handle->devDesc == nullptr) {
    aclError err = aclrtMalloc(&handle->devDesc, sizeof(handle->hostDesc), ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_SUCCESS) {
      SetAclError("aclrtMalloc TritonAscendGinDev", err);
      return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
    }
  }
  aclError err = aclrtMemcpy(handle->devDesc, sizeof(handle->hostDesc), &handle->hostDesc,
                             sizeof(handle->hostDesc), ACL_MEMCPY_HOST_TO_DEVICE);
  if (err != ACL_SUCCESS) {
    SetAclError("aclrtMemcpy TritonAscendGinDev", err);
    return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

int FillDescriptorFromTileXR(TritonAscendGinRuntimeHandle *handle) {
  if (handle == nullptr || handle->tilexrComm == nullptr) {
    SetLastError("TileXR runtime handle is null");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (handle->tilexr.getHostArgs == nullptr) {
    SetLastError("TileXRGetCommArgsHost is unavailable");
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }

  TileXRCommArgsCompat *tilexrArgs = nullptr;
  int ret = handle->tilexr.getHostArgs(handle->tilexrComm, &tilexrArgs);
  if (ret != 0 || tilexrArgs == nullptr) {
    SetLastError("TileXRGetCommArgsHost failed with code " + std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_TILEXR_ERROR;
  }
  if (tilexrArgs->rank < 0 || tilexrArgs->rankSize <= 0 ||
      tilexrArgs->rank >= tilexrArgs->rankSize ||
      tilexrArgs->rankSize > static_cast<int32_t>(kTileXRMaxRanks)) {
    SetLastError("TileXR CommArgs rank metadata is invalid");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  TritonAscendGinDev next = {};
  next.version = kDescriptorVersion;
  next.backend_kind = TRITON_ASCEND_GIN_BACKEND_KIND_TILEXR_IPC_PEER_MEM;
  next.rank = static_cast<uint32_t>(tilexrArgs->rank);
  next.nranks = static_cast<uint32_t>(tilexrArgs->rankSize);
  next.flags = tilexrArgs->extraFlag;
  next.window_bytes = handle->windowBytes;
  next.signal_stride = handle->signalStride;
  next.signal_slots = handle->signalSlots;

  for (int32_t rank = 0; rank < tilexrArgs->rankSize; ++rank) {
    uint64_t base = PtrToU64(tilexrArgs->peerMems[rank]);
    next.peer_signal_base[rank] = base;
    next.peer_window_base[rank] = base == 0 ? 0 : base + handle->ipcDataOffset;
  }

  if (handle->tilexr.getDevArgs != nullptr) {
    void *tilexrDevArgs = nullptr;
    ret = handle->tilexr.getDevArgs(handle->tilexrComm, &tilexrDevArgs);
    if (ret == 0) {
      next.tilexr_comm_args_dev = PtrToU64(tilexrDevArgs);
    }
  }

  handle->hostDesc = next;
  return CopyDescriptorToDevice(handle);
}

int FillDescriptorFromHcclPeerMem(TritonAscendGinRuntimeHandle *handle) {
  if (handle == nullptr || handle->hcclComm == nullptr) {
    SetLastError("HCCL runtime handle is null");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (!HcclSymbolsComplete(handle->hccl)) {
    SetLastError("HCCL peer-memory symbols are unavailable");
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }

  uint32_t rank = 0;
  uint32_t nranks = 0;
  int ret = handle->hccl.getRankId(handle->hcclComm, &rank);
  if (ret != 0) {
    SetLastError("HcclGetRankId failed with code " + std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  ret = handle->hccl.getRankSize(handle->hcclComm, &nranks);
  if (ret != 0) {
    SetLastError("HcclGetRankSize failed with code " + std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  if (nranks == 0 || nranks > kTileXRMaxRanks || rank >= nranks) {
    SetLastError("HCCL rank metadata is invalid");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  TritonAscendGinDev next = handle->hostDesc;
  next.version = kDescriptorVersion;
  next.backend_kind = TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_PEER_MEM;
  next.rank = rank;
  next.nranks = nranks;
  next.window_bytes = handle->windowBytes;
  next.signal_stride = handle->signalStride;
  next.signal_slots = handle->signalSlots;
  handle->hostDesc = next;
  return CopyDescriptorToDevice(handle);
}

int RegisterHcclSymmetricBases(TritonAscendGinRuntimeHandle *runtime, void *localPtr, uint64_t bytes,
                               const char *what, uint64_t *bases, uint64_t *windowHandle) {
  if (runtime == nullptr || localPtr == nullptr || bases == nullptr || windowHandle == nullptr || bytes == 0) {
    SetLastError(std::string("TritonAscendGinRegister") + what + " received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (runtime->hostDesc.backend_kind != TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_PEER_MEM ||
      runtime->hcclComm == nullptr) {
    SetLastError(std::string("TritonAscendGinRegister") + what +
                 " requires an HCCL peer-memory communicator");
    return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
  }

  void *symWindow = nullptr;
  size_t offset = 0;
  int ret = runtime->hccl.commSymWinGet(runtime->hcclComm, localPtr, static_cast<size_t>(bytes),
                                       &symWindow, &offset);
  if (ret != 0 || symWindow == nullptr) {
    SetLastError(std::string("HcclCommSymWinGet for ") + what + " failed with code " +
                 std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }

  const uint32_t rank = runtime->hostDesc.rank;
  const uint32_t nranks = runtime->hostDesc.nranks;
  for (uint32_t peer = 0; peer < nranks; ++peer) {
    if (peer == rank) {
      bases[peer] = PtrToU64(localPtr);
      continue;
    }
    void *peerPtr = nullptr;
    ret = runtime->hccl.symWinGetPeerPointer(symWindow, offset, peer, &peerPtr);
    if (ret != 0 || peerPtr == nullptr) {
      SetLastError(std::string("HcclSymWinGetPeerPointer for ") + what + " peer " +
                   std::to_string(peer) + " failed with code " + std::to_string(ret));
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }
    bases[peer] = PtrToU64(peerPtr);
  }

  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] hccl register_%s rank=%u nranks=%u ptr=%p bytes=%llu offset=%llu\n",
                 what, rank, nranks, localPtr, static_cast<unsigned long long>(bytes),
                 static_cast<unsigned long long>(offset));
  }

  *windowHandle = kWindowHandleDefault;
  return CopyDescriptorToDevice(runtime);
}

TritonAscendGinRuntimeHandle *AsHandle(TritonAscendGinHandle handle) {
  return static_cast<TritonAscendGinRuntimeHandle *>(handle);
}

}  // namespace

extern "C" int TritonAscendGinCreateFromTileXR(TritonAscendGinTileXRHandle tilexrComm,
                                                 const TritonAscendGinTileXROptions *options,
                                                 TritonAscendGinHandle *handle) {
  if (handle == nullptr || tilexrComm == nullptr) {
    SetLastError("TritonAscendGinCreateFromTileXR received null input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  *handle = nullptr;

  TritonAscendGinRuntimeHandle *next = new (std::nothrow) TritonAscendGinRuntimeHandle();
  if (next == nullptr) {
    SetLastError("failed to allocate TritonAscendGinRuntimeHandle");
    return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
  }

  next->tilexrComm = tilexrComm;
  if (options != nullptr) {
    next->ipcDataOffset = options->ipc_data_offset == 0 ? kDefaultTileXRIpcDataOffset : options->ipc_data_offset;
    next->windowBytes = options->window_bytes == 0 ? kDefaultTileXRWindowBytes : options->window_bytes;
    next->signalStride = options->signal_stride == 0 ? kDefaultSignalStride : options->signal_stride;
    next->signalSlots = options->signal_slots == 0 ? kDefaultSignalSlots : options->signal_slots;
  }

  int ret = LoadTileXRSymbols(options == nullptr ? nullptr : options->tilexr_library_path, &next->tilexr);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    delete next;
    return ret;
  }

  ret = FillDescriptorFromTileXR(next);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    TritonAscendGinDestroy(next);
    return ret;
  }

  *handle = next;
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

extern "C" int TritonAscendGinRefreshFromTileXR(TritonAscendGinHandle handle) {
  return FillDescriptorFromTileXR(AsHandle(handle));
}

extern "C" int TritonAscendGinCreateFromHcclPeerMem(TritonAscendGinHcclHandle hcclComm,
                                                      const TritonAscendGinHcclPeerMemOptions *options,
                                                      TritonAscendGinHandle *handle) {
  if (handle == nullptr || hcclComm == nullptr) {
    SetLastError("TritonAscendGinCreateFromHcclPeerMem received null input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  *handle = nullptr;

  TritonAscendGinRuntimeHandle *next = new (std::nothrow) TritonAscendGinRuntimeHandle();
  if (next == nullptr) {
    SetLastError("failed to allocate TritonAscendGinRuntimeHandle");
    return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
  }

  next->hcclComm = hcclComm;
  if (options != nullptr) {
    next->windowBytes = options->window_bytes == 0 ? kDefaultTileXRWindowBytes : options->window_bytes;
    next->signalStride = options->signal_stride == 0 ? kDefaultSignalStride : options->signal_stride;
    next->signalSlots = options->signal_slots == 0 ? kDefaultSignalSlots : options->signal_slots;
  }

  int ret = LoadHcclPeerMemSymbols(options == nullptr ? nullptr : options->hccl_library_path, &next->hccl);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    delete next;
    return ret;
  }

  ret = FillDescriptorFromHcclPeerMem(next);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    TritonAscendGinDestroy(next);
    return ret;
  }

  *handle = next;
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

extern "C" int TritonAscendGinGetDevComm(TritonAscendGinHandle handle, uint64_t *devComm) {
  auto *runtime = AsHandle(handle);
  if (runtime == nullptr || devComm == nullptr || runtime->devDesc == nullptr) {
    SetLastError("TritonAscendGinGetDevComm received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  *devComm = PtrToU64(runtime->devDesc);
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

extern "C" int TritonAscendGinGetHostComm(TritonAscendGinHandle handle, TritonAscendGinDev *hostComm) {
  auto *runtime = AsHandle(handle);
  if (runtime == nullptr || hostComm == nullptr) {
    SetLastError("TritonAscendGinGetHostComm received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  std::memcpy(hostComm, &runtime->hostDesc, sizeof(*hostComm));
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

extern "C" int TritonAscendGinRegisterWindow(TritonAscendGinHandle handle, void *localPtr, uint64_t bytes,
                                               const char *rendezvousId, uint64_t *windowHandle) {
  auto *runtime = AsHandle(handle);
  if (runtime == nullptr || localPtr == nullptr || bytes == 0 || windowHandle == nullptr) {
    SetLastError("TritonAscendGinRegisterWindow received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (runtime->hostDesc.nranks == 0 || runtime->hostDesc.nranks > kTileXRMaxRanks) {
    SetLastError("TritonAscendGinRegisterWindow requires an initialized communicator");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (runtime->hostDesc.backend_kind == TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_PEER_MEM) {
    (void)rendezvousId;
    runtime->windowBytes = bytes;
    runtime->hostDesc.window_bytes = bytes;
    return RegisterHcclSymmetricBases(runtime, localPtr, bytes, "window",
                                      runtime->hostDesc.peer_window_base, windowHandle);
  }

  ClearRegisteredUserWindow(runtime);

  char localKey[kAclIpcKeyBytes] = {};
  uint64_t localWindowOffset = 0;
  uint64_t localExportBytes = 0;
  aclError err = ExportUserWindowKey(localPtr, bytes, localKey, sizeof(localKey),
                                     &localWindowOffset, &localExportBytes);
  if (DebugEnabled()) {
    int32_t deviceId = -1;
    aclError devErr = aclrtGetDevice(&deviceId);
    std::fprintf(stderr,
                 "[triton_ascend_gin] rank=%u register_window ptr=%p bytes=%llu offset=%llu "
                 "export_bytes=%llu device=%d get_device_err=%d export_err=%d key=%s\n",
                 runtime->hostDesc.rank, localPtr, static_cast<unsigned long long>(bytes),
                 static_cast<unsigned long long>(localWindowOffset),
                 static_cast<unsigned long long>(localExportBytes),
                 deviceId, static_cast<int>(devErr), static_cast<int>(err), localKey);
  }
  if (err != ACL_SUCCESS) {
    SetAclError("aclrtIpcMemGetExportKey user window", err);
    return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
  }

  const std::string id = SanitizeRendezvousId(rendezvousId);
  const std::string legacyDir = EnvOrDefault("TRITON_ASCEND_GIN_RENDEZVOUS_DIR", "/tmp");
  const std::string dir = EnvOrDefault("TRITON_ASCEND_GIN_RENDEZVOUS_DIR", legacyDir.c_str());
  const uint32_t rank = runtime->hostDesc.rank;
  const uint32_t nranks = runtime->hostDesc.nranks;
  const std::string localPath = WindowKeyPath(dir, id, rank);
  const std::string localReadyPath = WindowReadyPath(dir, id, rank);
  (void)std::remove(localPath.c_str());
  (void)std::remove(localReadyPath.c_str());
  std::ostringstream localInfo;
  localInfo << localKey << " " << localWindowOffset << " " << bytes;
  if (!WriteTextFile(localPath, localInfo.str())) {
    SetLastError("failed to write window rendezvous key file: " + localPath);
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }

  const int timeoutMs = GetEnvInt(
      "TRITON_ASCEND_GIN_RENDEZVOUS_TIMEOUT_MS",
      GetEnvInt("TRITON_ASCEND_GIN_RENDEZVOUS_TIMEOUT_MS", 120000));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  std::string rawInfos[kTileXRMaxRanks];
  WindowRendezvousInfo windowInfos[kTileXRMaxRanks];
  while (true) {
    bool ready = true;
    for (uint32_t peer = 0; peer < nranks; ++peer) {
      if (rawInfos[peer].empty()) {
        bool peerReady = ReadTextFile(WindowKeyPath(dir, id, peer), &rawInfos[peer]);
        if (!peerReady || !ParseWindowRendezvousInfo(rawInfos[peer], &windowInfos[peer])) {
          rawInfos[peer].clear();
          ready = false;
        }
      }
      if (rawInfos[peer].empty()) {
        ready = false;
      }
    }
    if (ready) {
      break;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      SetLastError("timeout waiting for window rendezvous id " + id);
      return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  runtime->hostDesc.window_bytes = bytes;
  runtime->userWindowBytes = bytes;
  for (uint32_t peer = 0; peer < nranks; ++peer) {
    runtime->userWindowKeys[peer] = windowInfos[peer].key;
    if (peer == rank) {
      runtime->userWindowPtrs[peer] = localPtr;
      runtime->hostDesc.peer_window_base[peer] = PtrToU64(localPtr);
      continue;
    }
    void *peerPtr = nullptr;
    err = aclrtIpcMemImportByKey(&peerPtr, windowInfos[peer].key.c_str(),
                                 ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS);
    if (DebugEnabled()) {
      int32_t deviceId = -1;
      aclError devErr = aclrtGetDevice(&deviceId);
      std::fprintf(stderr,
                   "[triton_ascend_gin] rank=%u import_window peer=%u device=%d get_device_err=%d "
                   "import_err=%d peer_ptr=%p offset=%llu key=%s\n",
                   rank, peer, deviceId, static_cast<int>(devErr), static_cast<int>(err),
                   peerPtr, static_cast<unsigned long long>(windowInfos[peer].windowOffset),
                   windowInfos[peer].key.c_str());
    }
    if (err != ACL_SUCCESS || peerPtr == nullptr) {
      SetAclError("aclrtIpcMemImportByKey user window", err);
      ClearRegisteredUserWindow(runtime);
      return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
    }
    runtime->userWindowPtrs[peer] = peerPtr;
    runtime->hostDesc.peer_window_base[peer] = PtrToU64(peerPtr) + windowInfos[peer].windowOffset;
    runtime->userWindowImported[peer] = true;
  }

  if (!WriteTextFile(localReadyPath, "ready")) {
    ClearRegisteredUserWindow(runtime);
    SetLastError("failed to write window rendezvous ready file: " + localReadyPath);
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }
  const auto readyDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (true) {
    bool ready = true;
    for (uint32_t peer = 0; peer < nranks; ++peer) {
      std::string ignored;
      if (!ReadTextFile(WindowReadyPath(dir, id, peer), &ignored)) {
        ready = false;
        break;
      }
    }
    if (ready) {
      break;
    }
    if (std::chrono::steady_clock::now() > readyDeadline) {
      ClearRegisteredUserWindow(runtime);
      SetLastError("timeout waiting for window import rendezvous id " + id);
      return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  runtime->userWindowRegistered = true;
  int ret = CopyDescriptorToDevice(runtime);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    ClearRegisteredUserWindow(runtime);
    return ret;
  }

  *windowHandle = kWindowHandleDefault;
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

extern "C" int TritonAscendGinRegisterSignalWindow(TritonAscendGinHandle handle, void *localPtr,
                                                     uint64_t bytes, const char *rendezvousId,
                                                     uint64_t *windowHandle) {
  auto *runtime = AsHandle(handle);
  if (runtime == nullptr || localPtr == nullptr || bytes == 0 || windowHandle == nullptr) {
    SetLastError("TritonAscendGinRegisterSignalWindow received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (runtime->hostDesc.nranks == 0 || runtime->hostDesc.nranks > kTileXRMaxRanks) {
    SetLastError("TritonAscendGinRegisterSignalWindow requires an initialized communicator");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (runtime->hostDesc.backend_kind != TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_PEER_MEM) {
    SetLastError("TritonAscendGinRegisterSignalWindow is currently only required for "
                 "HCCL peer-memory communicators");
    return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
  }
  (void)rendezvousId;
  return RegisterHcclSymmetricBases(runtime, localPtr, bytes, "signal_window",
                                    runtime->hostDesc.peer_signal_base, windowHandle);
}

extern "C" const char *TritonAscendGinGetLastError(void) {
  return g_lastError.c_str();
}

extern "C" void TritonAscendGinDestroy(TritonAscendGinHandle handle) {
  auto *runtime = AsHandle(handle);
  if (runtime == nullptr) {
    return;
  }
  ClearRegisteredUserWindow(runtime);
  if (runtime->devDesc != nullptr) {
    aclrtFree(runtime->devDesc);
    runtime->devDesc = nullptr;
  }
  CloseTileXRSymbols(&runtime->tilexr);
  CloseHcclPeerMemSymbols(&runtime->hccl);
  delete runtime;
}

