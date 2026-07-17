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
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#if defined(__linux__)
#include <dlfcn.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res.h>
#include <strings.h>
#include <unistd.h>
#endif

namespace {

constexpr uint32_t kTileXRMaxRanks = TRITON_ASCEND_GIN_MAX_RANKS;
constexpr uint64_t kDefaultTileXRIpcDataOffset = 2ull * 1024ull * 1024ull;
constexpr uint64_t kDefaultTileXRWindowBytes = 100ull * 1024ull * 1024ull;
constexpr uint64_t kDefaultHcclSymWindowMinBytes = 2ull * 1024ull * 1024ull;
constexpr uint64_t kDefaultSignalStride = sizeof(uint64_t);
constexpr uint32_t kDefaultSignalSlots = TRITON_ASCEND_GIN_DEFAULT_SIGNAL_SLOTS;
constexpr uint32_t kDescriptorVersion = 1;
constexpr size_t kAclIpcKeyBytes = 65;
constexpr uint32_t kWindowHandleDefault = 1;
constexpr uint64_t kMaxWindowBaseSearchOffset = 2ull * 1024ull * 1024ull;
constexpr uint64_t kWindowExportGranularity = 4096;
constexpr int kAclRtFeatureNotSupport = 207000;
constexpr int kAclRtIpcExportFailed = 507899;
constexpr int kHcclENotSupport = 5;
constexpr uint64_t kHcclAivInOpbaseBytes = 36ull * 1024ull * 1024ull;
constexpr uint64_t kHcclAivOutOpbaseBytes = 4ull * 1024ull * 1024ull;
constexpr uint64_t kHcclAivCommInfoBytes = 33ull * 1024ull * 1024ull;
constexpr uint64_t kHcclAivTagAddrOffset = 16ull * 1024ull;
constexpr uint64_t kHcclAivFlagAddrOffset = 40ull * 1024ull;
#if defined(__linux__)
constexpr size_t kHcclBufferManagerStorageBytes = 64ull * 1024ull;
constexpr uint32_t kHcclOpExchangeMaxLength = 128;
constexpr uint32_t kHcclOpExchangeTagLength = 160;
constexpr uint32_t kHcclInvalidRankId = 0xffffffffu;
constexpr uint32_t kHcclCmdAllGather = 6;
constexpr uint32_t kHcclOpExecuteAiv = 3;
constexpr uint32_t kHcclReduceReserved = 255;
constexpr uint32_t kHcclDataTypeFp32 = 4;
constexpr uint32_t kHcclMaxNumBlocks = 56;
constexpr uint32_t kHcclAivDevType910B = 2;
constexpr uint32_t kHcclAivDevIdDefault = 16;
constexpr int32_t kHcclAivExecTimeout = 1091;
constexpr uint32_t kHcclAivDirectMaxRanks = 16;
#endif

thread_local std::string g_lastError;

void SetLastError(const std::string &message) {
  g_lastError = message;
}

void SetAclError(const char *what, aclError err) {
  g_lastError = std::string(what) + " failed with acl error " + std::to_string(static_cast<int>(err));
}

const char *HcclSymWinRegisterHint(int ret) {
  if (ret == kHcclENotSupport) {
    return " (HCCL_E_NOT_SUPPORT: HcclCommSymWinRegister requires HCCL symmetric-window "
           "product support, for example Atlas A3; Atlas A2/910B-class packages report this as not supported)";
  }
  return "";
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
using HcclCommSymWinRegisterFn = int (*)(void *, void *, uint64_t, void **, uint32_t);
using HcclCommSymWinDeregisterFn = int (*)(void *);
using HcclCommSymWinGetFn = int (*)(void *, void *, size_t, void **, size_t *);
using HcclSymWinGetPeerPointerFn = int (*)(void *, size_t, uint32_t, void **);
using HcclMemAllocFn = int (*)(void **, uint64_t);
using HcclMemFreeFn = int (*)(void *);
#if defined(__linux__)
using HcclRankGraphGetLayersFn = int (*)(void *, uint32_t **, uint32_t *);
using HcclRankGraphGetLinksFn = int (*)(void *, uint32_t, uint32_t, uint32_t, CommLink **, uint32_t *);
using HcclCommMemRegFn = int (*)(void *, const char *, const CommMem *, HcclMemHandle *);
using HcclChannelAcquireFn = int (*)(void *, CommEngine, const HcclChannelDesc *, uint32_t, ChannelHandle *);
using HcclChannelGetRemoteMemsFn = int (*)(void *, ChannelHandle, uint32_t *, CommMem **, char ***);
using HcclGetHcclBufferFn = int (*)(void *, void **, uint64_t *);
using HcclChannelGetHcclBufferFn = int (*)(void *, ChannelHandle, void **, uint64_t *);
using HcclThreadAcquireFn = int (*)(void *, CommEngine, uint32_t, uint32_t, ThreadHandle *);
using HcclEngineCtxGetFn = int (*)(void *, const char *, CommEngine, void **, uint64_t *);
using HcclEngineCtxCreateFn = int (*)(void *, const char *, CommEngine, uint64_t, void **);
using HcclCommAddExchangeInfoFn = int (*)(void *, const void *, uint32_t);
using HcclGetCommNameFn = int (*)(void *, char *);
#endif
using RtIpcSetMemoryNameFn = int (*)(const void *, uint64_t, char *, uint32_t);
using RtIpcOpenMemoryFn = int (*)(void **, const char *);
using RtIpcCloseMemoryFn = int (*)(const void *);
using RtSetIpcMemPidFn = int (*)(const char *, int32_t *, int32_t);
using RtDeviceGetBareTgidFn = int (*)(uint32_t *);

struct HcclPeerMemSymbols {
  HcclGetRankIdFn getRankId = nullptr;
  HcclGetRankSizeFn getRankSize = nullptr;
  HcclCommSymWinRegisterFn commSymWinRegister = nullptr;
  HcclCommSymWinDeregisterFn commSymWinDeregister = nullptr;
  HcclCommSymWinGetFn commSymWinGet = nullptr;
  HcclSymWinGetPeerPointerFn symWinGetPeerPointer = nullptr;
  void *libraries[4] = {};
  uint32_t libraryCount = 0;
};

#if defined(__linux__)
struct HcclChannelSymbols {
  HcclGetRankIdFn getRankId = nullptr;
  HcclGetRankSizeFn getRankSize = nullptr;
  HcclRankGraphGetLayersFn rankGraphGetLayers = nullptr;
  HcclRankGraphGetLinksFn rankGraphGetLinks = nullptr;
  HcclCommMemRegFn commMemReg = nullptr;
  HcclChannelAcquireFn channelAcquire = nullptr;
  HcclChannelGetRemoteMemsFn channelGetRemoteMems = nullptr;
  HcclGetHcclBufferFn getHcclBuffer = nullptr;
  HcclChannelGetHcclBufferFn channelGetHcclBuffer = nullptr;
  HcclThreadAcquireFn threadAcquire = nullptr;
  HcclEngineCtxGetFn engineCtxGet = nullptr;
  HcclEngineCtxCreateFn engineCtxCreate = nullptr;
  HcclCommAddExchangeInfoFn commAddExchangeInfo = nullptr;
  HcclGetCommNameFn getCommName = nullptr;
  void *libraries[4] = {};
  uint32_t libraryCount = 0;
};

using HcclAlgBufferManagerCtorFn = void (*)(void *);
using HcclAlgBufferManagerDtorFn = void (*)(void *);
using HcclAlgCreateCommAivBufferFn = int (*)(void *, bool);
using HcclAlgCreateCommInfoAivBufferFn = int (*)(void *);
using HcclAlgReleaseCommAivBufferFn = int (*)(void *);
using HcclAlgClearCommAivBufferFn = int (*)(void *);
using HcclAlgGetDeviceMemRefFn = void *(*)(void *);
using HcclAlgGetIndependentOpCclBufferFn = int (*)(void *, void *&, unsigned long &);

enum class HcclAivKernelArgsTypeCompat {
  ARGS_TYPE_SERVER = 0,
  ARGS_TYPE_SUPERPOD = 1,
  ARGS_TYPE_SIMPLE = 2,
  ARGS_TYPE_DEFAULT
};

struct HcclAivOpCounterInfoCompat {
  uint64_t headCountMem = 0;
  uint64_t tailCountMem = 0;
  uint64_t addOneMem = 0;
  uint32_t memSize = 0;
  bool isEnableCounter = false;
};

struct HcclAivOpArgsCompat {
  uint32_t cmdType = 0;
  const void *input = nullptr;
  const void *output = nullptr;
  uint64_t count = 0;
  uint32_t dataType = 0;
  uint32_t op = 0;
  uint32_t root = 0;
  bool isOpBase = false;
};

struct HcclAivTopoArgsCompat {
  uint32_t rank = 0;
  uint32_t rankSize = 0;
  uint32_t devId = 0;
  uint32_t serverId = 0;
  uint32_t serverNum = 0;
  uint32_t devType = 0;
  std::string identify;
};

struct HcclAivResourceArgsCompat {
  std::string commTag;
  aclrtStream stream = nullptr;
  void **buffersIn = nullptr;
  void **buffersOut = nullptr;
  uint64_t bufferSize = 0;
  uint32_t numBlocks = 0;
  int32_t aivTag = 0;
};

struct HcclAivAlgArgsCompat {
  int32_t step = -1;
  bool isSmallCount = false;
  uint32_t deterministic = 0;
  HcclAivKernelArgsTypeCompat argsType =
      HcclAivKernelArgsTypeCompat::ARGS_TYPE_SERVER;
  int32_t execTimeOut = kHcclAivExecTimeout;
  bool execTimeOutSet = false;
  bool isNpuDirectRoce = false;
  uint64_t rmaInfo = 0;
};

struct HcclAivProfilingInfoCompat {
  uint64_t beginTime = 0;
  HcclAivOpCounterInfoCompat counter = {};
};

using HcclAivRegisterKernelFn = int (*)(uint32_t);
using HcclAivClearSyncBufFn = int (*)(void **, const HcclAivResourceArgsCompat &,
                                      const HcclAivTopoArgsCompat &,
                                      HcclAivAlgArgsCompat);
using HcclAivExecuteKernelLaunchFn = int (*)(
    const HcclAivOpArgsCompat &, const HcclAivTopoArgsCompat &,
    const HcclAivResourceArgsCompat &, const HcclAivAlgArgsCompat &,
    HcclAivProfilingInfoCompat &);
using HcclMemNameGetInstanceFn = void *(*)(int32_t);
using HcclMemNameSetIpcMemFn = int (*)(void *, void *, unsigned long long,
                                       uint8_t *, uint32_t,
                                       unsigned long long &, int32_t,
                                       int32_t, bool);
using HcclMemNameOpenIpcMemFn = int (*)(void *, void **, unsigned long long,
                                        const uint8_t *, uint32_t,
                                        unsigned long long, bool &, bool);

struct HcclAlgBufferSymbols {
  HcclAlgBufferManagerCtorFn ctor = nullptr;
  HcclAlgBufferManagerDtorFn dtor = nullptr;
  HcclAlgCreateCommAivBufferFn createCommAivBuffer = nullptr;
  HcclAlgCreateCommInfoAivBufferFn createCommInfoAivBuffer = nullptr;
  HcclAlgReleaseCommAivBufferFn releaseCommAivBuffer = nullptr;
  HcclAlgClearCommAivBufferFn clearCommAivBuffer = nullptr;
  HcclAlgGetDeviceMemRefFn getInAivOpbaseBuffer = nullptr;
  HcclAlgGetDeviceMemRefFn getOutAivOpbaseBuffer = nullptr;
  HcclAlgGetDeviceMemRefFn getAivCommInfoBuffer = nullptr;
  HcclAlgGetIndependentOpCclBufferFn getIndependentOpCclBuffer = nullptr;
  HcclAivRegisterKernelFn registerKernel = nullptr;
  HcclAivClearSyncBufFn clearAivSyncBuf = nullptr;
  HcclAivExecuteKernelLaunchFn executeKernelLaunch = nullptr;
  void *libraries[4] = {};
  uint32_t libraryCount = 0;
};

struct HcclPlfIpcSymbols {
  HcclMemNameGetInstanceFn memNameGetInstance = nullptr;
  HcclMemNameSetIpcMemFn memNameSetIpcMem = nullptr;
  HcclMemNameOpenIpcMemFn memNameOpenIpcMem = nullptr;
  void *libraries[4] = {};
  uint32_t libraryCount = 0;
};

struct HcclOpExchangeInfoCompat {
  uint64_t cclBufferSize = 0;
  uint32_t root = kHcclInvalidRankId;
  uint32_t opType = 0;
  uint32_t opExecuteConfig = 0;
  uint32_t reduceType = kHcclReduceReserved;
  uint32_t dataType = kHcclDataTypeFp32;
  uint64_t count = 0;
  uint32_t aivCoreLimit = kHcclMaxNumBlocks;
  char group[kHcclOpExchangeMaxLength] = {};
  char tag[kHcclOpExchangeTagLength] = {};
};

static_assert(sizeof(HcclOpExchangeInfoCompat) == 336,
              "HCCL OpExchangeInfo compatible layout changed");
#endif

struct HcclAllocatorSymbols {
  HcclMemAllocFn memAlloc = nullptr;
  HcclMemFreeFn memFree = nullptr;
  void *libraries[4] = {};
  uint32_t libraryCount = 0;
  bool resolved = false;
};

struct RtsIpcSymbols {
  RtIpcSetMemoryNameFn setMemoryName = nullptr;
  RtIpcOpenMemoryFn openMemory = nullptr;
  RtIpcCloseMemoryFn closeMemory = nullptr;
  RtSetIpcMemPidFn setMemPid = nullptr;
  RtDeviceGetBareTgidFn getBareTgid = nullptr;
  void *library = nullptr;
  bool resolved = false;
};

std::mutex g_hcclAllocatorMutex;
HcclAllocatorSymbols g_hcclAllocatorSymbols;
std::mutex g_rtsIpcMutex;
RtsIpcSymbols g_rtsIpcSymbols;

struct HcclVmmAllocation {
  aclrtDrvMemHandle handle = nullptr;
  uint64_t bytes = 0;
  bool hcclOwned = false;
};

std::mutex g_hcclVmmAllocationMutex;
std::unordered_map<void *, HcclVmmAllocation> g_hcclVmmAllocations;

struct TritonAscendGinRuntimeHandle {
  void *tilexrComm = nullptr;
  TileXRSymbols tilexr;
  void *hcclComm = nullptr;
  HcclPeerMemSymbols hccl;
#if defined(__linux__)
  HcclChannelSymbols hcclChannel;
#endif
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
  bool userWindowRtIpc[kTileXRMaxRanks] = {};
  void *hcclDataSymWindow = nullptr;
  void *hcclSignalSymWindow = nullptr;
#if defined(__linux__)
  CommEngine hcclChannelEngine = COMM_ENGINE_AIV;
  HcclMemHandle hcclWindowMemHandle = nullptr;
  HcclMemHandle hcclSignalMemHandle = nullptr;
  void *hcclAivCommInfo = nullptr;
  HcclMemHandle hcclAivCommInfoMemHandle = nullptr;
  bool hcclAivCommInfoRegistered = false;
  ChannelHandle hcclWindowChannels[kTileXRMaxRanks] = {};
  ChannelHandle hcclSignalChannels[kTileXRMaxRanks] = {};
  HcclAlgBufferSymbols hcclAlgBuffer;
  HcclPlfIpcSymbols hcclPlfIpc;
  void *hcclAivBufferManagerStorage = nullptr;
  bool hcclAivBufferManagerConstructed = false;
  bool hcclAivOpbaseBuffersCreated = false;
  uint64_t hcclInAivOpbasePtr = 0;
  uint64_t hcclInAivOpbaseBytes = 0;
  uint64_t hcclOutAivOpbasePtr = 0;
  uint64_t hcclOutAivOpbaseBytes = 0;
  int32_t hcclAivDirectTag = 0;
#endif
};

struct WindowRendezvousInfo {
  std::string key;
  uint64_t windowOffset = 0;
  uint64_t bytes = 0;
  uint32_t pid = 0;
  bool rtIpc = false;
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
         symbols.commSymWinRegister != nullptr && symbols.commSymWinDeregister != nullptr &&
         symbols.commSymWinGet != nullptr;
}

void ResolveHcclSymbolsFromHandle(HcclPeerMemSymbols *symbols, void *library) {
  ResolveOne(&symbols->getRankId, library, "HcclGetRankId");
  ResolveOne(&symbols->getRankSize, library, "HcclGetRankSize");
  ResolveOne(&symbols->commSymWinRegister, library, "HcclCommSymWinRegister");
  ResolveOne(&symbols->commSymWinDeregister, library, "HcclCommSymWinDeregister");
  ResolveOne(&symbols->commSymWinGet, library, "HcclCommSymWinGet");
  ResolveOne(&symbols->symWinGetPeerPointer, library, "HcclSymWinGetPeerPointer");
}

#if defined(__linux__)
bool HcclChannelSymbolsComplete(const HcclChannelSymbols &symbols) {
  return symbols.getRankId != nullptr && symbols.getRankSize != nullptr &&
         symbols.rankGraphGetLayers != nullptr && symbols.rankGraphGetLinks != nullptr &&
         symbols.commMemReg != nullptr && symbols.channelAcquire != nullptr &&
         symbols.channelGetRemoteMems != nullptr && symbols.getHcclBuffer != nullptr &&
         symbols.channelGetHcclBuffer != nullptr && symbols.engineCtxGet != nullptr &&
         symbols.engineCtxCreate != nullptr;
}

void ResolveHcclChannelSymbolsFromHandle(HcclChannelSymbols *symbols, void *library) {
  ResolveOne(&symbols->getRankId, library, "HcclGetRankId");
  ResolveOne(&symbols->getRankSize, library, "HcclGetRankSize");
  ResolveOne(&symbols->rankGraphGetLayers, library, "HcclRankGraphGetLayers");
  ResolveOne(&symbols->rankGraphGetLinks, library, "HcclRankGraphGetLinks");
  ResolveOne(&symbols->commMemReg, library, "HcclCommMemReg");
  ResolveOne(&symbols->channelAcquire, library, "HcclChannelAcquire");
  ResolveOne(&symbols->channelGetRemoteMems, library, "HcclChannelGetRemoteMems");
  ResolveOne(&symbols->getHcclBuffer, library, "HcclGetHcclBuffer");
  ResolveOne(&symbols->channelGetHcclBuffer, library, "HcclChannelGetHcclBuffer");
  ResolveOne(&symbols->threadAcquire, library, "HcclThreadAcquire");
  ResolveOne(&symbols->engineCtxGet, library, "HcclEngineCtxGet");
  ResolveOne(&symbols->engineCtxCreate, library, "HcclEngineCtxCreate");
  ResolveOne(&symbols->commAddExchangeInfo, library, "HcclCommAddExchangeInfo");
  ResolveOne(&symbols->getCommName, library, "HcclGetCommName");
}

bool HcclAlgBufferSymbolsComplete(const HcclAlgBufferSymbols &symbols) {
  return symbols.ctor != nullptr && symbols.dtor != nullptr &&
         symbols.createCommAivBuffer != nullptr &&
         symbols.createCommInfoAivBuffer != nullptr &&
         symbols.releaseCommAivBuffer != nullptr &&
         symbols.clearCommAivBuffer != nullptr &&
         symbols.getInAivOpbaseBuffer != nullptr &&
         symbols.getOutAivOpbaseBuffer != nullptr &&
         symbols.getAivCommInfoBuffer != nullptr &&
         symbols.getIndependentOpCclBuffer != nullptr;
}

bool HcclAivDirectSymbolsComplete(const HcclAlgBufferSymbols &symbols) {
  return HcclAlgBufferSymbolsComplete(symbols) &&
         symbols.registerKernel != nullptr &&
         symbols.clearAivSyncBuf != nullptr &&
         symbols.executeKernelLaunch != nullptr;
}

bool HcclPlfIpcSymbolsComplete(const HcclPlfIpcSymbols &symbols) {
  return symbols.memNameGetInstance != nullptr &&
         symbols.memNameSetIpcMem != nullptr &&
         symbols.memNameOpenIpcMem != nullptr;
}

void ResolveHcclAlgBufferSymbolsFromHandle(HcclAlgBufferSymbols *symbols, void *library) {
  ResolveOne(&symbols->ctor, library, "_ZN4hccl16CCLBufferManagerC1Ev");
  ResolveOne(&symbols->dtor, library, "_ZN4hccl16CCLBufferManagerD1Ev");
  ResolveOne(&symbols->createCommAivBuffer, library,
             "_ZN4hccl16CCLBufferManager19CreateCommAIVbufferEb");
  ResolveOne(&symbols->createCommInfoAivBuffer, library,
             "_ZN4hccl16CCLBufferManager23CreateCommInfoAIVbufferEv");
  ResolveOne(&symbols->releaseCommAivBuffer, library,
             "_ZN4hccl16CCLBufferManager20ReleaseCommAIVbufferEv");
  ResolveOne(&symbols->clearCommAivBuffer, library,
             "_ZN4hccl16CCLBufferManager18ClearCommAIVbufferEv");
  ResolveOne(&symbols->getInAivOpbaseBuffer, library,
             "_ZN4hccl16CCLBufferManager20GetInAivOpbaseBufferEv");
  ResolveOne(&symbols->getOutAivOpbaseBuffer, library,
             "_ZN4hccl16CCLBufferManager21GetOutAivOpbaseBufferEv");
  ResolveOne(&symbols->getAivCommInfoBuffer, library,
             "_ZN4hccl16CCLBufferManager20GetAivCommInfoBufferEv");
  ResolveOne(&symbols->getIndependentOpCclBuffer, library,
             "_ZN4hccl16CCLBufferManager25GetIndependentOpCCLbufferERPvRm");
  ResolveOne(&symbols->registerKernel, library, "_ZN4hccl14RegisterKernelE7DevType");
  ResolveOne(&symbols->clearAivSyncBuf, library,
             "_ZN4hccl15ClearAivSyncBufEPPvRKNS_15AivResourceArgsERKNS_11AivTopoArgsENS_10AivAlgArgsE");
  ResolveOne(&symbols->executeKernelLaunch, library,
             "_ZN4hccl19ExecuteKernelLaunchERKNS_9AivOpArgsERKNS_11AivTopoArgsERKNS_15AivResourceArgsERKNS_10AivAlgArgsERNS_16AivProfilingInfoE");
}

void ResolveHcclPlfIpcSymbolsFromHandle(HcclPlfIpcSymbols *symbols, void *library) {
  ResolveOne(&symbols->memNameGetInstance, library,
             "_ZN4hccl17MemNameRepository11GetInstanceEi");
  ResolveOne(&symbols->memNameSetIpcMem, library,
             "_ZN4hccl17MemNameRepository9SetIpcMemEPvyPhjRyiib");
  ResolveOne(&symbols->memNameOpenIpcMem, library,
             "_ZN4hccl17MemNameRepository10OpenIpcMemEPPvyPKhjyRbb");
}
#endif

void ResolveHcclAllocatorSymbolsFromHandle(HcclAllocatorSymbols *symbols, void *library) {
  ResolveOne(&symbols->memAlloc, library, "HcclMemAlloc");
  ResolveOne(&symbols->memFree, library, "HcclMemFree");
}

bool HcclAllocatorSymbolsComplete(const HcclAllocatorSymbols &symbols) {
  return symbols.memAlloc != nullptr && symbols.memFree != nullptr;
}

bool RtsIpcSymbolsComplete(const RtsIpcSymbols &symbols) {
  return symbols.setMemoryName != nullptr && symbols.openMemory != nullptr &&
         symbols.closeMemory != nullptr && symbols.setMemPid != nullptr &&
         symbols.getBareTgid != nullptr;
}

void ResolveRtsIpcSymbolsFromHandle(RtsIpcSymbols *symbols, void *library) {
  ResolveOne(&symbols->setMemoryName, library, "rtIpcSetMemoryName");
  ResolveOne(&symbols->openMemory, library, "rtIpcOpenMemory");
  ResolveOne(&symbols->closeMemory, library, "rtIpcCloseMemory");
  ResolveOne(&symbols->setMemPid, library, "rtSetIpcMemPid");
  ResolveOne(&symbols->getBareTgid, library, "rtDeviceGetBareTgid");
}

bool EnsureRtsIpcSymbolsLocked() {
  RtsIpcSymbols &symbols = g_rtsIpcSymbols;
  if (symbols.resolved && RtsIpcSymbolsComplete(symbols)) {
    return true;
  }

  symbols.setMemoryName =
      reinterpret_cast<RtIpcSetMemoryNameFn>(ResolveDefaultSymbol("rtIpcSetMemoryName"));
  symbols.openMemory =
      reinterpret_cast<RtIpcOpenMemoryFn>(ResolveDefaultSymbol("rtIpcOpenMemory"));
  symbols.closeMemory =
      reinterpret_cast<RtIpcCloseMemoryFn>(ResolveDefaultSymbol("rtIpcCloseMemory"));
  symbols.setMemPid =
      reinterpret_cast<RtSetIpcMemPidFn>(ResolveDefaultSymbol("rtSetIpcMemPid"));
  symbols.getBareTgid =
      reinterpret_cast<RtDeviceGetBareTgidFn>(ResolveDefaultSymbol("rtDeviceGetBareTgid"));
  if (RtsIpcSymbolsComplete(symbols)) {
    symbols.resolved = true;
    return true;
  }

  void *library = dlopen("libruntime.so", RTLD_NOW | RTLD_LOCAL);
  if (library != nullptr) {
    symbols.library = library;
    ResolveRtsIpcSymbolsFromHandle(&symbols, library);
    if (RtsIpcSymbolsComplete(symbols)) {
      symbols.resolved = true;
      return true;
    }
  }

  symbols.resolved = true;
  const char *err = dlerror();
  SetLastError(std::string("failed to resolve RTS IPC symbols") +
               (err == nullptr ? "" : std::string(": ") + err));
  return false;
}

bool EnsureRtsIpcSymbols() {
  std::lock_guard<std::mutex> lock(g_rtsIpcMutex);
  return EnsureRtsIpcSymbolsLocked();
}

bool EnsureHcclAllocatorSymbolsLocked() {
  HcclAllocatorSymbols &symbols = g_hcclAllocatorSymbols;
  if (symbols.resolved && HcclAllocatorSymbolsComplete(symbols)) {
    return true;
  }

  symbols.memAlloc = reinterpret_cast<HcclMemAllocFn>(ResolveDefaultSymbol("HcclMemAlloc"));
  symbols.memFree = reinterpret_cast<HcclMemFreeFn>(ResolveDefaultSymbol("HcclMemFree"));
  if (HcclAllocatorSymbolsComplete(symbols)) {
    symbols.resolved = true;
    return true;
  }

  const char *envPath = std::getenv("TRITON_ASCEND_HCCL_LIB");
  const char *candidates[3] = {
      envPath,
      "libhcomm.so",
      "libhccl.so",
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
    if (symbols.libraryCount < 4) {
      symbols.libraries[symbols.libraryCount++] = library;
    }
    ResolveHcclAllocatorSymbolsFromHandle(&symbols, library);
    if (HcclAllocatorSymbolsComplete(symbols)) {
      symbols.resolved = true;
      return true;
    }
  }

  symbols.resolved = true;
  SetLastError("failed to resolve HCCL allocator symbols (HcclMemAlloc, HcclMemFree). " + dlErrors);
  return false;
}

bool EnsureHcclAllocatorSymbols() {
  std::lock_guard<std::mutex> lock(g_hcclAllocatorMutex);
  return EnsureHcclAllocatorSymbolsLocked();
}

void ResolveHcclSymbolsFromDefault(HcclPeerMemSymbols *symbols) {
  if (symbols == nullptr) {
    return;
  }
  symbols->getRankId = reinterpret_cast<HcclGetRankIdFn>(ResolveDefaultSymbol("HcclGetRankId"));
  symbols->getRankSize = reinterpret_cast<HcclGetRankSizeFn>(ResolveDefaultSymbol("HcclGetRankSize"));
  symbols->commSymWinRegister =
      reinterpret_cast<HcclCommSymWinRegisterFn>(ResolveDefaultSymbol("HcclCommSymWinRegister"));
  symbols->commSymWinDeregister =
      reinterpret_cast<HcclCommSymWinDeregisterFn>(ResolveDefaultSymbol("HcclCommSymWinDeregister"));
  symbols->commSymWinGet =
      reinterpret_cast<HcclCommSymWinGetFn>(ResolveDefaultSymbol("HcclCommSymWinGet"));
  symbols->symWinGetPeerPointer =
      reinterpret_cast<HcclSymWinGetPeerPointerFn>(ResolveDefaultSymbol("HcclSymWinGetPeerPointer"));
}

#if defined(__linux__)
void ResolveHcclChannelSymbolsFromDefault(HcclChannelSymbols *symbols) {
  if (symbols == nullptr) {
    return;
  }
  symbols->getRankId = reinterpret_cast<HcclGetRankIdFn>(ResolveDefaultSymbol("HcclGetRankId"));
  symbols->getRankSize = reinterpret_cast<HcclGetRankSizeFn>(ResolveDefaultSymbol("HcclGetRankSize"));
  symbols->rankGraphGetLayers =
      reinterpret_cast<HcclRankGraphGetLayersFn>(ResolveDefaultSymbol("HcclRankGraphGetLayers"));
  symbols->rankGraphGetLinks =
      reinterpret_cast<HcclRankGraphGetLinksFn>(ResolveDefaultSymbol("HcclRankGraphGetLinks"));
  symbols->commMemReg = reinterpret_cast<HcclCommMemRegFn>(ResolveDefaultSymbol("HcclCommMemReg"));
  symbols->channelAcquire =
      reinterpret_cast<HcclChannelAcquireFn>(ResolveDefaultSymbol("HcclChannelAcquire"));
  symbols->channelGetRemoteMems =
      reinterpret_cast<HcclChannelGetRemoteMemsFn>(ResolveDefaultSymbol("HcclChannelGetRemoteMems"));
  symbols->getHcclBuffer = reinterpret_cast<HcclGetHcclBufferFn>(ResolveDefaultSymbol("HcclGetHcclBuffer"));
  symbols->channelGetHcclBuffer =
      reinterpret_cast<HcclChannelGetHcclBufferFn>(ResolveDefaultSymbol("HcclChannelGetHcclBuffer"));
  symbols->threadAcquire = reinterpret_cast<HcclThreadAcquireFn>(ResolveDefaultSymbol("HcclThreadAcquire"));
  symbols->engineCtxGet =
      reinterpret_cast<HcclEngineCtxGetFn>(ResolveDefaultSymbol("HcclEngineCtxGet"));
  symbols->engineCtxCreate =
      reinterpret_cast<HcclEngineCtxCreateFn>(ResolveDefaultSymbol("HcclEngineCtxCreate"));
  symbols->commAddExchangeInfo =
      reinterpret_cast<HcclCommAddExchangeInfoFn>(ResolveDefaultSymbol("HcclCommAddExchangeInfo"));
  symbols->getCommName = reinterpret_cast<HcclGetCommNameFn>(ResolveDefaultSymbol("HcclGetCommName"));
}

int LoadHcclChannelSymbols(const char *libraryPath, HcclChannelSymbols *symbols) {
  if (symbols == nullptr) {
    SetLastError("HCCL channel symbol output is null");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  ResolveHcclChannelSymbolsFromDefault(symbols);
  if (HcclChannelSymbolsComplete(*symbols) &&
      symbols->commAddExchangeInfo != nullptr && symbols->getCommName != nullptr) {
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }

  const char *envPath = std::getenv("TRITON_ASCEND_HCCL_LIB");
  const char *candidates[4] = {
      libraryPath,
      envPath,
      "libhcomm.so",
      "libhccl.so",
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
    if (symbols->libraryCount < 4) {
      symbols->libraries[symbols->libraryCount++] = library;
    }
    ResolveHcclChannelSymbolsFromHandle(symbols, library);
    if (HcclChannelSymbolsComplete(*symbols) &&
        symbols->commAddExchangeInfo != nullptr && symbols->getCommName != nullptr) {
      return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
    }
  }

  if (HcclChannelSymbolsComplete(*symbols)) {
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }

  SetLastError("failed to resolve required HCCL channel symbols "
               "(HcclGetRankId, HcclGetRankSize, HcclRankGraphGetLayers, "
               "HcclRankGraphGetLinks, HcclCommMemReg, HcclChannelAcquire, "
               "HcclChannelGetRemoteMems, HcclGetHcclBuffer, "
               "HcclChannelGetHcclBuffer, HcclEngineCtxGet, "
               "HcclEngineCtxCreate). " +
               dlErrors);
  return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
}

void CloseHcclChannelSymbols(HcclChannelSymbols *symbols) {
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

int LoadHcclAlgBufferSymbols(const char *libraryPath, HcclAlgBufferSymbols *symbols) {
  if (symbols == nullptr) {
    SetLastError("HCCL alg buffer symbol output is null");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  ResolveHcclAlgBufferSymbolsFromHandle(symbols, RTLD_DEFAULT);
  if (HcclAlgBufferSymbolsComplete(*symbols)) {
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }

  const char *envPath = std::getenv("TRITON_ASCEND_HCCL_ALG_LIB");
  const char *candidates[3] = {
      libraryPath,
      envPath,
      "libhccl_alg.so",
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
    if (symbols->libraryCount < 4) {
      symbols->libraries[symbols->libraryCount++] = library;
    }
    ResolveHcclAlgBufferSymbolsFromHandle(symbols, library);
    if (HcclAlgBufferSymbolsComplete(*symbols)) {
      return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
    }
  }

  SetLastError("failed to resolve required HCCL alg CCLBufferManager symbols "
               "(CreateCommAIVbuffer, CreateCommInfoAIVbuffer, "
               "GetIndependentOpCCLbuffer, GetIn/OutAivOpbaseBuffer, "
               "GetAivCommInfoBuffer). These are internal libhccl_alg.so "
               "C++ ABI symbols. " +
               dlErrors);
  return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
}

void CloseHcclAlgBufferSymbols(HcclAlgBufferSymbols *symbols) {
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

int LoadHcclPlfIpcSymbols(const char *libraryPath, HcclPlfIpcSymbols *symbols) {
  if (symbols == nullptr) {
    SetLastError("HCCL PLF IPC symbol output is null");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  ResolveHcclPlfIpcSymbolsFromHandle(symbols, RTLD_DEFAULT);
  if (HcclPlfIpcSymbolsComplete(*symbols)) {
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }

  const char *envPath = std::getenv("TRITON_ASCEND_HCCL_PLF_LIB");
  const char *candidates[3] = {
      libraryPath,
      envPath,
      "libhccl_plf.so",
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
    if (symbols->libraryCount < 4) {
      symbols->libraries[symbols->libraryCount++] = library;
    }
    ResolveHcclPlfIpcSymbolsFromHandle(symbols, library);
    if (HcclPlfIpcSymbolsComplete(*symbols)) {
      return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
    }
  }

  SetLastError("failed to resolve HCCL PLF MemNameRepository symbols. " +
               dlErrors);
  return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
}

void CloseHcclPlfIpcSymbols(HcclPlfIpcSymbols *symbols) {
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
#endif

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

  SetLastError("failed to resolve required HCCL peer-memory symbols "
               "(HcclGetRankId, HcclGetRankSize, HcclCommSymWinRegister, "
               "HcclCommSymWinDeregister, HcclCommSymWinGet). "
               "HcclSymWinGetPeerPointer is optional and was not required for loading. " +
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

uint64_t GetEnvU64(const char *name, uint64_t fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char *end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 0);
  if (end == value) {
    return fallback;
  }
  return static_cast<uint64_t>(parsed);
}

uint32_t GetEnvU32(const char *name, uint32_t fallback) {
  uint64_t parsed = GetEnvU64(name, fallback);
  if (parsed > std::numeric_limits<uint32_t>::max()) {
    return fallback;
  }
  return static_cast<uint32_t>(parsed);
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

std::string WindowPidReadyPath(const std::string &dir, const std::string &id, uint32_t rank) {
  std::ostringstream os;
  os << dir << "/triton_ascend_gin_window_" << id << "_rank" << rank << ".pidready";
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
  *value = std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
  return !value->empty();
}

bool ParseWindowRendezvousInfo(const std::string &raw, WindowRendezvousInfo *info) {
  if (info == nullptr) {
    return false;
  }
  std::istringstream in(raw);
  WindowRendezvousInfo next = {};
  std::string first;
  if (!(in >> first)) {
    return false;
  }
  if (first == "rt" || first == "acl") {
    next.rtIpc = first == "rt";
    if (!(in >> next.key)) {
      return false;
    }
  } else {
    next.key = first;
  }
  if (next.key.empty()) {
    return false;
  }
  if (!(in >> next.windowOffset)) {
    next.windowOffset = 0;
  }
  if (!(in >> next.bytes)) {
    next.bytes = 0;
  }
  if (next.rtIpc && !(in >> next.pid)) {
    return false;
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
    uint64_t pageRoundedBytes = RoundUp(neededBytes, kDefaultTileXRIpcDataOffset);
    uint64_t byteCandidates[] = {roundedBytes, pageRoundedBytes, neededBytes};
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
    }
  }

  return lastErr;
}

int ExportUserWindowRtIpc(void *localPtr, uint64_t bytes, char *key, size_t keyBytes,
                          uint64_t *windowOffset, uint64_t *exportBytes, uint32_t *pid) {
  if (localPtr == nullptr || key == nullptr || keyBytes == 0 || windowOffset == nullptr ||
      exportBytes == nullptr || pid == nullptr) {
    SetLastError("ExportUserWindowRtIpc received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (!EnsureRtsIpcSymbols()) {
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }

  const uintptr_t ptr = reinterpret_cast<uintptr_t>(localPtr);
  const uint64_t alignments[] = {0, 4096, 65536, 2ull * 1024ull * 1024ull};
  uintptr_t tried[sizeof(alignments) / sizeof(alignments[0])] = {};
  size_t triedCount = 0;
  int lastRet = -1;

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
    uint64_t pageRoundedBytes = RoundUp(neededBytes, kDefaultTileXRIpcDataOffset);
    uint64_t byteCandidates[] = {roundedBytes, pageRoundedBytes, neededBytes};
    for (uint64_t candidateBytes : byteCandidates) {
      if (candidateBytes < neededBytes) {
        continue;
      }
      char candidateKey[kAclIpcKeyBytes] = {};
      lastRet = g_rtsIpcSymbols.setMemoryName(reinterpret_cast<void *>(base),
                                              candidateBytes, candidateKey,
                                              static_cast<uint32_t>(keyBytes));
      if (DebugEnabled()) {
        std::fprintf(stderr,
                     "[triton_ascend_gin] rt_ipc_export try ptr=%p base=%p bytes=%llu "
                     "offset=%llu candidate=%llu ret=%d key=%s\n",
                     localPtr, reinterpret_cast<void *>(base),
                     static_cast<unsigned long long>(bytes),
                     static_cast<unsigned long long>(offset),
                     static_cast<unsigned long long>(candidateBytes),
                     lastRet, candidateKey);
      }
      if (lastRet == 0) {
        std::memset(key, 0, keyBytes);
        std::memcpy(key, candidateKey,
                    keyBytes < kAclIpcKeyBytes ? keyBytes : kAclIpcKeyBytes);
        *windowOffset = offset;
        *exportBytes = candidateBytes;
        int ret = g_rtsIpcSymbols.getBareTgid(pid);
        if (ret != 0 || *pid == 0) {
          SetLastError("rtDeviceGetBareTgid failed with rt error " + std::to_string(ret));
          return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
        }
        return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
      }
    }
  }

  if (lastRet != 0) {
    SetLastError("rtIpcSetMemoryName user window failed with rt error " +
                 std::to_string(lastRet));
    return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
  }
  SetLastError("rtIpcSetMemoryName user window did not produce an exportable base");
  return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
}

int SetRtIpcWindowPeerPids(const char *key, const WindowRendezvousInfo *infos,
                           uint32_t nranks, uint32_t rank) {
  if (key == nullptr || infos == nullptr) {
    SetLastError("SetRtIpcWindowPeerPids received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (!EnsureRtsIpcSymbols()) {
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }
  for (uint32_t peer = 0; peer < nranks; ++peer) {
    if (peer == rank) {
      continue;
    }
    int32_t pid = static_cast<int32_t>(infos[peer].pid);
    int ret = g_rtsIpcSymbols.setMemPid(key, &pid, 1);
    if (ret != 0) {
      SetLastError("rtSetIpcMemPid user window peer " + std::to_string(peer) +
                   " failed with rt error " + std::to_string(ret));
      return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
    }
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

void ClearRegisteredUserWindow(TritonAscendGinRuntimeHandle *handle) {
  if (handle == nullptr) {
    return;
  }
  for (uint32_t rank = 0; rank < handle->hostDesc.nranks && rank < kTileXRMaxRanks; ++rank) {
    if (handle->userWindowImported[rank] && handle->userWindowRtIpc[rank] &&
        handle->userWindowPtrs[rank] != nullptr) {
      if (EnsureRtsIpcSymbols()) {
        (void)g_rtsIpcSymbols.closeMemory(handle->userWindowPtrs[rank]);
      }
    } else if (handle->userWindowImported[rank] && !handle->userWindowKeys[rank].empty()) {
      (void)aclrtIpcMemClose(handle->userWindowKeys[rank].c_str());
    }
    handle->userWindowKeys[rank].clear();
    handle->userWindowPtrs[rank] = nullptr;
    handle->userWindowImported[rank] = false;
    handle->userWindowRtIpc[rank] = false;
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

int DeregisterHcclSymWindow(TritonAscendGinRuntimeHandle *runtime, void **symWindow, const char *what) {
  if (runtime == nullptr || symWindow == nullptr || *symWindow == nullptr) {
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }
  if (runtime->hccl.commSymWinDeregister == nullptr) {
    *symWindow = nullptr;
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }
  int ret = runtime->hccl.commSymWinDeregister(*symWindow);
  if (ret != 0) {
    SetLastError(std::string("HcclCommSymWinDeregister for ") + what + " failed with code " +
                 std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  *symWindow = nullptr;
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

uint64_t HcclSymWindowRegisterBytes(uint64_t bytes) {
  const int configuredMin =
      GetEnvInt("TRITON_ASCEND_GIN_HCCL_SYM_WIN_MIN_BYTES",
                static_cast<int>(kDefaultHcclSymWindowMinBytes));
  const uint64_t minBytes = configuredMin <= 0 ? 0 : static_cast<uint64_t>(configuredMin);
  if (minBytes == 0) {
    return bytes;
  }
  uint64_t rounded = bytes < minBytes ? minBytes : bytes;
  const uint64_t remainder = rounded % minBytes;
  if (remainder == 0) {
    return rounded;
  }
  const uint64_t padding = minBytes - remainder;
  if (rounded > std::numeric_limits<uint64_t>::max() - padding) {
    return rounded;
  }
  return rounded + padding;
}

uint64_t AlignUpU64(uint64_t value, uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const uint64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return value + (alignment - remainder);
}

uint64_t HcclSignalWindowBytes(const TritonAscendGinRuntimeHandle *runtime) {
  if (runtime == nullptr || runtime->hostDesc.nranks == 0) {
    return 0;
  }
  const uint64_t slots = runtime->signalSlots == 0 ? kDefaultSignalSlots : runtime->signalSlots;
  const uint64_t stride = runtime->signalStride == 0 ? kDefaultSignalStride : runtime->signalStride;
  return static_cast<uint64_t>(runtime->hostDesc.nranks) * slots * stride;
}

#if defined(__linux__)
bool HcclChannelEngineUsesHcommThread(CommEngine engine) {
  return engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS;
}

CommEngine HcclChannelThreadEngine(CommEngine engine) {
  if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
    return COMM_ENGINE_AICPU_TS;
  }
  return engine;
}

int FillDescriptorFromHcclChannel(TritonAscendGinRuntimeHandle *handle) {
  if (handle == nullptr || handle->hcclComm == nullptr) {
    SetLastError("HCCL channel runtime handle is null");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (!HcclChannelSymbolsComplete(handle->hcclChannel)) {
    SetLastError("HCCL channel symbols are unavailable");
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }

  uint32_t rank = 0;
  uint32_t nranks = 0;
  int ret = handle->hcclChannel.getRankId(handle->hcclComm, &rank);
  if (ret != 0) {
    SetLastError("HcclGetRankId failed with code " + std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  ret = handle->hcclChannel.getRankSize(handle->hcclComm, &nranks);
  if (ret != 0) {
    SetLastError("HcclGetRankSize failed with code " + std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  if (nranks == 0 || nranks > kTileXRMaxRanks || rank >= nranks) {
    SetLastError("HCCL channel rank metadata is invalid");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  ThreadHandle thread = handle->hostDesc.hccl_thread_handle;
  if (thread == 0 && HcclChannelEngineUsesHcommThread(handle->hcclChannelEngine)) {
    if (handle->hcclChannel.threadAcquire == nullptr) {
      SetLastError("HcclThreadAcquire symbol was not found");
      return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
    }
    const CommEngine threadEngine = HcclChannelThreadEngine(handle->hcclChannelEngine);
    ret = handle->hcclChannel.threadAcquire(handle->hcclComm, threadEngine, 1, 1, &thread);
    if (ret != 0 || thread == 0) {
      SetLastError("HcclThreadAcquire failed with code " + std::to_string(ret) +
                   " engine=" + std::to_string(static_cast<int>(threadEngine)) +
                   " thread=" + std::to_string(static_cast<unsigned long long>(thread)));
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }
    if (DebugEnabled()) {
      std::fprintf(stderr,
                   "[triton_ascend_gin] HcclThreadAcquire rank=%u engine=%d "
                   "thread=0x%llx\n",
                   rank, static_cast<int>(threadEngine),
                   static_cast<unsigned long long>(thread));
    }
  }

  TritonAscendGinDev next = handle->hostDesc;
  next.version = kDescriptorVersion;
  next.backend_kind = TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_CHANNEL;
  next.rank = rank;
  next.nranks = nranks;
  next.window_bytes = handle->windowBytes;
  next.signal_stride = handle->signalStride;
  next.signal_slots = handle->signalSlots;
  next.hccl_thread_handle = thread;
  handle->hostDesc = next;
  return CopyDescriptorToDevice(handle);
}

bool HcclLinkProtocolIsValid(CommProtocol protocol) {
  return protocol == COMM_PROTOCOL_HCCS || protocol == COMM_PROTOCOL_ROCE ||
         protocol == COMM_PROTOCOL_PCIE || protocol == COMM_PROTOCOL_SIO ||
         protocol == COMM_PROTOCOL_UBC_CTP || protocol == COMM_PROTOCOL_UBC_TP ||
         protocol == COMM_PROTOCOL_UB_MEM;
}

bool HcclAivAllowExperimentalHccs() {
  const char *value = std::getenv("TRITON_ASCEND_GIN_HCCL_AIV_ALLOW_HCCS");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool HcclLinkProtocolAllowedForEngine(CommEngine engine, CommProtocol protocol) {
  if (!HcclLinkProtocolIsValid(protocol)) {
    return false;
  }
  if (engine == COMM_ENGINE_AIV) {
    // Match HCCL 9.1 GetProtocolByEngine(COMM_ENGINE_AIV): UB_MEM first,
    // PCIE fallback. Some environments do not expose UB_MEM in rank graph.
    return protocol == COMM_PROTOCOL_UB_MEM || protocol == COMM_PROTOCOL_PCIE ||
           (HcclAivAllowExperimentalHccs() && protocol == COMM_PROTOCOL_HCCS);
  }
  return true;
}

int HcclLinkProtocolScore(CommEngine engine, CommProtocol protocol) {
  if (engine == COMM_ENGINE_AIV) {
    switch (protocol) {
    case COMM_PROTOCOL_UB_MEM:
      return 0;
    case COMM_PROTOCOL_PCIE:
      return 1;
    case COMM_PROTOCOL_HCCS:
      return 10;
    default:
      return 100;
    }
  }
  switch (protocol) {
  case COMM_PROTOCOL_UB_MEM:
    return 0;
  case COMM_PROTOCOL_UBC_CTP:
  case COMM_PROTOCOL_UBC_TP:
    return 1;
  case COMM_PROTOCOL_HCCS:
    return 2;
  case COMM_PROTOCOL_PCIE:
    return 3;
  case COMM_PROTOCOL_ROCE:
    return 4;
  case COMM_PROTOCOL_SIO:
    return 5;
  default:
    return 100;
  }
}

uint64_t HcclPeerBit(uint32_t peer) {
  return peer < 64 ? (1ull << peer) : 0;
}

uint64_t HcclRankMask(uint32_t nranks) {
  if (nranks == 0) {
    return 0;
  }
  return nranks >= 64 ? ~0ull : ((1ull << nranks) - 1ull);
}

uint64_t HcclProtocolBit(CommProtocol protocol) {
  int value = static_cast<int>(protocol);
  return value >= 0 && value < 64 ? (1ull << static_cast<uint32_t>(value)) : 0;
}

void HcclChannelProbeMarkProtocol(TritonAscendGinHcclChannelProbe *probe, uint32_t peer,
                                  CommProtocol protocol) {
  if (probe == nullptr) {
    return;
  }
  const uint64_t peerBit = HcclPeerBit(peer);
  probe->protocol_mask |= HcclProtocolBit(protocol);
  switch (protocol) {
  case COMM_PROTOCOL_HCCS:
    probe->hccs_peer_mask |= peerBit;
    break;
  case COMM_PROTOCOL_ROCE:
    probe->roce_peer_mask |= peerBit;
    break;
  case COMM_PROTOCOL_PCIE:
    probe->pcie_peer_mask |= peerBit;
    break;
  case COMM_PROTOCOL_UBC_CTP:
    probe->ubc_ctp_peer_mask |= peerBit;
    break;
  case COMM_PROTOCOL_UBC_TP:
    probe->ubc_tp_peer_mask |= peerBit;
    break;
  case COMM_PROTOCOL_UB_MEM:
    probe->ub_mem_peer_mask |= peerBit;
    break;
  default:
    break;
  }
}

void HcclChannelProbeRecordFirstError(TritonAscendGinHcclChannelProbe *probe,
                                      int status, uint32_t peer) {
  if (probe != nullptr && probe->first_error_status == 0 && status != 0) {
    probe->first_error_status = status;
    probe->first_error_peer = peer;
  }
}

void HcclChannelProbeRecordFirstAcquire(TritonAscendGinHcclChannelProbe *probe,
                                        int acquireRet, uint32_t peer,
                                        CommEngine engine,
                                        const HcclChannelDesc &desc,
                                        uint32_t channelCount,
                                        ChannelHandle channel) {
  if (probe == nullptr || probe->first_acquire_ret != 0 || acquireRet == 0) {
    return;
  }
  probe->first_acquire_ret = acquireRet;
  probe->first_acquire_peer = peer;
  probe->first_acquire_engine = static_cast<uint32_t>(engine);
  probe->first_acquire_protocol = static_cast<uint32_t>(desc.channelProtocol);
  probe->first_acquire_channel_count = channelCount;
  probe->first_acquire_mem_handle_count = desc.memHandleNum;
  probe->first_acquire_channel = static_cast<uint64_t>(channel);
}

void HcclAlgBufferProbeRecordFirstError(TritonAscendGinHcclAlgBufferProbe *probe,
                                        int status) {
  if (probe != nullptr && probe->first_error_status == 0 && status != 0) {
    probe->first_error_status = status;
  }
}

uint64_t HcclAlgDeviceMemBase(void *deviceMem) {
  if (deviceMem == nullptr) {
    return 0;
  }
  return PtrToU64(*reinterpret_cast<void **>(deviceMem));
}

int EnsureHcclAivOpbaseBuffers(TritonAscendGinRuntimeHandle *runtime) {
  if (runtime == nullptr) {
    SetLastError("EnsureHcclAivOpbaseBuffers received null runtime");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (runtime->hcclAivOpbaseBuffersCreated) {
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }

  int ret = LoadHcclAlgBufferSymbols(nullptr, &runtime->hcclAlgBuffer);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    return ret;
  }

  if (runtime->hcclAivBufferManagerStorage == nullptr) {
    void *storage = nullptr;
    int allocRet = posix_memalign(&storage, 64, kHcclBufferManagerStorageBytes);
    if (allocRet != 0 || storage == nullptr) {
      SetLastError("posix_memalign for HCCL AIV CCLBufferManager storage failed: " +
                   std::string(std::strerror(allocRet)));
      return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
    }
    std::memset(storage, 0, kHcclBufferManagerStorageBytes);
    runtime->hcclAivBufferManagerStorage = storage;
  }

  if (!runtime->hcclAivBufferManagerConstructed) {
    runtime->hcclAlgBuffer.ctor(runtime->hcclAivBufferManagerStorage);
    runtime->hcclAivBufferManagerConstructed = true;
  }

  void *cclPtr = nullptr;
  unsigned long cclBytes = 0;
  ret = runtime->hcclAlgBuffer.getIndependentOpCclBuffer(
      runtime->hcclAivBufferManagerStorage, cclPtr, cclBytes);
  if (ret != 0 || cclPtr == nullptr || cclBytes == 0) {
    SetLastError("CCLBufferManager::GetIndependentOpCCLbuffer failed before "
                 "CreateCommAIVbuffer with code " + std::to_string(ret) +
                 " ptr=" + std::to_string(PtrToU64(cclPtr)) +
                 " bytes=" + std::to_string(static_cast<uint64_t>(cclBytes)));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }

  ret = runtime->hcclAlgBuffer.createCommAivBuffer(runtime->hcclAivBufferManagerStorage, true);
  if (ret != 0) {
    SetLastError("CCLBufferManager::CreateCommAIVbuffer(opbase=true) failed with code " +
                 std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }

  void *inAivDeviceMem =
      runtime->hcclAlgBuffer.getInAivOpbaseBuffer(runtime->hcclAivBufferManagerStorage);
  void *outAivDeviceMem =
      runtime->hcclAlgBuffer.getOutAivOpbaseBuffer(runtime->hcclAivBufferManagerStorage);
  runtime->hcclInAivOpbasePtr = HcclAlgDeviceMemBase(inAivDeviceMem);
  runtime->hcclOutAivOpbasePtr = HcclAlgDeviceMemBase(outAivDeviceMem);
  runtime->hcclInAivOpbaseBytes =
      runtime->hcclInAivOpbasePtr == 0 ? 0 : kHcclAivInOpbaseBytes;
  runtime->hcclOutAivOpbaseBytes =
      runtime->hcclOutAivOpbasePtr == 0 ? 0 : kHcclAivOutOpbaseBytes;
  if (runtime->hcclInAivOpbasePtr == 0 || runtime->hcclOutAivOpbasePtr == 0) {
    SetLastError("CCLBufferManager returned null HCCL AIV opbase buffer pointers: in=0x" +
                 std::to_string(runtime->hcclInAivOpbasePtr) + " out=0x" +
                 std::to_string(runtime->hcclOutAivOpbasePtr));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }

  aclError aclRet = aclrtMemset(reinterpret_cast<void *>(runtime->hcclInAivOpbasePtr),
                                runtime->hcclInAivOpbaseBytes, 0,
                                runtime->hcclInAivOpbaseBytes);
  if (aclRet != ACL_SUCCESS) {
    SetAclError("aclrtMemset HCCL AIV input opbase buffer", aclRet);
    return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
  }
  aclRet = aclrtMemset(reinterpret_cast<void *>(runtime->hcclOutAivOpbasePtr),
                       runtime->hcclOutAivOpbaseBytes, 0,
                       runtime->hcclOutAivOpbaseBytes);
  if (aclRet != ACL_SUCCESS) {
    SetAclError("aclrtMemset HCCL AIV output opbase buffer", aclRet);
    return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
  }

  runtime->hcclAivOpbaseBuffersCreated = true;
  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] HCCL AIV opbase buffers rank=%u "
                 "ccl=0x%llx ccl_bytes=%llu in=0x%llx bytes=%llu out=0x%llx bytes=%llu\n",
                 runtime->hostDesc.rank,
                 static_cast<unsigned long long>(PtrToU64(cclPtr)),
                 static_cast<unsigned long long>(cclBytes),
                 static_cast<unsigned long long>(runtime->hcclInAivOpbasePtr),
                 static_cast<unsigned long long>(runtime->hcclInAivOpbaseBytes),
                 static_cast<unsigned long long>(runtime->hcclOutAivOpbasePtr),
                 static_cast<unsigned long long>(runtime->hcclOutAivOpbaseBytes));
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

void ReleaseHcclAivOpbaseBuffers(TritonAscendGinRuntimeHandle *runtime) {
  if (runtime == nullptr) {
    return;
  }
  if (runtime->hcclAivBufferManagerConstructed &&
      HcclAlgBufferSymbolsComplete(runtime->hcclAlgBuffer) &&
      runtime->hcclAivBufferManagerStorage != nullptr) {
    if (runtime->hcclAivOpbaseBuffersCreated) {
      (void)runtime->hcclAlgBuffer.clearCommAivBuffer(runtime->hcclAivBufferManagerStorage);
      (void)runtime->hcclAlgBuffer.releaseCommAivBuffer(runtime->hcclAivBufferManagerStorage);
    }
    runtime->hcclAlgBuffer.dtor(runtime->hcclAivBufferManagerStorage);
  }
  runtime->hcclAivBufferManagerConstructed = false;
  runtime->hcclAivOpbaseBuffersCreated = false;
  runtime->hcclInAivOpbasePtr = 0;
  runtime->hcclInAivOpbaseBytes = 0;
  runtime->hcclOutAivOpbasePtr = 0;
  runtime->hcclOutAivOpbaseBytes = 0;
  if (runtime->hcclAivBufferManagerStorage != nullptr) {
    std::free(runtime->hcclAivBufferManagerStorage);
    runtime->hcclAivBufferManagerStorage = nullptr;
  }
  CloseHcclAlgBufferSymbols(&runtime->hcclAlgBuffer);
}

struct HcclAivIpcRecord {
  std::string key;
  uint64_t bytes = 0;
  uint64_t offset = 0;
};

bool HcclAivIpcHccsEnabled();

std::string HcclAivDirectFilePath(const std::string &dir, const std::string &id,
                                  uint32_t rank, const char *suffix) {
  return dir + "/triton_gin_hccl_aiv_direct_" + id + ".rank" +
         std::to_string(rank) + "." + suffix;
}

int WaitReadTextFile(const std::string &path, const std::chrono::steady_clock::time_point &deadline,
                     std::string *value, const char *what) {
  while (true) {
    if (ReadTextFile(path, value)) {
      return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      SetLastError(std::string("timeout waiting for ") + what + " file " + path);
      return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

bool ParseHcclAivIpcRecord(const std::string &text, HcclAivIpcRecord *inRecord,
                           HcclAivIpcRecord *outRecord) {
  if (inRecord == nullptr || outRecord == nullptr) {
    return false;
  }
  std::istringstream stream(text);
  std::string label;
  stream >> label >> inRecord->key >> inRecord->bytes >> inRecord->offset;
  if (label != "in" || inRecord->key.empty() || inRecord->bytes == 0) {
    return false;
  }
  stream >> label >> outRecord->key >> outRecord->bytes >> outRecord->offset;
  return label == "out" && !outRecord->key.empty() && outRecord->bytes != 0;
}

int EnsureHcclPlfIpcSymbols(TritonAscendGinRuntimeHandle *runtime) {
  if (runtime == nullptr) {
    SetLastError("EnsureHcclPlfIpcSymbols received null runtime");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (HcclPlfIpcSymbolsComplete(runtime->hcclPlfIpc)) {
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }
  return LoadHcclPlfIpcSymbols(nullptr, &runtime->hcclPlfIpc);
}

int CurrentAclDeviceId(int32_t *deviceId) {
  if (deviceId == nullptr) {
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  aclError err = aclrtGetDevice(deviceId);
  if (err != ACL_SUCCESS) {
    SetAclError("aclrtGetDevice", err);
    return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

int ExportHcclAivIpcRecordForPeers(TritonAscendGinRuntimeHandle *runtime,
                                   void *ptr, uint64_t bytes, const char *what,
                                   const int32_t *peerPids,
                                   HcclAivIpcRecord *record) {
  if (runtime == nullptr || ptr == nullptr || bytes == 0 || peerPids == nullptr ||
      record == nullptr) {
    SetLastError("ExportHcclAivIpcRecordForPeers received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  int ret = EnsureHcclPlfIpcSymbols(runtime);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    return ret;
  }
  int32_t deviceId = 0;
  ret = CurrentAclDeviceId(&deviceId);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    return ret;
  }
  void *repo = runtime->hcclPlfIpc.memNameGetInstance(deviceId);
  if (repo == nullptr) {
    SetLastError("HCCL MemNameRepository::GetInstance returned null for device " +
                 std::to_string(deviceId));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }

  uint8_t key[kAclIpcKeyBytes] = {};
  unsigned long long offset = 0;
  bool exported = false;
  const bool useHccsIpc = HcclAivIpcHccsEnabled();
  for (uint32_t peer = 0; peer < runtime->hostDesc.nranks; ++peer) {
    if (peer == runtime->hostDesc.rank) {
      continue;
    }
    ret = runtime->hcclPlfIpc.memNameSetIpcMem(
        repo, ptr, static_cast<unsigned long long>(bytes), key,
        static_cast<uint32_t>(sizeof(key)), offset, peerPids[peer], -1,
        useHccsIpc);
    if (ret != 0) {
      SetLastError(std::string("MemNameRepository::SetIpcMem failed for ") +
                   what + " peer=" + std::to_string(peer) +
                   " ret=" + std::to_string(ret) +
                   " ptr=" + std::to_string(PtrToU64(ptr)) +
                   " bytes=" + std::to_string(bytes));
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }
    exported = true;
  }
  if (!exported) {
    SetLastError("HCCL AIV direct requires at least two ranks");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  record->key = reinterpret_cast<const char *>(key);
  record->bytes = bytes;
  record->offset = static_cast<uint64_t>(offset);
  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] HCCL AIV direct export rank=%u %s "
                 "ptr=%p bytes=%llu offset=%llu key=%s\n",
                 runtime->hostDesc.rank, what, ptr,
                 static_cast<unsigned long long>(bytes),
                 static_cast<unsigned long long>(record->offset),
                 record->key.c_str());
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

int ImportHcclAivIpcRecord(TritonAscendGinRuntimeHandle *runtime,
                            const HcclAivIpcRecord &record, const char *what,
                            void **ptr) {
  if (runtime == nullptr || record.key.empty() || record.bytes == 0 || ptr == nullptr) {
    SetLastError("ImportHcclAivIpcRecord received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  int ret = EnsureHcclPlfIpcSymbols(runtime);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    return ret;
  }
  int32_t deviceId = 0;
  ret = CurrentAclDeviceId(&deviceId);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    return ret;
  }
  void *repo = runtime->hcclPlfIpc.memNameGetInstance(deviceId);
  if (repo == nullptr) {
    SetLastError("HCCL MemNameRepository::GetInstance returned null for device " +
                 std::to_string(deviceId));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  bool firstOpened = false;
  *ptr = nullptr;
  ret = runtime->hcclPlfIpc.memNameOpenIpcMem(
      repo, ptr, static_cast<unsigned long long>(record.bytes),
      reinterpret_cast<const uint8_t *>(record.key.c_str()),
      kAclIpcKeyBytes, static_cast<unsigned long long>(record.offset),
      firstOpened, HcclAivIpcHccsEnabled());
  if (ret != 0 || *ptr == nullptr) {
    SetLastError(std::string("MemNameRepository::OpenIpcMem failed for ") +
                 what + " ret=" + std::to_string(ret) +
                 " key=" + record.key +
                 " bytes=" + std::to_string(record.bytes) +
                 " offset=" + std::to_string(record.offset));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] HCCL AIV direct import rank=%u %s "
                 "ptr=%p bytes=%llu offset=%llu first_opened=%d key=%s\n",
                 runtime->hostDesc.rank, what, *ptr,
                 static_cast<unsigned long long>(record.bytes),
                 static_cast<unsigned long long>(record.offset),
                 firstOpened ? 1 : 0, record.key.c_str());
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

bool HcclAivIpcHccsEnabled() {
  const char *value = std::getenv("TRITON_ASCEND_GIN_HCCL_AIV_IPC_HCCS");
  if (value == nullptr) {
    return false;
  }
  return std::strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 ||
         strcasecmp(value, "yes") == 0 || strcasecmp(value, "on") == 0;
}

int ExchangeHcclAivDirectBuffers(TritonAscendGinRuntimeHandle *runtime,
                                 const char *rendezvousId,
                                 void **buffersIn,
                                 void **buffersOut) {
  if (runtime == nullptr || buffersIn == nullptr || buffersOut == nullptr) {
    SetLastError("ExchangeHcclAivDirectBuffers received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  const uint32_t rank = runtime->hostDesc.rank;
  const uint32_t nranks = runtime->hostDesc.nranks;
  if (nranks < 2 || nranks > kHcclAivDirectMaxRanks || rank >= nranks) {
    SetLastError("HCCL AIV direct allgather supports rank_size in [2," +
                 std::to_string(kHcclAivDirectMaxRanks) + "], got " +
                 std::to_string(nranks));
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  std::memset(buffersIn, 0, sizeof(void *) * kHcclAivDirectMaxRanks);
  std::memset(buffersOut, 0, sizeof(void *) * kHcclAivDirectMaxRanks);
  buffersIn[rank] = reinterpret_cast<void *>(runtime->hcclInAivOpbasePtr);
  buffersOut[rank] = reinterpret_cast<void *>(runtime->hcclOutAivOpbasePtr);

  const std::string id = SanitizeRendezvousId(rendezvousId);
  const std::string dir = EnvOrDefault("TRITON_ASCEND_GIN_RENDEZVOUS_DIR", "/tmp");
  const int timeoutMs = GetEnvInt(
      "TRITON_ASCEND_GIN_HCCL_AIV_DIRECT_TIMEOUT_MS",
      GetEnvInt("TRITON_ASCEND_GIN_RENDEZVOUS_TIMEOUT_MS", 120000));
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);

  const std::string pidPath = HcclAivDirectFilePath(dir, id, rank, "pid");
  const std::string ipcPath = HcclAivDirectFilePath(dir, id, rank, "ipc");
  (void)std::remove(pidPath.c_str());
  (void)std::remove(ipcPath.c_str());
  if (!WriteTextFile(pidPath, std::to_string(static_cast<int32_t>(getpid())))) {
    SetLastError("failed to write HCCL AIV direct pid file: " + pidPath);
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }

  int32_t peerPids[kHcclAivDirectMaxRanks] = {};
  peerPids[rank] = static_cast<int32_t>(getpid());
  for (uint32_t peer = 0; peer < nranks; ++peer) {
    if (peer == rank) {
      continue;
    }
    std::string text;
    int ret = WaitReadTextFile(HcclAivDirectFilePath(dir, id, peer, "pid"),
                               deadline, &text, "HCCL AIV direct pid");
    if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      return ret;
    }
    peerPids[peer] = std::atoi(text.c_str());
    if (peerPids[peer] <= 0) {
      SetLastError("invalid HCCL AIV direct peer pid for rank " +
                   std::to_string(peer) + ": " + text);
      return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
    }
  }

  HcclAivIpcRecord inRecord;
  HcclAivIpcRecord outRecord;
  int ret = ExportHcclAivIpcRecordForPeers(
      runtime, reinterpret_cast<void *>(runtime->hcclInAivOpbasePtr),
      runtime->hcclInAivOpbaseBytes, "aiv_input", peerPids, &inRecord);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    return ret;
  }
  ret = ExportHcclAivIpcRecordForPeers(
      runtime, reinterpret_cast<void *>(runtime->hcclOutAivOpbasePtr),
      runtime->hcclOutAivOpbaseBytes, "aiv_output", peerPids, &outRecord);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    return ret;
  }
  std::ostringstream info;
  info << "in " << inRecord.key << " " << inRecord.bytes << " "
       << inRecord.offset << "\n";
  info << "out " << outRecord.key << " " << outRecord.bytes << " "
       << outRecord.offset << "\n";
  if (!WriteTextFile(ipcPath, info.str())) {
    SetLastError("failed to write HCCL AIV direct IPC file: " + ipcPath);
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }

  for (uint32_t peer = 0; peer < nranks; ++peer) {
    if (peer == rank) {
      continue;
    }
    std::string text;
    ret = WaitReadTextFile(HcclAivDirectFilePath(dir, id, peer, "ipc"),
                           deadline, &text, "HCCL AIV direct IPC");
    if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      return ret;
    }
    HcclAivIpcRecord peerIn;
    HcclAivIpcRecord peerOut;
    if (!ParseHcclAivIpcRecord(text, &peerIn, &peerOut)) {
      SetLastError("invalid HCCL AIV direct IPC record for peer " +
                   std::to_string(peer));
      return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
    }
    ret = ImportHcclAivIpcRecord(runtime, peerOut, "peer_aiv_output",
                                 &buffersOut[peer]);
    if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      return ret;
    }
    ret = ImportHcclAivIpcRecord(runtime, peerIn, "peer_aiv_input",
                                 &buffersIn[peer]);
    if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      return ret;
    }
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

int NextHcclAivDirectTag(TritonAscendGinRuntimeHandle *runtime) {
  runtime->hcclAivDirectTag = (runtime->hcclAivDirectTag % 1000) + 1;
  return runtime->hcclAivDirectTag;
}

std::string HcclCommNameOrFallback(TritonAscendGinRuntimeHandle *runtime,
                                   const char *fallback) {
  if (runtime != nullptr && runtime->hcclChannel.getCommName != nullptr &&
      runtime->hcclComm != nullptr) {
    char name[kHcclOpExchangeMaxLength] = {};
    if (runtime->hcclChannel.getCommName(runtime->hcclComm, name) == 0 &&
        name[0] != '\0') {
      return std::string(name);
    }
  }
  return fallback == nullptr || fallback[0] == '\0' ? "triton_gin" : fallback;
}

bool HcclChannelEngineUsesLegacyCclBuffer(CommEngine engine) {
  return engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CPU_TS ||
         engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS;
}

bool HcclAivDirectChannelAcquireEnabled() {
  const char *value = std::getenv("TRITON_ASCEND_GIN_HCCL_AIV_DIRECT_ACQUIRE");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

CommEngine HcclRegisterAcquireEngine(CommEngine requestedEngine) {
  if (requestedEngine == COMM_ENGINE_AIV && !HcclAivDirectChannelAcquireEnabled()) {
    return COMM_ENGINE_CPU;
  }
  return requestedEngine;
}

const char *HcclChannelEngineName(CommEngine engine) {
  switch (engine) {
  case COMM_ENGINE_CPU:
    return "cpu";
  case COMM_ENGINE_CPU_TS:
    return "cpu_ts";
  case COMM_ENGINE_AICPU:
    return "aicpu";
  case COMM_ENGINE_AICPU_TS:
    return "aicpu_ts";
  case COMM_ENGINE_AIV:
    return "aiv";
  case COMM_ENGINE_CCU:
    return "ccu";
  default:
    return "unknown";
  }
}

bool HcclChannelExchangeInfoEnabled() {
  const char *value = std::getenv("TRITON_ASCEND_GIN_HCCL_ADD_EXCHANGE_INFO");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("TRITON_ASCEND_GIN_HCCL_PROBE_ADD_EXCHANGE_INFO");
  }
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool HcclProbeAivDescriptorViaCpuEnabled() {
  const char *value =
      std::getenv("TRITON_ASCEND_GIN_HCCL_PROBE_AIV_DESCRIPTOR_VIA_CPU");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool HcclProbeAivDescriptorSymmetricFallbackEnabled() {
  const char *value =
      std::getenv("TRITON_ASCEND_GIN_HCCL_PROBE_AIV_DESCRIPTOR_SYMMETRIC_FALLBACK");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void CopyCStringBounded(char *dst, size_t dstSize, const std::string &src) {
  if (dst == nullptr || dstSize == 0) {
    return;
  }
  std::snprintf(dst, dstSize, "%s", src.c_str());
  dst[dstSize - 1] = '\0';
}

uint32_t GetHcclExchangeEnvU32(const char *primary, const char *legacy,
                               uint32_t fallback) {
  const char *value = std::getenv(primary);
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv(legacy);
  }
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char *end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 0);
  if (end == value || parsed > std::numeric_limits<uint32_t>::max()) {
    return fallback;
  }
  return static_cast<uint32_t>(parsed);
}

uint64_t GetHcclExchangeEnvU64(const char *primary, const char *legacy,
                               uint64_t fallback) {
  const char *value = std::getenv(primary);
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv(legacy);
  }
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char *end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 0);
  if (end == value) {
    return fallback;
  }
  return static_cast<uint64_t>(parsed);
}

std::string GetHcclExchangeEnvString(const char *primary, const char *legacy,
                                     const std::string &fallback) {
  const char *value = std::getenv(primary);
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv(legacy);
  }
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return std::string(value);
}

int HcclMaybeAddChannelExchangeInfo(TritonAscendGinRuntimeHandle *runtime,
                                    uint64_t knownCclBufferBytes,
                                    TritonAscendGinHcclChannelProbe *probe) {
  if (runtime == nullptr) {
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (!HcclChannelExchangeInfoEnabled()) {
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }
  if (probe != nullptr) {
    probe->exchange_info_enabled = 1;
    probe->exchange_info_symbol_available =
        runtime->hcclChannel.commAddExchangeInfo == nullptr ? 0u : 1u;
  }
  if (runtime->hcclChannel.commAddExchangeInfo == nullptr) {
    if (probe != nullptr) {
      probe->exchange_info_status = TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
    }
    SetLastError("HcclCommAddExchangeInfo symbol was not found");
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }

  uint64_t cclBufferBytes = knownCclBufferBytes;
  if (cclBufferBytes == 0 && runtime->hcclChannel.getHcclBuffer != nullptr) {
    void *unusedBuffer = nullptr;
    int ret = runtime->hcclChannel.getHcclBuffer(runtime->hcclComm, &unusedBuffer,
                                                 &cclBufferBytes);
    if (ret != 0) {
      if (probe != nullptr) {
        probe->exchange_info_status = ret;
      }
      SetLastError("HcclGetHcclBuffer for exchange info failed with code " +
                   std::to_string(ret));
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }
  }

  HcclOpExchangeInfoCompat info = {};
  info.cclBufferSize = cclBufferBytes;
  info.root = kHcclInvalidRankId;
  info.opType = kHcclCmdAllGather;
  info.opExecuteConfig = GetHcclExchangeEnvU32(
      "TRITON_ASCEND_GIN_HCCL_EXCHANGE_OP_EXECUTE_CONFIG",
      "TRITON_ASCEND_GIN_HCCL_PROBE_EXCHANGE_OP_EXECUTE_CONFIG",
      kHcclOpExecuteAiv);
  info.reduceType = kHcclReduceReserved;
  info.dataType = GetHcclExchangeEnvU32(
      "TRITON_ASCEND_GIN_HCCL_EXCHANGE_DATA_TYPE",
      "TRITON_ASCEND_GIN_HCCL_PROBE_EXCHANGE_DATA_TYPE",
      kHcclDataTypeFp32);
  info.count = GetHcclExchangeEnvU64(
      "TRITON_ASCEND_GIN_HCCL_EXCHANGE_COUNT",
      "TRITON_ASCEND_GIN_HCCL_PROBE_EXCHANGE_COUNT",
      32);
  info.aivCoreLimit = GetHcclExchangeEnvU32(
      "TRITON_ASCEND_GIN_HCCL_EXCHANGE_AIV_CORE_LIMIT",
      "TRITON_ASCEND_GIN_HCCL_PROBE_EXCHANGE_AIV_CORE_LIMIT",
      kHcclMaxNumBlocks);

  std::string group;
  if (runtime->hcclChannel.getCommName != nullptr) {
    char commName[kHcclOpExchangeMaxLength] = {};
    int ret = runtime->hcclChannel.getCommName(runtime->hcclComm, commName);
    if (ret != 0) {
      if (probe != nullptr) {
        probe->exchange_info_status = ret;
      }
      SetLastError("HcclGetCommName for exchange info failed with code " +
                   std::to_string(ret));
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }
    commName[kHcclOpExchangeMaxLength - 1] = '\0';
    group = commName;
  }
  CopyCStringBounded(info.group, sizeof(info.group), group);

  const std::string defaultTag =
      group.empty() ? "AllGather_triton_gin_probe" : ("AllGather_" + group);
  std::string tag = GetHcclExchangeEnvString(
      "TRITON_ASCEND_GIN_HCCL_EXCHANGE_TAG",
      "TRITON_ASCEND_GIN_HCCL_PROBE_EXCHANGE_TAG",
      defaultTag);
  CopyCStringBounded(info.tag, sizeof(info.tag), tag);

  if (probe != nullptr) {
    probe->exchange_info_op_execute_config = info.opExecuteConfig;
    probe->exchange_info_ccl_buffer_bytes = info.cclBufferSize;
    probe->exchange_info_count = info.count;
    probe->exchange_info_data_type = info.dataType;
    probe->exchange_info_aiv_core_limit = info.aivCoreLimit;
  }

  int ret = runtime->hcclChannel.commAddExchangeInfo(runtime->hcclComm, &info,
                                                     sizeof(info));
  if (probe != nullptr) {
    probe->exchange_info_status = ret;
  }
  if (ret != 0) {
    SetLastError("HcclCommAddExchangeInfo failed with code " +
                 std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

int EnsureHcclAivCommInfo(TritonAscendGinRuntimeHandle *runtime, const char *rendezvousId) {
  if (runtime == nullptr || runtime->hcclComm == nullptr) {
    SetLastError("EnsureHcclAivCommInfo received invalid HCCL runtime");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (runtime->hcclAivCommInfoRegistered) {
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }

  const std::string tag = "triton_gin_aiv_comm_" + SanitizeRendezvousId(rendezvousId);
  void *ctx = nullptr;
  uint64_t ctxSize = kHcclAivCommInfoBytes;
  int ret = runtime->hcclChannel.engineCtxGet(runtime->hcclComm, tag.c_str(),
                                              COMM_ENGINE_AIV, &ctx, &ctxSize);
  if (ret == 0 && ctx != nullptr) {
    if (ctxSize < kHcclAivCommInfoBytes) {
      SetLastError("HcclEngineCtxGet for " + tag + " returned " +
                   std::to_string(ctxSize) + " bytes; required " +
                   std::to_string(kHcclAivCommInfoBytes));
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }
  } else {
    ctx = nullptr;
    ctxSize = kHcclAivCommInfoBytes;
    ret = runtime->hcclChannel.engineCtxCreate(runtime->hcclComm, tag.c_str(),
                                               COMM_ENGINE_AIV, kHcclAivCommInfoBytes, &ctx);
    if (ret != 0 || ctx == nullptr) {
      SetLastError("HcclEngineCtxCreate for " + tag + " failed with code " +
                   std::to_string(ret) + " ptr=" + std::to_string(PtrToU64(ctx)));
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }
    aclError err = aclrtMemset(ctx, kHcclAivCommInfoBytes, 0, kHcclAivCommInfoBytes);
    if (err != ACL_SUCCESS) {
      SetAclError("aclrtMemset HCCL AIV comm-info", err);
      return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
    }
  }

  CommMem regMem = {};
  regMem.type = COMM_MEM_TYPE_DEVICE;
  regMem.addr = ctx;
  regMem.size = kHcclAivCommInfoBytes;
  HcclMemHandle memHandle = nullptr;
  ret = runtime->hcclChannel.commMemReg(runtime->hcclComm, tag.c_str(), &regMem, &memHandle);
  if (ret != 0) {
    SetLastError("HcclCommMemReg AIV comm-info for " + tag + " failed with code " +
                 std::to_string(ret) + " ptr=" + std::to_string(PtrToU64(ctx)) +
                 " bytes=" + std::to_string(kHcclAivCommInfoBytes) +
                 " mem_handle=" + std::to_string(PtrToU64(memHandle)));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }

  runtime->hcclAivCommInfo = ctx;
  runtime->hcclAivCommInfoMemHandle = memHandle;
  runtime->hcclAivCommInfoRegistered = true;
  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] HCCL AIV comm-info rank=%u tag=%s "
                 "ptr=%p bytes=%llu mem_handle=%p\n",
                 runtime->hostDesc.rank, tag.c_str(), ctx,
                 static_cast<unsigned long long>(kHcclAivCommInfoBytes), memHandle);
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

int WriteAndVerifyHcclAivCommInfoEntry(TritonAscendGinRuntimeHandle *runtime,
                                        uint64_t offset, uint64_t value,
                                        const char *what, uint64_t *readBack) {
  if (runtime == nullptr || runtime->hcclAivCommInfo == nullptr ||
      offset > kHcclAivCommInfoBytes ||
      sizeof(value) > kHcclAivCommInfoBytes - offset) {
    SetLastError(std::string("HCCL AIV comm-info ") +
                 (what == nullptr ? "entry" : what) + " received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  auto *base = static_cast<uint8_t *>(runtime->hcclAivCommInfo);
  aclError err = aclrtMemcpy(base + offset, sizeof(value), &value, sizeof(value),
                             ACL_MEMCPY_HOST_TO_DEVICE);
  if (err != ACL_SUCCESS) {
    SetAclError((std::string("aclrtMemcpy HCCL AIV comm-info write ") +
                 (what == nullptr ? "entry" : what))
                    .c_str(),
                err);
    return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
  }

  uint64_t readValue = 0;
  err = aclrtMemcpy(&readValue, sizeof(readValue), base + offset, sizeof(readValue),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (err != ACL_SUCCESS) {
    SetAclError((std::string("aclrtMemcpy HCCL AIV comm-info read ") +
                 (what == nullptr ? "entry" : what))
                    .c_str(),
                err);
    return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
  }
  if (readValue != value) {
    SetLastError(std::string("HCCL AIV comm-info readback mismatch for ") +
                 (what == nullptr ? "entry" : what));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  if (readBack != nullptr) {
    *readBack = readValue;
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

int ProbeHcclAivCommInfoLocalLayout(TritonAscendGinRuntimeHandle *runtime,
                                    void *localCclBuffer,
                                    TritonAscendGinHcclChannelProbe *probe) {
  if (runtime == nullptr || probe == nullptr || runtime->hcclAivCommInfo == nullptr ||
      localCclBuffer == nullptr || runtime->hostDesc.rank >= kTileXRMaxRanks) {
    if (probe != nullptr) {
      probe->aiv_comm_info_layout_status = TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
    }
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  const uint32_t rank = runtime->hostDesc.rank;
  const uint64_t gmInOffset = static_cast<uint64_t>(rank) * sizeof(uint64_t);
  const uint64_t gmOutOffset = kHcclAivTagAddrOffset + static_cast<uint64_t>(rank) * sizeof(uint64_t);
  const uint64_t localIn = PtrToU64(localCclBuffer);
  const uint64_t localOut = PtrToU64(runtime->hcclAivCommInfo);
  const uint64_t localFlagBase = localOut + kHcclAivFlagAddrOffset;

  probe->aiv_comm_info_gm_in_offset = gmInOffset;
  probe->aiv_comm_info_gm_out_offset = gmOutOffset;
  probe->aiv_comm_info_local_gm_in = localIn;
  probe->aiv_comm_info_local_gm_out = localOut;
  probe->aiv_comm_info_local_flag_base = localFlagBase;

  uint64_t readIn = 0;
  uint64_t readOut = 0;
  int ret = WriteAndVerifyHcclAivCommInfoEntry(runtime, gmInOffset, localIn,
                                               "GM_IN local entry", &readIn);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    probe->aiv_comm_info_layout_status = ret;
    return ret;
  }
  ret = WriteAndVerifyHcclAivCommInfoEntry(runtime, gmOutOffset, localOut,
                                           "GM_OUT local entry", &readOut);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    probe->aiv_comm_info_layout_status = ret;
    return ret;
  }

  probe->aiv_comm_info_layout_status = TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] HCCL AIV comm-info local layout rank=%u "
                 "gm_in_off=%llu gm_out_off=%llu gm_in=0x%llx gm_out=0x%llx "
                 "flag_base=0x%llx\n",
                 rank,
                 static_cast<unsigned long long>(gmInOffset),
                 static_cast<unsigned long long>(gmOutOffset),
                 static_cast<unsigned long long>(localIn),
                 static_cast<unsigned long long>(localOut),
                 static_cast<unsigned long long>(localFlagBase));
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

int BuildHcclChannelDescForEngine(TritonAscendGinRuntimeHandle *runtime,
                                  CommEngine engine, uint32_t peer,
                                  HcclMemHandle *memHandle,
                                  HcclChannelDesc *desc) {
  if (runtime == nullptr || desc == nullptr) {
    SetLastError("BuildHcclChannelDesc received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  int ret = HcclChannelDescInit(desc, 1);
  if (ret != 0) {
    SetLastError("HcclChannelDescInit failed with code " + std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  desc->remoteRank = peer;
  desc->notifyNum = 3;
  desc->memHandles = memHandle;
  desc->memHandleNum = memHandle == nullptr ? 0 : 1;

  uint32_t *layers = nullptr;
  uint32_t layerNum = 0;
  ret = runtime->hcclChannel.rankGraphGetLayers(runtime->hcclComm, &layers, &layerNum);
  if (ret == 0 && layers != nullptr && layerNum != 0) {
    if (DebugEnabled()) {
      std::fprintf(stderr,
                    "[triton_ascend_gin] HcclRankGraphGetLayers rank=%u peer=%u "
                    "layer_num=%u engine=%s(%d)\n",
                    runtime->hostDesc.rank, peer, layerNum,
                    HcclChannelEngineName(engine),
                    static_cast<int>(engine));
    }
    bool found = false;
    CommLink best = {};
    int bestScore = 100;
    for (uint32_t layerIdx = 0; layerIdx < layerNum; ++layerIdx) {
      CommLink *links = nullptr;
      uint32_t linkNum = 0;
      int linkRet = runtime->hcclChannel.rankGraphGetLinks(
          runtime->hcclComm, layers[layerIdx], runtime->hostDesc.rank, peer, &links, &linkNum);
      if (linkRet != 0 || links == nullptr || linkNum == 0) {
        if (DebugEnabled()) {
          std::fprintf(stderr,
                       "[triton_ascend_gin] HcclRankGraphGetLinks rank=%u peer=%u "
                       "layer=%u link_ret=%d link_num=%u links=%p\n",
                       runtime->hostDesc.rank, peer, layers[layerIdx], linkRet,
                       linkNum, links);
        }
        continue;
      }
      for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
        CommProtocol protocol = links[linkIdx].linkAttr.linkProtocol;
        if (DebugEnabled()) {
          std::fprintf(stderr,
                       "[triton_ascend_gin] HCCL link rank=%u peer=%u layer=%u "
                       "idx=%u protocol=%d\n",
                       runtime->hostDesc.rank, peer, layers[layerIdx], linkIdx,
                       static_cast<int>(protocol));
        }
        if (!HcclLinkProtocolAllowedForEngine(engine, protocol)) {
          continue;
        }
        int score = HcclLinkProtocolScore(engine, protocol);
        if (!found || score < bestScore) {
          best = links[linkIdx];
          bestScore = score;
          found = true;
        }
      }
    }
    if (found) {
      desc->channelProtocol = best.linkAttr.linkProtocol;
      desc->localEndpoint = best.srcEndpointDesc;
      desc->remoteEndpoint = best.dstEndpointDesc;
      return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
    }
  }

  if (engine == COMM_ENGINE_AIV) {
    SetLastError("HCCL AIV channel requires a COMM_PROTOCOL_UB_MEM or "
                 "COMM_PROTOCOL_PCIE link from rank " +
                 std::to_string(runtime->hostDesc.rank) + " to peer " +
                 std::to_string(peer) +
                 (HcclAivAllowExperimentalHccs()
                      ? "; experimental HCCS fallback was enabled but no usable link was exposed"
                      : "; HcclRankGraphGetLinks did not expose one"));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }

  desc->channelProtocol = COMM_PROTOCOL_HCCS;
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

int BuildHcclChannelDesc(TritonAscendGinRuntimeHandle *runtime, uint32_t peer,
                         HcclMemHandle *memHandle, HcclChannelDesc *desc) {
  if (runtime == nullptr) {
    SetLastError("BuildHcclChannelDesc received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  return BuildHcclChannelDescForEngine(runtime, runtime->hcclChannelEngine, peer,
                                       memHandle, desc);
}

void HcclAivDescriptorViaCpuRecordFailure(TritonAscendGinHcclChannelProbe *probe,
                                          int status, int hcclRet,
                                          uint32_t peer, uint32_t protocol,
                                          ChannelHandle channel) {
  if (probe == nullptr || status == TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    return;
  }
  if (probe->aiv_descriptor_via_cpu_status == TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    probe->aiv_descriptor_via_cpu_status = status;
  }
  if (probe->aiv_descriptor_via_cpu_first_ret == 0) {
    probe->aiv_descriptor_via_cpu_first_ret =
        hcclRet == 0 ? status : hcclRet;
    probe->aiv_descriptor_via_cpu_first_peer = peer;
    probe->aiv_descriptor_via_cpu_first_protocol = protocol;
    probe->aiv_descriptor_via_cpu_first_channel = static_cast<uint64_t>(channel);
  }
}

int ProbeHcclAivDescriptorViaCpuChannels(TritonAscendGinRuntimeHandle *runtime,
                                         void *localCclBuffer,
                                         uint64_t localCclBufferBytes,
                                         TritonAscendGinHcclChannelProbe *probe) {
  (void)localCclBufferBytes;
  if (probe != nullptr) {
    probe->aiv_descriptor_via_cpu_enabled = 1;
    probe->aiv_descriptor_via_cpu_status = TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }
  if (runtime == nullptr || probe == nullptr || runtime->hcclComm == nullptr ||
      runtime->hcclAivCommInfo == nullptr || localCclBuffer == nullptr ||
      runtime->hostDesc.nranks == 0 || runtime->hostDesc.nranks > kTileXRMaxRanks ||
      runtime->hostDesc.rank >= runtime->hostDesc.nranks) {
    HcclAivDescriptorViaCpuRecordFailure(
        probe, TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE,
        TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE,
        runtime == nullptr ? 0 : runtime->hostDesc.rank, 0, 0);
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  const uint32_t rank = runtime->hostDesc.rank;
  const uint32_t nranks = runtime->hostDesc.nranks;
  const uint64_t rankBit = HcclPeerBit(rank);
  const uint64_t fullMask = HcclRankMask(nranks);

  uint64_t readValue = 0;
  int ret = WriteAndVerifyHcclAivCommInfoEntry(
      runtime, static_cast<uint64_t>(rank) * sizeof(uint64_t),
      PtrToU64(localCclBuffer), "GM_IN local entry via CPU descriptor",
      &readValue);
  if (ret == TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    probe->aiv_descriptor_via_cpu_hccl_buffer_peer_mask |= rankBit;
    probe->aiv_descriptor_via_cpu_readback_gm_in_mask |= rankBit;
  } else {
    HcclAivDescriptorViaCpuRecordFailure(probe, ret, ret, rank, 0, 0);
  }

  ret = WriteAndVerifyHcclAivCommInfoEntry(
      runtime, kHcclAivTagAddrOffset + static_cast<uint64_t>(rank) * sizeof(uint64_t),
      PtrToU64(runtime->hcclAivCommInfo), "GM_OUT local entry via CPU descriptor",
      &readValue);
  if (ret == TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    probe->aiv_descriptor_via_cpu_remote_mem_peer_mask |= rankBit;
    probe->aiv_descriptor_via_cpu_readback_gm_out_mask |= rankBit;
  } else {
    HcclAivDescriptorViaCpuRecordFailure(probe, ret, ret, rank, 0, 0);
  }

  HcclMemHandle aivMemHandle = runtime->hcclAivCommInfoMemHandle;
  HcclMemHandle *aivMemHandlePtr =
      runtime->hcclAivCommInfoRegistered ? &aivMemHandle : nullptr;
  const std::string aivCommInfoTag = "triton_gin_aiv_comm_probe";
  const bool useSymmetricFallback =
      HcclProbeAivDescriptorSymmetricFallbackEnabled();
  auto writeSymmetricGmOut = [&](uint32_t peer, uint32_t protocol,
                                 ChannelHandle channel) -> bool {
    const uint64_t peerBit = HcclPeerBit(peer);
    int writeRet = WriteAndVerifyHcclAivCommInfoEntry(
        runtime, kHcclAivTagAddrOffset + static_cast<uint64_t>(peer) * sizeof(uint64_t),
        PtrToU64(runtime->hcclAivCommInfo),
        "GM_OUT symmetric AIV comm-info fallback via CPU descriptor",
        &readValue);
    if (writeRet == TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      probe->aiv_descriptor_via_cpu_symmetric_gm_out_peer_mask |= peerBit;
      probe->aiv_descriptor_via_cpu_readback_gm_out_mask |= peerBit;
      return true;
    }
    HcclAivDescriptorViaCpuRecordFailure(probe, writeRet, writeRet, peer,
                                         protocol, channel);
    return false;
  };

  for (uint32_t peer = 0; peer < nranks; ++peer) {
    if (peer == rank) {
      continue;
    }
    const uint64_t peerBit = HcclPeerBit(peer);
    HcclChannelDesc desc = {};
    ret = BuildHcclChannelDescForEngine(runtime, COMM_ENGINE_CPU, peer,
                                        aivMemHandlePtr, &desc);
    if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      HcclAivDescriptorViaCpuRecordFailure(probe, ret, ret, peer, 0, 0);
      continue;
    }

    ChannelHandle channel = 0;
    int acquireRet = runtime->hcclChannel.channelAcquire(
        runtime->hcclComm, COMM_ENGINE_CPU, &desc, 1, &channel);
    if (acquireRet != 0 || channel == 0) {
      HcclAivDescriptorViaCpuRecordFailure(
          probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR, acquireRet, peer,
          static_cast<uint32_t>(desc.channelProtocol), channel);
      continue;
    }
    probe->aiv_descriptor_via_cpu_acquire_peer_mask |= peerBit;

    void *remoteBuffer = nullptr;
    uint64_t remoteBufferBytes = 0;
    ret = runtime->hcclChannel.channelGetHcclBuffer(
        runtime->hcclComm, channel, &remoteBuffer, &remoteBufferBytes);
    if (ret == 0 && remoteBuffer != nullptr) {
      int writeRet = WriteAndVerifyHcclAivCommInfoEntry(
          runtime, static_cast<uint64_t>(peer) * sizeof(uint64_t),
          PtrToU64(remoteBuffer), "GM_IN remote CCL buffer via CPU descriptor",
          &readValue);
      if (writeRet == TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
        probe->aiv_descriptor_via_cpu_hccl_buffer_peer_mask |= peerBit;
        probe->aiv_descriptor_via_cpu_readback_gm_in_mask |= peerBit;
      } else {
        HcclAivDescriptorViaCpuRecordFailure(
            probe, writeRet, writeRet, peer,
            static_cast<uint32_t>(desc.channelProtocol), channel);
      }
    } else {
      HcclAivDescriptorViaCpuRecordFailure(
          probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR, ret, peer,
          static_cast<uint32_t>(desc.channelProtocol), channel);
    }

    uint32_t memNum = 0;
    CommMem *remoteMems = nullptr;
    char **memTags = nullptr;
    ret = runtime->hcclChannel.channelGetRemoteMems(runtime->hcclComm, channel,
                                                    &memNum, &remoteMems, &memTags);
    if (ret != 0 || memNum == 0 || remoteMems == nullptr) {
      if (useSymmetricFallback &&
          writeSymmetricGmOut(peer, static_cast<uint32_t>(desc.channelProtocol),
                              channel)) {
        continue;
      }
      HcclAivDescriptorViaCpuRecordFailure(
          probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR, ret, peer,
          static_cast<uint32_t>(desc.channelProtocol), channel);
      continue;
    }

    uint32_t selected = memNum - 1;
    if (memTags != nullptr) {
      for (uint32_t idx = 0; idx < memNum; ++idx) {
        if (memTags[idx] != nullptr && aivCommInfoTag == memTags[idx]) {
          selected = idx;
          break;
        }
      }
    }
    if (remoteMems[selected].addr == nullptr ||
        remoteMems[selected].size < kHcclAivCommInfoBytes) {
      if (useSymmetricFallback &&
          writeSymmetricGmOut(peer, static_cast<uint32_t>(desc.channelProtocol),
                              channel)) {
        continue;
      }
      HcclAivDescriptorViaCpuRecordFailure(
          probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR,
          TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR, peer,
          static_cast<uint32_t>(desc.channelProtocol), channel);
      continue;
    }

    int writeRet = WriteAndVerifyHcclAivCommInfoEntry(
        runtime, kHcclAivTagAddrOffset + static_cast<uint64_t>(peer) * sizeof(uint64_t),
        PtrToU64(remoteMems[selected].addr),
        "GM_OUT remote AIV comm-info via CPU descriptor", &readValue);
    if (writeRet == TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      probe->aiv_descriptor_via_cpu_remote_mem_peer_mask |= peerBit;
      probe->aiv_descriptor_via_cpu_readback_gm_out_mask |= peerBit;
    } else {
      HcclAivDescriptorViaCpuRecordFailure(
          probe, writeRet, writeRet, peer,
          static_cast<uint32_t>(desc.channelProtocol), channel);
    }
  }

  if ((probe->aiv_descriptor_via_cpu_readback_gm_in_mask & fullMask) != fullMask ||
      (probe->aiv_descriptor_via_cpu_readback_gm_out_mask & fullMask) != fullMask) {
    HcclAivDescriptorViaCpuRecordFailure(
        probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR,
        TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR, rank, 0, 0);
  }

  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] HCCL AIV descriptor via CPU rank=%u "
                 "nranks=%u status=%d acquire=0x%llx ccl=0x%llx "
                 "remote_mem=0x%llx symmetric_gm_out=0x%llx "
                 "gm_in=0x%llx gm_out=0x%llx\n",
                 rank, nranks, probe->aiv_descriptor_via_cpu_status,
                 static_cast<unsigned long long>(
                     probe->aiv_descriptor_via_cpu_acquire_peer_mask),
                 static_cast<unsigned long long>(
                     probe->aiv_descriptor_via_cpu_hccl_buffer_peer_mask),
                 static_cast<unsigned long long>(
                     probe->aiv_descriptor_via_cpu_remote_mem_peer_mask),
                 static_cast<unsigned long long>(
                     probe->aiv_descriptor_via_cpu_symmetric_gm_out_peer_mask),
                 static_cast<unsigned long long>(
                     probe->aiv_descriptor_via_cpu_readback_gm_in_mask),
                 static_cast<unsigned long long>(
                     probe->aiv_descriptor_via_cpu_readback_gm_out_mask));
  }

  return probe->aiv_descriptor_via_cpu_status;
}

int RegisterHcclChannelLegacyCclBufferBases(TritonAscendGinRuntimeHandle *runtime, void *localPtr,
                                            uint64_t bytes, const char *rendezvousId, const char *what,
                                            uint64_t *bases, uint64_t *windowHandle) {
  if (runtime == nullptr || localPtr == nullptr || bases == nullptr || windowHandle == nullptr ||
      bytes == 0) {
    SetLastError(std::string("TritonAscendGinRegister") + what + " received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  const bool isSignalWindow = bases == runtime->hostDesc.peer_signal_base;
  if (isSignalWindow) {
    SetLastError("HCCL legacy channel uses signal slots embedded in the data window; "
                 "separate signal_window registration is not supported");
    return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
  }

  uint64_t signalBytes = HcclSignalWindowBytes(runtime);
  uint64_t signalOffset = AlignUpU64(bytes, kDefaultSignalStride);
  if (signalOffset < bytes ||
      signalBytes > std::numeric_limits<uint64_t>::max() - signalOffset) {
    SetLastError(std::string("HCCL legacy channel register for ") + what +
                 " overflowed while reserving embedded signal slots");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  uint64_t registerBytes = signalOffset + signalBytes;

  void *localBuffer = nullptr;
  uint64_t localBufferBytes = 0;
  int ret = runtime->hcclChannel.getHcclBuffer(runtime->hcclComm, &localBuffer, &localBufferBytes);
  if (ret != 0 || localBuffer == nullptr || localBufferBytes == 0) {
    SetLastError("HcclGetHcclBuffer failed with code " + std::to_string(ret) +
                 " ptr=" + std::to_string(PtrToU64(localBuffer)) +
                 " bytes=" + std::to_string(localBufferBytes));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  if (registerBytes > localBufferBytes) {
    SetLastError("HCCL legacy channel window requires " + std::to_string(registerBytes) +
                 " bytes but HCCL CCL buffer has only " + std::to_string(localBufferBytes) +
                 " bytes");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (localPtr != localBuffer) {
    SetLastError("HCCL legacy channel requires the registered window tensor to be backed by "
                 "the local HCCL CCL buffer. tensor_ptr=" + std::to_string(PtrToU64(localPtr)) +
                 " hccl_buffer_ptr=" + std::to_string(PtrToU64(localBuffer)) +
                 ". Use gin_runtime.hccl_buffer_tensor(...) for this backend.");
    return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
  }

  HcclMemHandle acquireMemHandle = nullptr;
  bool useAcquireMemHandle = false;
  if (runtime->hcclChannelEngine == COMM_ENGINE_AIV) {
    ret = EnsureHcclAivCommInfo(runtime, rendezvousId);
    if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      return ret;
    }
    acquireMemHandle = runtime->hcclAivCommInfoMemHandle;
    useAcquireMemHandle = true;
  }

  const uint32_t rank = runtime->hostDesc.rank;
  const uint32_t nranks = runtime->hostDesc.nranks;
  for (uint32_t peer = 0; peer < nranks; ++peer) {
    if (peer == rank) {
      bases[peer] = PtrToU64(localBuffer);
      runtime->hostDesc.hccl_channel_handle[peer] = 0;
      continue;
    }

    const CommEngine acquireEngine =
        HcclRegisterAcquireEngine(runtime->hcclChannelEngine);
    HcclChannelDesc desc = {};
    ret = BuildHcclChannelDescForEngine(
        runtime, acquireEngine, peer,
        useAcquireMemHandle ? &acquireMemHandle : nullptr, &desc);
    if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      return ret;
    }

    ChannelHandle channel = 0;
    ret = HcclMaybeAddChannelExchangeInfo(runtime, localBufferBytes, nullptr);
    if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      return ret;
    }
    ret = runtime->hcclChannel.channelAcquire(runtime->hcclComm, acquireEngine,
                                              &desc, 1, &channel);
    if (ret != 0 || channel == 0) {
      SetLastError("HcclChannelAcquire legacy CCL buffer for peer " + std::to_string(peer) +
                   " failed with code " + std::to_string(ret) +
                   " requested_engine=" + HcclChannelEngineName(runtime->hcclChannelEngine) +
                   "(" + std::to_string(static_cast<int>(runtime->hcclChannelEngine)) + ")" +
                   " acquire_engine=" + HcclChannelEngineName(acquireEngine) +
                   "(" + std::to_string(static_cast<int>(acquireEngine)) + ")" +
                   " protocol=" + std::to_string(static_cast<int>(desc.channelProtocol)));
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }
    runtime->hcclWindowChannels[peer] = channel;
    runtime->hostDesc.hccl_channel_handle[peer] = static_cast<uint64_t>(channel);

    void *remoteBuffer = nullptr;
    uint64_t remoteBufferBytes = 0;
    ret = runtime->hcclChannel.channelGetHcclBuffer(runtime->hcclComm, channel,
                                                    &remoteBuffer, &remoteBufferBytes);
    if (ret != 0 || remoteBuffer == nullptr || remoteBufferBytes < registerBytes) {
      SetLastError("HcclChannelGetHcclBuffer for peer " + std::to_string(peer) +
                   " failed with code " + std::to_string(ret) +
                   " ptr=" + std::to_string(PtrToU64(remoteBuffer)) +
                   " bytes=" + std::to_string(remoteBufferBytes) +
                   " required=" + std::to_string(registerBytes));
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }
    bases[peer] = PtrToU64(remoteBuffer);
  }

  runtime->windowBytes = bytes;
  runtime->hostDesc.window_bytes = bytes;
  for (uint32_t peer = 0; peer < nranks; ++peer) {
    runtime->hostDesc.peer_signal_base[peer] =
        bases[peer] == 0 ? 0 : bases[peer] + signalOffset;
  }

  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] hccl_channel legacy register_%s rank=%u nranks=%u "
                 "ptr=%p ccl_buffer=%p ccl_bytes=%llu data_bytes=%llu "
                 "signal_offset=%llu signal_bytes=%llu register_bytes=%llu "
                 "requested_engine=%s(%d) acquire_engine=%s(%d) rendezvous=%s\n",
                 what, rank, nranks, localPtr, localBuffer,
                 static_cast<unsigned long long>(localBufferBytes),
                 static_cast<unsigned long long>(bytes),
                 static_cast<unsigned long long>(signalOffset),
                 static_cast<unsigned long long>(signalBytes),
                 static_cast<unsigned long long>(registerBytes),
                 HcclChannelEngineName(runtime->hcclChannelEngine),
                 static_cast<int>(runtime->hcclChannelEngine),
                 HcclChannelEngineName(HcclRegisterAcquireEngine(runtime->hcclChannelEngine)),
                 static_cast<int>(HcclRegisterAcquireEngine(runtime->hcclChannelEngine)),
                 rendezvousId == nullptr ? "<null>" : rendezvousId);
  }

  *windowHandle = kWindowHandleDefault;
  return CopyDescriptorToDevice(runtime);
}

int RegisterHcclChannelBases(TritonAscendGinRuntimeHandle *runtime, void *localPtr, uint64_t bytes,
                             const char *rendezvousId, const char *what, uint64_t *bases,
                             uint64_t *windowHandle) {
  if (runtime == nullptr || localPtr == nullptr || bases == nullptr || windowHandle == nullptr ||
      bytes == 0) {
    SetLastError(std::string("TritonAscendGinRegister") + what + " received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (runtime->hostDesc.backend_kind != TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_CHANNEL ||
      runtime->hcclComm == nullptr) {
    SetLastError(std::string("TritonAscendGinRegister") + what +
                 " requires an HCCL channel communicator");
    return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
  }
  if (HcclChannelEngineUsesLegacyCclBuffer(runtime->hcclChannelEngine)) {
    return RegisterHcclChannelLegacyCclBufferBases(runtime, localPtr, bytes,
                                                   rendezvousId, what, bases,
                                                   windowHandle);
  }

  const bool isSignalWindow = bases == runtime->hostDesc.peer_signal_base;
  uint64_t signalBytes = 0;
  uint64_t signalOffset = 0;
  uint64_t registerBytes = bytes;
  if (!isSignalWindow) {
    signalBytes = HcclSignalWindowBytes(runtime);
    signalOffset = AlignUpU64(bytes, kDefaultSignalStride);
    if (signalOffset < bytes ||
        signalBytes > std::numeric_limits<uint64_t>::max() - signalOffset) {
      SetLastError(std::string("HCCL channel register for ") + what +
                   " overflowed while reserving embedded signal slots");
      return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
    }
    registerBytes = signalOffset + signalBytes;
  }

  const std::string id = SanitizeRendezvousId(rendezvousId);
  std::string tag = "triton_gin_" + std::string(what) + "_" + id;
  CommMem mem = {};
  mem.type = COMM_MEM_TYPE_DEVICE;
  mem.addr = localPtr;
  mem.size = registerBytes;
  HcclMemHandle memHandle = nullptr;
  int ret = runtime->hcclChannel.commMemReg(runtime->hcclComm, tag.c_str(), &mem, &memHandle);
  const char *independentOp = std::getenv("HCCL_INDEPENDENT_OP");
  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] HcclCommMemReg %s rank=%u tag=%s ptr=%p "
                 "bytes=%llu ret=%d mem_handle=%p engine=%d HCCL_INDEPENDENT_OP=%s\n",
                 what, runtime->hostDesc.rank, tag.c_str(), localPtr,
                 static_cast<unsigned long long>(registerBytes), ret, memHandle,
                 static_cast<int>(runtime->hcclChannelEngine),
                 independentOp == nullptr ? "<unset>" : independentOp);
  }
  if (ret == 0 && memHandle == nullptr && !isSignalWindow) {
    if (DebugEnabled()) {
      std::fprintf(stderr,
                   "[triton_ascend_gin] HcclCommMemReg %s returned a null "
                   "mem_handle; falling back to HCCL CCL buffer channel path\n",
                   what);
    }
    return RegisterHcclChannelLegacyCclBufferBases(runtime, localPtr, bytes,
                                                   rendezvousId, what, bases,
                                                   windowHandle);
  }
  if (ret != 0 || memHandle == nullptr) {
    SetLastError(std::string("HcclCommMemReg for ") + what + " failed with code " +
                 std::to_string(ret) + " tag=" + tag +
                 " ptr=" + std::to_string(PtrToU64(localPtr)) +
                 " bytes=" + std::to_string(registerBytes) +
                 " mem_handle=" + std::to_string(PtrToU64(memHandle)) +
                 " engine=" + std::to_string(static_cast<int>(runtime->hcclChannelEngine)) +
                 " HCCL_INDEPENDENT_OP=" +
                 (independentOp == nullptr ? std::string("<unset>") : std::string(independentOp)) +
                 ". Ensure HCCL_INDEPENDENT_OP is set before HcclCommInit.");
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }

  ChannelHandle *channels = isSignalWindow ? runtime->hcclSignalChannels : runtime->hcclWindowChannels;
  if (isSignalWindow) {
    runtime->hcclSignalMemHandle = memHandle;
  } else {
    runtime->hcclWindowMemHandle = memHandle;
  }

  const uint32_t rank = runtime->hostDesc.rank;
  const uint32_t nranks = runtime->hostDesc.nranks;
  for (uint32_t peer = 0; peer < nranks; ++peer) {
    if (peer == rank) {
      bases[peer] = PtrToU64(localPtr);
      continue;
    }

    const CommEngine acquireEngine =
        HcclRegisterAcquireEngine(runtime->hcclChannelEngine);
    HcclChannelDesc desc = {};
    ret = BuildHcclChannelDescForEngine(runtime, acquireEngine, peer, &memHandle,
                                        &desc);
    if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      return ret;
    }

    ChannelHandle channel = 0;
    ret = HcclMaybeAddChannelExchangeInfo(runtime, 0, nullptr);
    if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      return ret;
    }
    ret = runtime->hcclChannel.channelAcquire(runtime->hcclComm, acquireEngine,
                                              &desc, 1, &channel);
    if (ret != 0 || channel == 0) {
      SetLastError(std::string("HcclChannelAcquire for ") + what + " peer " +
                   std::to_string(peer) + " failed with code " + std::to_string(ret) +
                   " requested_engine=" + HcclChannelEngineName(runtime->hcclChannelEngine) +
                   "(" + std::to_string(static_cast<int>(runtime->hcclChannelEngine)) + ")" +
                   " acquire_engine=" + HcclChannelEngineName(acquireEngine) +
                   "(" + std::to_string(static_cast<int>(acquireEngine)) + ")" +
                   " protocol=" + std::to_string(static_cast<int>(desc.channelProtocol)));
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }
    channels[peer] = channel;

    uint32_t memNum = 0;
    CommMem *remoteMems = nullptr;
    char **memTags = nullptr;
    ret = runtime->hcclChannel.channelGetRemoteMems(runtime->hcclComm, channel, &memNum,
                                                    &remoteMems, &memTags);
    if (ret != 0 || memNum == 0 || remoteMems == nullptr) {
      SetLastError(std::string("HcclChannelGetRemoteMems for ") + what + " peer " +
                   std::to_string(peer) + " failed with code " + std::to_string(ret) +
                   " mem_num=" + std::to_string(memNum) +
                   ". Ensure HCCL_INDEPENDENT_OP is set before HcclCommInit.");
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }

    uint32_t selected = memNum - 1;
    if (memTags != nullptr) {
      for (uint32_t idx = 0; idx < memNum; ++idx) {
        if (memTags[idx] != nullptr && tag == memTags[idx]) {
          selected = idx;
          break;
        }
      }
    }
    if (remoteMems[selected].addr == nullptr) {
      SetLastError(std::string("HcclChannelGetRemoteMems for ") + what + " peer " +
                   std::to_string(peer) + " returned null addr at index " +
                   std::to_string(selected));
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }
    bases[peer] = PtrToU64(remoteMems[selected].addr);
  }

  if (!isSignalWindow) {
    for (uint32_t peer = 0; peer < nranks; ++peer) {
      runtime->hostDesc.peer_signal_base[peer] =
          bases[peer] == 0 ? 0 : bases[peer] + signalOffset;
    }
  }

  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] hccl_channel register_%s rank=%u nranks=%u "
                 "ptr=%p bytes=%llu signal_offset=%llu signal_bytes=%llu "
                  "register_bytes=%llu requested_engine=%s(%d) acquire_engine=%s(%d) "
                  "tag=%s\n",
                  what, rank, nranks, localPtr, static_cast<unsigned long long>(bytes),
                  static_cast<unsigned long long>(signalOffset),
                  static_cast<unsigned long long>(signalBytes),
                  static_cast<unsigned long long>(registerBytes),
                  HcclChannelEngineName(runtime->hcclChannelEngine),
                  static_cast<int>(runtime->hcclChannelEngine),
                  HcclChannelEngineName(HcclRegisterAcquireEngine(runtime->hcclChannelEngine)),
                  static_cast<int>(HcclRegisterAcquireEngine(runtime->hcclChannelEngine)),
                  tag.c_str());
  }

  *windowHandle = kWindowHandleDefault;
  return CopyDescriptorToDevice(runtime);
}
#endif

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

  const bool isSignalWindow = bases == runtime->hostDesc.peer_signal_base;
  void **storedSymWindow = isSignalWindow ? &runtime->hcclSignalSymWindow : &runtime->hcclDataSymWindow;
  int ret = DeregisterHcclSymWindow(runtime, storedSymWindow, what);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    return ret;
  }

  void *registeredSymWindow = nullptr;
  uint64_t signalBytes = 0;
  uint64_t signalOffset = 0;
  uint64_t registerBytes = bytes;
  if (!isSignalWindow) {
    signalBytes = HcclSignalWindowBytes(runtime);
    signalOffset = AlignUpU64(bytes, kDefaultSignalStride);
    if (signalOffset < bytes ||
        signalBytes > std::numeric_limits<uint64_t>::max() - signalOffset) {
      SetLastError(std::string("HcclCommSymWinRegister for ") + what +
                   " overflowed while reserving embedded signal slots");
      return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
    }
    registerBytes = signalOffset + signalBytes;
  }
  const uint64_t registerFloorBytes = HcclSymWindowRegisterBytes(registerBytes);
  uint32_t flag = static_cast<uint32_t>(GetEnvInt("TRITON_ASCEND_GIN_HCCL_SYM_WIN_FLAG", 0));
  ret = runtime->hccl.commSymWinRegister(runtime->hcclComm, localPtr, registerFloorBytes,
                                         &registeredSymWindow, flag);
  if (ret != 0 || registeredSymWindow == nullptr) {
    SetLastError(std::string("HcclCommSymWinRegister for ") + what + " failed with code " +
                 std::to_string(ret) + " ptr=" + std::to_string(PtrToU64(localPtr)) +
                 " bytes=" + std::to_string(bytes) +
                 " signal_offset=" + std::to_string(signalOffset) +
                 " signal_bytes=" + std::to_string(signalBytes) +
                 " register_bytes=" + std::to_string(registerBytes) +
                 " register_floor_bytes=" + std::to_string(registerFloorBytes) +
                 " flag=" + std::to_string(flag) + HcclSymWinRegisterHint(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }

  void *symWindow = registeredSymWindow;
  size_t offset = 0;
  ret = runtime->hccl.commSymWinGet(runtime->hcclComm, localPtr, static_cast<size_t>(registerFloorBytes),
                                   &symWindow, &offset);
  if (ret != 0 || symWindow == nullptr) {
    (void)DeregisterHcclSymWindow(runtime, &registeredSymWindow, what);
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
    if (runtime->hccl.symWinGetPeerPointer == nullptr) {
      // CANN 9.1.0 exposes HcclCommSymWinGet/Register in libhcomm, but some
      // packages ship only the HcclSymWinGetPeerPointer declaration. Try the
      // symmetric-VA contract first; the end-to-end smoke test validates
      // whether the platform maps peer windows at the same VA.
      bases[peer] = PtrToU64(localPtr);
      continue;
    }
    void *peerPtr = nullptr;
    ret = runtime->hccl.symWinGetPeerPointer(symWindow, offset, peer, &peerPtr);
    if (ret != 0 || peerPtr == nullptr) {
      (void)DeregisterHcclSymWindow(runtime, &registeredSymWindow, what);
      SetLastError(std::string("HcclSymWinGetPeerPointer for ") + what + " peer " +
                   std::to_string(peer) + " failed with code " + std::to_string(ret));
      return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    }
    bases[peer] = PtrToU64(peerPtr);
  }

  if (!isSignalWindow) {
    for (uint32_t peer = 0; peer < nranks; ++peer) {
      runtime->hostDesc.peer_signal_base[peer] =
          bases[peer] == 0 ? 0 : bases[peer] + signalOffset;
    }
  }

  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] hccl register_%s rank=%u nranks=%u ptr=%p bytes=%llu "
                 "signal_offset=%llu signal_bytes=%llu register_bytes=%llu "
                 "register_floor_bytes=%llu flag=%u offset=%llu\n",
                 what, rank, nranks, localPtr, static_cast<unsigned long long>(bytes),
                 static_cast<unsigned long long>(signalOffset),
                 static_cast<unsigned long long>(signalBytes),
                 static_cast<unsigned long long>(registerBytes),
                 static_cast<unsigned long long>(registerFloorBytes), flag,
                 static_cast<unsigned long long>(offset));
  }

  *storedSymWindow = registeredSymWindow;
  *windowHandle = kWindowHandleDefault;
  return CopyDescriptorToDevice(runtime);
}

TritonAscendGinRuntimeHandle *AsHandle(TritonAscendGinHandle handle) {
  return static_cast<TritonAscendGinRuntimeHandle *>(handle);
}

}  // namespace

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinCreateFromTileXR(
    TritonAscendGinTileXRHandle tilexrComm, const TritonAscendGinTileXROptions *options,
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

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinRefreshFromTileXR(
    TritonAscendGinHandle handle) {
  return FillDescriptorFromTileXR(AsHandle(handle));
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinCreateFromHcclPeerMem(
    TritonAscendGinHcclHandle hcclComm, const TritonAscendGinHcclPeerMemOptions *options,
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

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinCreateFromHcclChannel(
    TritonAscendGinHcclHandle hcclComm, const TritonAscendGinHcclChannelOptions *options,
    TritonAscendGinHandle *handle) {
#if defined(__linux__)
  if (hcclComm == nullptr || handle == nullptr) {
    SetLastError("TritonAscendGinCreateFromHcclChannel received null input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  *handle = nullptr;

  auto *next = new (std::nothrow) TritonAscendGinRuntimeHandle();
  if (next == nullptr) {
    SetLastError("failed to allocate HCCL channel runtime handle");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  next->hcclComm = hcclComm;
  if (options != nullptr) {
    next->windowBytes = options->window_bytes == 0 ? kDefaultTileXRWindowBytes : options->window_bytes;
    next->signalStride = options->signal_stride == 0 ? kDefaultSignalStride : options->signal_stride;
    next->signalSlots = options->signal_slots == 0 ? kDefaultSignalSlots : options->signal_slots;
    next->hcclChannelEngine = static_cast<CommEngine>(options->engine);
  }

  int ret = LoadHcclChannelSymbols(options == nullptr ? nullptr : options->hccl_library_path,
                                   &next->hcclChannel);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    delete next;
    return ret;
  }

  ret = FillDescriptorFromHcclChannel(next);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    TritonAscendGinDestroy(next);
    return ret;
  }

  *handle = next;
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
#else
  (void)hcclComm;
  (void)options;
  (void)handle;
  SetLastError("Triton Ascend HCCL channel bridge is only implemented for Linux hosts");
  return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
#endif
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinProbeHcclChannel(
    TritonAscendGinHcclHandle hcclComm, const TritonAscendGinHcclChannelOptions *options,
    TritonAscendGinHcclChannelProbe *probe) {
#if defined(__linux__)
  if (hcclComm == nullptr || probe == nullptr) {
    SetLastError("TritonAscendGinProbeHcclChannel received null input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  std::memset(probe, 0, sizeof(*probe));
  probe->struct_size = sizeof(*probe);
  probe->engine = options == nullptr ? static_cast<uint32_t>(COMM_ENGINE_AIV) : options->engine;

  TritonAscendGinRuntimeHandle runtime;
  runtime.hcclComm = hcclComm;
  runtime.hcclChannelEngine = static_cast<CommEngine>(probe->engine);
  if (options != nullptr) {
    runtime.windowBytes = options->window_bytes == 0 ? kDefaultTileXRWindowBytes : options->window_bytes;
    runtime.signalStride = options->signal_stride == 0 ? kDefaultSignalStride : options->signal_stride;
    runtime.signalSlots = options->signal_slots == 0 ? kDefaultSignalSlots : options->signal_slots;
  }

  int ret = LoadHcclChannelSymbols(options == nullptr ? nullptr : options->hccl_library_path,
                                   &runtime.hcclChannel);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    probe->first_error_status = ret;
    return ret;
  }
  probe->exchange_info_enabled = HcclChannelExchangeInfoEnabled() ? 1u : 0u;
  probe->exchange_info_symbol_available =
      runtime.hcclChannel.commAddExchangeInfo == nullptr ? 0u : 1u;

  uint32_t rank = 0;
  uint32_t nranks = 0;
  ret = runtime.hcclChannel.getRankId(runtime.hcclComm, &rank);
  if (ret != 0) {
    SetLastError("HcclGetRankId failed with code " + std::to_string(ret));
    probe->first_error_status = TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  ret = runtime.hcclChannel.getRankSize(runtime.hcclComm, &nranks);
  if (ret != 0) {
    SetLastError("HcclGetRankSize failed with code " + std::to_string(ret));
    probe->first_error_status = TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  if (nranks == 0 || nranks > kTileXRMaxRanks || rank >= nranks) {
    SetLastError("HCCL channel probe rank metadata is invalid");
    probe->first_error_status = TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  runtime.hostDesc.rank = rank;
  runtime.hostDesc.nranks = nranks;
  probe->rank = rank;
  probe->nranks = nranks;

  void *localBuffer = nullptr;
  uint64_t localBufferBytes = 0;
  ret = runtime.hcclChannel.getHcclBuffer(runtime.hcclComm, &localBuffer, &localBufferBytes);
  probe->ccl_buffer_status = ret;
  probe->ccl_buffer_ptr = PtrToU64(localBuffer);
  probe->ccl_buffer_bytes = localBufferBytes;
  if (ret != 0) {
    HcclChannelProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR, rank);
  }

  HcclMemHandle acquireMemHandle = nullptr;
  HcclMemHandle *acquireMemHandlePtr = nullptr;
  if (runtime.hcclChannelEngine == COMM_ENGINE_AIV) {
    ret = EnsureHcclAivCommInfo(&runtime, "probe");
    probe->aiv_comm_info_status = ret;
    probe->aiv_comm_info_ptr = PtrToU64(runtime.hcclAivCommInfo);
    probe->aiv_comm_info_bytes = runtime.hcclAivCommInfoRegistered ? kHcclAivCommInfoBytes : 0;
    probe->aiv_comm_info_mem_handle = PtrToU64(runtime.hcclAivCommInfoMemHandle);
    if (ret == TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      int layoutRet = ProbeHcclAivCommInfoLocalLayout(&runtime, localBuffer, probe);
      if (layoutRet != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
        HcclChannelProbeRecordFirstError(probe, layoutRet, rank);
      }
      if (HcclProbeAivDescriptorViaCpuEnabled()) {
        int descriptorRet = ProbeHcclAivDescriptorViaCpuChannels(
            &runtime, localBuffer, localBufferBytes, probe);
        if (descriptorRet != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
          HcclChannelProbeRecordFirstError(probe, descriptorRet, rank);
        }
      }
      acquireMemHandle = runtime.hcclAivCommInfoMemHandle;
      acquireMemHandlePtr = &acquireMemHandle;
    } else {
      HcclChannelProbeRecordFirstError(probe, ret, rank);
    }
  }

  uint32_t *layers = nullptr;
  uint32_t layerNum = 0;
  ret = runtime.hcclChannel.rankGraphGetLayers(runtime.hcclComm, &layers, &layerNum);
  if (ret != 0 || layers == nullptr || layerNum == 0) {
    SetLastError("HcclRankGraphGetLayers failed with code " + std::to_string(ret) +
                 " layer_num=" + std::to_string(layerNum));
    probe->first_error_status = TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
    return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
  }
  probe->layer_count = layerNum;

  for (uint32_t peer = 0; peer < nranks; ++peer) {
    if (peer == rank) {
      continue;
    }
    probe->peer_mask |= HcclPeerBit(peer);
    bool sawValidProtocol = false;
    for (uint32_t layerIdx = 0; layerIdx < layerNum; ++layerIdx) {
      CommLink *links = nullptr;
      uint32_t linkNum = 0;
      int linkRet = runtime.hcclChannel.rankGraphGetLinks(
          runtime.hcclComm, layers[layerIdx], rank, peer, &links, &linkNum);
      if (linkRet != 0 || links == nullptr || linkNum == 0) {
        HcclChannelProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR, peer);
        continue;
      }
      for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
        CommProtocol protocol = links[linkIdx].linkAttr.linkProtocol;
        HcclChannelProbeMarkProtocol(probe, peer, protocol);
        sawValidProtocol = sawValidProtocol || HcclLinkProtocolIsValid(protocol);
      }
    }
    if (!sawValidProtocol) {
      HcclChannelProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR, peer);
    }
  }

  const char *probeAcquire = std::getenv("TRITON_ASCEND_GIN_HCCL_PROBE_ACQUIRE");
  const bool doAcquire = probeAcquire == nullptr || probeAcquire[0] == '\0' || probeAcquire[0] != '0';
  if (doAcquire) {
    for (uint32_t peer = 0; peer < nranks; ++peer) {
      if (peer == rank) {
        continue;
      }
      uint64_t aivPeerMask = probe->ub_mem_peer_mask | probe->pcie_peer_mask;
      if (HcclAivAllowExperimentalHccs()) {
        aivPeerMask |= probe->hccs_peer_mask;
      }
      if (runtime.hcclChannelEngine == COMM_ENGINE_AIV &&
          (aivPeerMask & HcclPeerBit(peer)) == 0) {
        HcclChannelProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR, peer);
        continue;
      }

      HcclChannelDesc desc = {};
      ret = BuildHcclChannelDesc(&runtime, peer, acquireMemHandlePtr, &desc);
      if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
        HcclChannelProbeRecordFirstError(probe, ret, peer);
        continue;
      }
      ret = HcclMaybeAddChannelExchangeInfo(&runtime, localBufferBytes, probe);
      if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
        HcclChannelProbeRecordFirstError(probe, ret, peer);
        continue;
      }
      ChannelHandle channel = 0;
      ret = runtime.hcclChannel.channelAcquire(runtime.hcclComm, runtime.hcclChannelEngine,
                                               &desc, 1, &channel);
      if (ret == 0 && channel != 0) {
        probe->acquire_peer_mask |= HcclPeerBit(peer);
      } else {
        HcclChannelProbeRecordFirstAcquire(probe, ret, peer, runtime.hcclChannelEngine,
                                           desc, 1, channel);
        HcclChannelProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR, peer);
      }
    }
  }

  uint64_t aivReachablePeerMask = probe->ub_mem_peer_mask | probe->pcie_peer_mask;
  if (HcclAivAllowExperimentalHccs()) {
    aivReachablePeerMask |= probe->hccs_peer_mask;
  }
  if (runtime.hcclChannelEngine == COMM_ENGINE_AIV &&
      (aivReachablePeerMask & probe->peer_mask) != probe->peer_mask) {
    HcclChannelProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR, rank);
  }

  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
#else
  (void)hcclComm;
  (void)options;
  (void)probe;
  SetLastError("Triton Ascend HCCL channel probe is only implemented for Linux hosts");
  return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
#endif
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinProbeHcclAlgBuffers(
    const char *hcclAlgLibraryPath, uint32_t device,
    TritonAscendGinHcclAlgBufferProbe *probe) {
#if defined(__linux__)
  if (probe == nullptr) {
    SetLastError("TritonAscendGinProbeHcclAlgBuffers received null probe");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  std::memset(probe, 0, sizeof(*probe));
  probe->struct_size = sizeof(*probe);
  probe->device = device;

  aclError setDeviceErr = aclrtSetDevice(static_cast<int32_t>(device));
  probe->acl_set_device_status = static_cast<int32_t>(setDeviceErr);
  if (setDeviceErr != ACL_SUCCESS) {
    SetAclError("aclrtSetDevice", setDeviceErr);
    HcclAlgBufferProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR);
    return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
  }

  HcclAlgBufferSymbols symbols;
  int ret = LoadHcclAlgBufferSymbols(hcclAlgLibraryPath, &symbols);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    HcclAlgBufferProbeRecordFirstError(probe, ret);
    CloseHcclAlgBufferSymbols(&symbols);
    return ret;
  }

  constexpr size_t kManagerStorageBytes = 64ull * 1024ull;
  void *storage = nullptr;
  int allocRet = posix_memalign(&storage, 64, kManagerStorageBytes);
  if (allocRet != 0 || storage == nullptr) {
    SetLastError("posix_memalign for HCCL CCLBufferManager storage failed: " +
                 std::string(std::strerror(allocRet)));
    CloseHcclAlgBufferSymbols(&symbols);
    return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
  }
  std::memset(storage, 0, kManagerStorageBytes);
  probe->manager_storage_bytes = kManagerStorageBytes;

  symbols.ctor(storage);

  void *cclPtr = nullptr;
  unsigned long cclBytes = 0;
  ret = symbols.getIndependentOpCclBuffer(storage, cclPtr, cclBytes);
  probe->get_independent_ccl_before_status = ret;
  probe->ccl_buffer_ptr_before = PtrToU64(cclPtr);
  probe->ccl_buffer_bytes_before = static_cast<uint64_t>(cclBytes);
  if (ret != 0 || cclPtr == nullptr || cclBytes == 0) {
    HcclAlgBufferProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR);
  }

  ret = symbols.createCommAivBuffer(storage, true);
  probe->create_comm_aiv_buffer_status = ret;
  if (ret != 0) {
    HcclAlgBufferProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR);
  }

  void *inAivDeviceMem = symbols.getInAivOpbaseBuffer(storage);
  void *outAivDeviceMem = symbols.getOutAivOpbaseBuffer(storage);
  probe->in_aiv_opbase_devmem_ptr = PtrToU64(inAivDeviceMem);
  probe->in_aiv_opbase_ptr = HcclAlgDeviceMemBase(inAivDeviceMem);
  probe->in_aiv_opbase_bytes = probe->in_aiv_opbase_ptr == 0 ? 0 : kHcclAivInOpbaseBytes;
  probe->out_aiv_opbase_devmem_ptr = PtrToU64(outAivDeviceMem);
  probe->out_aiv_opbase_ptr = HcclAlgDeviceMemBase(outAivDeviceMem);
  probe->out_aiv_opbase_bytes = probe->out_aiv_opbase_ptr == 0 ? 0 : kHcclAivOutOpbaseBytes;

  ret = symbols.createCommInfoAivBuffer(storage);
  probe->create_comm_info_aiv_buffer_status = ret;
  if (ret != 0) {
    HcclAlgBufferProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR);
  }

  void *commInfoDeviceMem = symbols.getAivCommInfoBuffer(storage);
  probe->aiv_comm_info_devmem_ptr = PtrToU64(commInfoDeviceMem);
  probe->aiv_comm_info_ptr = HcclAlgDeviceMemBase(commInfoDeviceMem);
  probe->aiv_comm_info_bytes = probe->aiv_comm_info_ptr == 0 ? 0 : kHcclAivCommInfoBytes;

  cclPtr = nullptr;
  cclBytes = 0;
  ret = symbols.getIndependentOpCclBuffer(storage, cclPtr, cclBytes);
  probe->get_independent_ccl_after_status = ret;
  probe->ccl_buffer_ptr_after = PtrToU64(cclPtr);
  probe->ccl_buffer_bytes_after = static_cast<uint64_t>(cclBytes);
  if (ret != 0 || cclPtr == nullptr || cclBytes == 0) {
    HcclAlgBufferProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR);
  }

  ret = symbols.clearCommAivBuffer(storage);
  probe->clear_comm_aiv_buffer_status = ret;
  if (ret != 0) {
    HcclAlgBufferProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR);
  }
  ret = symbols.releaseCommAivBuffer(storage);
  probe->release_comm_aiv_buffer_status = ret;
  if (ret != 0) {
    HcclAlgBufferProbeRecordFirstError(probe, TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR);
  }

  symbols.dtor(storage);
  std::free(storage);
  CloseHcclAlgBufferSymbols(&symbols);
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
#else
  (void)hcclAlgLibraryPath;
  (void)device;
  (void)probe;
  SetLastError("Triton Ascend HCCL alg buffer probe is only implemented for Linux hosts");
  return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
#endif
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinGetDevComm(TritonAscendGinHandle handle,
                                                                        uint64_t *devComm) {
  auto *runtime = AsHandle(handle);
  if (runtime == nullptr || devComm == nullptr || runtime->devDesc == nullptr) {
    SetLastError("TritonAscendGinGetDevComm received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  *devComm = PtrToU64(runtime->devDesc);
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinGetHostComm(
    TritonAscendGinHandle handle, TritonAscendGinDev *hostComm) {
  auto *runtime = AsHandle(handle);
  if (runtime == nullptr || hostComm == nullptr) {
    SetLastError("TritonAscendGinGetHostComm received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  std::memcpy(hostComm, &runtime->hostDesc, sizeof(*hostComm));
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinGetHcclBuffer(
    TritonAscendGinHandle handle, uint64_t *ptr, uint64_t *bytes) {
#if defined(__linux__)
  auto *runtime = AsHandle(handle);
  if (runtime == nullptr || ptr == nullptr || bytes == nullptr) {
    SetLastError("TritonAscendGinGetHcclBuffer received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (runtime->hostDesc.backend_kind != TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_CHANNEL ||
      runtime->hcclComm == nullptr) {
    SetLastError("TritonAscendGinGetHcclBuffer requires an HCCL channel communicator");
    return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
  }
  if (!HcclChannelSymbolsComplete(runtime->hcclChannel)) {
    SetLastError("HCCL channel symbols are unavailable");
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }
  void *buffer = nullptr;
  uint64_t bufferBytes = 0;
  int ret = runtime->hcclChannel.getHcclBuffer(runtime->hcclComm, &buffer, &bufferBytes);
  if (ret != 0 || buffer == nullptr || bufferBytes == 0) {
    SetLastError("HcclGetHcclBuffer failed with code " + std::to_string(ret) +
                 " ptr=" + std::to_string(PtrToU64(buffer)) +
                 " bytes=" + std::to_string(bufferBytes));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  *ptr = PtrToU64(buffer);
  *bytes = bufferBytes;
  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] HcclGetHcclBuffer rank=%u ptr=%p bytes=%llu\n",
                 runtime->hostDesc.rank, buffer,
                 static_cast<unsigned long long>(bufferBytes));
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
#else
  (void)handle;
  (void)ptr;
  (void)bytes;
  SetLastError("Triton Ascend HCCL buffer query is only implemented for Linux hosts");
  return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
#endif
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinGetHcclAivOpbaseBuffer(
    TritonAscendGinHandle handle, uint32_t which, uint64_t *ptr, uint64_t *bytes) {
#if defined(__linux__)
  auto *runtime = AsHandle(handle);
  if (runtime == nullptr || ptr == nullptr || bytes == nullptr) {
    SetLastError("TritonAscendGinGetHcclAivOpbaseBuffer received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (runtime->hostDesc.backend_kind != TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_CHANNEL ||
      runtime->hcclComm == nullptr) {
    SetLastError("TritonAscendGinGetHcclAivOpbaseBuffer requires an HCCL channel communicator");
    return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
  }
  if (runtime->hcclChannelEngine != COMM_ENGINE_AIV) {
    SetLastError("TritonAscendGinGetHcclAivOpbaseBuffer requires the HCCL AIV channel engine");
    return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
  }
  int ret = EnsureHcclAivOpbaseBuffers(runtime);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    return ret;
  }
  if (which == 0) {
    *ptr = runtime->hcclInAivOpbasePtr;
    *bytes = runtime->hcclInAivOpbaseBytes;
  } else if (which == 1) {
    *ptr = runtime->hcclOutAivOpbasePtr;
    *bytes = runtime->hcclOutAivOpbaseBytes;
  } else {
    SetLastError("TritonAscendGinGetHcclAivOpbaseBuffer selector must be 0(in) or 1(out)");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  return (*ptr == 0 || *bytes == 0) ? TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR
                                    : TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
#else
  (void)handle;
  (void)which;
  (void)ptr;
  (void)bytes;
  SetLastError("Triton Ascend HCCL AIV opbase query is only implemented for Linux hosts");
  return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
#endif
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinHcclAivAllGather(
    TritonAscendGinHandle handle, void *sendBuf, void *recvBuf,
    uint64_t sendCount, uint32_t dataType, void *stream,
    const char *rendezvousId) {
#if defined(__linux__)
  auto *runtime = AsHandle(handle);
  if (runtime == nullptr || sendBuf == nullptr || recvBuf == nullptr ||
      stream == nullptr || sendCount == 0) {
    SetLastError("TritonAscendGinHcclAivAllGather received invalid input");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }
  if (runtime->hostDesc.backend_kind != TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_CHANNEL ||
      runtime->hcclComm == nullptr) {
    SetLastError("TritonAscendGinHcclAivAllGather requires an HCCL channel communicator");
    return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
  }
  if (runtime->hcclChannelEngine != COMM_ENGINE_AIV) {
    SetLastError("TritonAscendGinHcclAivAllGather requires channel engine AIV");
    return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
  }
  if (runtime->hostDesc.nranks < 2 ||
      runtime->hostDesc.nranks > kHcclAivDirectMaxRanks) {
    SetLastError("TritonAscendGinHcclAivAllGather supports rank_size in [2," +
                 std::to_string(kHcclAivDirectMaxRanks) + "]");
    return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
  }

  int ret = EnsureHcclAivOpbaseBuffers(runtime);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    return ret;
  }
  if (!HcclAivDirectSymbolsComplete(runtime->hcclAlgBuffer)) {
    SetLastError("HCCL legacy AIV launch symbols were not resolved from libhccl_alg.so");
    return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
  }

  void *buffersIn[kHcclAivDirectMaxRanks] = {};
  void *buffersOut[kHcclAivDirectMaxRanks] = {};
  ret = ExchangeHcclAivDirectBuffers(runtime, rendezvousId, buffersIn, buffersOut);
  if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
    return ret;
  }

  const std::string commName = HcclCommNameOrFallback(runtime, rendezvousId);
  const std::string commTag = commName + "_opbase";
  const uint32_t numBlocks = runtime->hostDesc.nranks;
  const int32_t aivTag = NextHcclAivDirectTag(runtime);

  HcclAivOpArgsCompat opArgs = {};
  opArgs.cmdType = kHcclCmdAllGather;
  opArgs.input = sendBuf;
  opArgs.output = recvBuf;
  opArgs.count = sendCount;
  opArgs.dataType = dataType;
  opArgs.op = kHcclReduceReserved;
  opArgs.root = 0;
  opArgs.isOpBase = true;

  HcclAivTopoArgsCompat topoArgs = {};
  topoArgs.rank = runtime->hostDesc.rank;
  topoArgs.rankSize = runtime->hostDesc.nranks;
  topoArgs.devId = kHcclAivDevIdDefault;
  topoArgs.serverId = 0;
  topoArgs.serverNum = 1;
  topoArgs.devType = kHcclAivDevType910B;
  topoArgs.identify = commName;

  HcclAivResourceArgsCompat resourceArgs = {};
  resourceArgs.commTag = commTag;
  resourceArgs.stream = reinterpret_cast<aclrtStream>(stream);
  resourceArgs.buffersIn = buffersIn;
  resourceArgs.buffersOut = buffersOut;
  resourceArgs.bufferSize = runtime->hcclInAivOpbaseBytes;
  resourceArgs.numBlocks = numBlocks;
  resourceArgs.aivTag = aivTag;

  HcclAivAlgArgsCompat algArgs = {};
  algArgs.execTimeOut = kHcclAivExecTimeout;
  algArgs.execTimeOutSet = true;
  HcclAivProfilingInfoCompat profilingInfo = {};

  ret = runtime->hcclAlgBuffer.registerKernel(kHcclAivDevType910B);
  if (ret != 0) {
    SetLastError("hccl::RegisterKernel(DEV_TYPE_910B) failed with code " +
                 std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] HCCL AIV direct allgather rank=%u "
                 "nranks=%u count=%llu dtype=%u tag=%d send=%p recv=%p "
                 "in0=%p out0=%p in1=%p out1=%p\n",
                 runtime->hostDesc.rank, runtime->hostDesc.nranks,
                 static_cast<unsigned long long>(sendCount), dataType, aivTag,
                 sendBuf, recvBuf, buffersIn[0], buffersOut[0],
                 buffersIn[1], buffersOut[1]);
  }

  ret = runtime->hcclAlgBuffer.clearAivSyncBuf(
      buffersOut, resourceArgs, topoArgs, algArgs);
  if (ret != 0) {
    SetLastError("hccl::ClearAivSyncBuf failed with code " + std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  ret = runtime->hcclAlgBuffer.executeKernelLaunch(
      opArgs, topoArgs, resourceArgs, algArgs, profilingInfo);
  if (ret != 0) {
    SetLastError("hccl::ExecuteKernelLaunch(allgather) failed with code " +
                 std::to_string(ret));
    return TRITON_ASCEND_GIN_RUNTIME_HCCL_ERROR;
  }
  return TRITON_ASCEND_GIN_RUNTIME_SUCCESS;
#else
  (void)handle;
  (void)sendBuf;
  (void)recvBuf;
  (void)sendCount;
  (void)dataType;
  (void)stream;
  (void)rendezvousId;
  SetLastError("Triton Ascend HCCL AIV allgather is only implemented for Linux hosts");
  return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
#endif
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinRegisterWindow(
    TritonAscendGinHandle handle, void *localPtr, uint64_t bytes, const char *rendezvousId,
    uint64_t *windowHandle) {
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
#if defined(__linux__)
  if (runtime->hostDesc.backend_kind == TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_CHANNEL) {
    runtime->windowBytes = bytes;
    runtime->hostDesc.window_bytes = bytes;
    return RegisterHcclChannelBases(runtime, localPtr, bytes, rendezvousId, "window",
                                    runtime->hostDesc.peer_window_base, windowHandle);
  }
#endif

  ClearRegisteredUserWindow(runtime);

  char localKey[kAclIpcKeyBytes] = {};
  uint64_t localWindowOffset = 0;
  uint64_t localExportBytes = 0;
  uint32_t localPid = 0;
  bool localRtIpc = false;
  const char *tilexrIpcMode = std::getenv("TRITON_ASCEND_GIN_TILEXR_IPC_MODE");
  const bool forceRtIpc =
      tilexrIpcMode != nullptr &&
      (std::strcmp(tilexrIpcMode, "rt") == 0 || std::strcmp(tilexrIpcMode, "rts") == 0);
  const bool forceAclIpc =
      tilexrIpcMode != nullptr &&
      (std::strcmp(tilexrIpcMode, "acl") == 0 || std::strcmp(tilexrIpcMode, "legacy") == 0);
  aclError err = ACL_ERROR_INVALID_PARAM;
  int rtStatus = TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
  if (!forceAclIpc) {
    rtStatus = ExportUserWindowRtIpc(localPtr, bytes, localKey, sizeof(localKey),
                                     &localWindowOffset, &localExportBytes, &localPid);
    if (rtStatus == TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      localRtIpc = true;
      err = ACL_SUCCESS;
    } else if (forceRtIpc) {
      return rtStatus;
    }
  }
  if (!localRtIpc) {
    err = ExportUserWindowKey(localPtr, bytes, localKey, sizeof(localKey),
                              &localWindowOffset, &localExportBytes);
    if (err != ACL_SUCCESS && !forceAclIpc &&
        (static_cast<int>(err) == kAclRtFeatureNotSupport ||
         static_cast<int>(err) == kAclRtIpcExportFailed)) {
      return rtStatus;
    }
  }
  if (DebugEnabled()) {
    int32_t deviceId = -1;
    aclError devErr = aclrtGetDevice(&deviceId);
    std::fprintf(stderr,
                 "[triton_ascend_gin] rank=%u register_window ptr=%p bytes=%llu offset=%llu "
                 "export_bytes=%llu device=%d get_device_err=%d export_err=%d rt_ipc=%d "
                 "pid=%u key=%s\n",
                 runtime->hostDesc.rank, localPtr, static_cast<unsigned long long>(bytes),
                 static_cast<unsigned long long>(localWindowOffset),
                 static_cast<unsigned long long>(localExportBytes),
                 deviceId, static_cast<int>(devErr), static_cast<int>(err),
                 localRtIpc ? 1 : 0, localPid, localKey);
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
  const std::string localPidReadyPath = WindowPidReadyPath(dir, id, rank);
  (void)std::remove(localPath.c_str());
  (void)std::remove(localReadyPath.c_str());
  (void)std::remove(localPidReadyPath.c_str());
  std::ostringstream localInfo;
  if (localRtIpc) {
    localInfo << "rt " << localKey << " " << localWindowOffset << " " << bytes << " "
              << localPid;
  } else {
    localInfo << "acl " << localKey << " " << localWindowOffset << " " << bytes;
  }
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

  const bool useRtIpc = windowInfos[rank].rtIpc;
  for (uint32_t peer = 0; peer < nranks; ++peer) {
    if (windowInfos[peer].rtIpc != useRtIpc) {
      SetLastError("mixed TileXR user-window IPC modes are not supported for rendezvous id " + id);
      return TRITON_ASCEND_GIN_RUNTIME_INVALID_VALUE;
    }
  }

  if (useRtIpc) {
    int ret = SetRtIpcWindowPeerPids(localKey, windowInfos, nranks, rank);
    if (ret != TRITON_ASCEND_GIN_RUNTIME_SUCCESS) {
      return ret;
    }
    if (!WriteTextFile(localPidReadyPath, "ready")) {
      SetLastError("failed to write window rendezvous pid-ready file: " + localPidReadyPath);
      return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
    }
    while (true) {
      bool ready = true;
      for (uint32_t peer = 0; peer < nranks; ++peer) {
        std::string ignored;
        if (!ReadTextFile(WindowPidReadyPath(dir, id, peer), &ignored)) {
          ready = false;
          break;
        }
      }
      if (ready) {
        break;
      }
      if (std::chrono::steady_clock::now() > deadline) {
        SetLastError("timeout waiting for window pid rendezvous id " + id);
        return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
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
    int importRet = 0;
    if (useRtIpc) {
      if (!EnsureRtsIpcSymbols()) {
        ClearRegisteredUserWindow(runtime);
        return TRITON_ASCEND_GIN_RUNTIME_NOT_FOUND;
      }
      importRet = g_rtsIpcSymbols.openMemory(&peerPtr, windowInfos[peer].key.c_str());
      err = importRet == 0 ? ACL_SUCCESS : static_cast<aclError>(importRet);
    } else {
      err = aclrtIpcMemImportByKey(&peerPtr, windowInfos[peer].key.c_str(),
                                   ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS);
      importRet = static_cast<int>(err);
    }
    if (DebugEnabled()) {
      int32_t deviceId = -1;
      aclError devErr = aclrtGetDevice(&deviceId);
      std::fprintf(stderr,
                   "[triton_ascend_gin] rank=%u import_window peer=%u device=%d get_device_err=%d "
                   "import_err=%d rt_ipc=%d peer_ptr=%p offset=%llu key=%s\n",
                   rank, peer, deviceId, static_cast<int>(devErr), static_cast<int>(err),
                   useRtIpc ? 1 : 0, peerPtr,
                   static_cast<unsigned long long>(windowInfos[peer].windowOffset),
                   windowInfos[peer].key.c_str());
    }
    if (err != ACL_SUCCESS || peerPtr == nullptr) {
      if (useRtIpc) {
        SetLastError("rtIpcOpenMemory user window peer " + std::to_string(peer) +
                     " failed with rt error " + std::to_string(importRet));
      } else {
        SetAclError("aclrtIpcMemImportByKey user window", err);
      }
      ClearRegisteredUserWindow(runtime);
      return TRITON_ASCEND_GIN_RUNTIME_ACL_ERROR;
    }
    runtime->userWindowPtrs[peer] = peerPtr;
    runtime->hostDesc.peer_window_base[peer] = PtrToU64(peerPtr) + windowInfos[peer].windowOffset;
    runtime->userWindowImported[peer] = true;
    runtime->userWindowRtIpc[peer] = useRtIpc;
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

extern "C" TRITON_ASCEND_GIN_RUNTIME_API int TritonAscendGinRegisterSignalWindow(
    TritonAscendGinHandle handle, void *localPtr, uint64_t bytes, const char *rendezvousId,
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
  if (runtime->hostDesc.backend_kind == TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_PEER_MEM) {
    (void)rendezvousId;
    return RegisterHcclSymmetricBases(runtime, localPtr, bytes, "signal_window",
                                      runtime->hostDesc.peer_signal_base, windowHandle);
  }
#if defined(__linux__)
  if (runtime->hostDesc.backend_kind == TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_CHANNEL) {
    return RegisterHcclChannelBases(runtime, localPtr, bytes, rendezvousId, "signal_window",
                                    runtime->hostDesc.peer_signal_base, windowHandle);
  }
#endif
  if (runtime->hostDesc.backend_kind != TRITON_ASCEND_GIN_BACKEND_KIND_HCCL_PEER_MEM) {
    SetLastError("TritonAscendGinRegisterSignalWindow is currently only required for "
                 "HCCL communicators");
    return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
  }
  return TRITON_ASCEND_GIN_RUNTIME_UNSUPPORTED;
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API void *TritonAscendGinHcclAllocatorAlloc(
    int64_t size, int device, void *stream) {
  (void)stream;
  if (size <= 0) {
    return nullptr;
  }

  aclError setDeviceErr = aclrtSetDevice(device);
  if (setDeviceErr != ACL_SUCCESS) {
    SetAclError("aclrtSetDevice", setDeviceErr);
    if (DebugEnabled()) {
      std::fprintf(stderr, "TritonAscendGinHcclAllocatorAlloc: %s\n", g_lastError.c_str());
    }
    return nullptr;
  }

  uint64_t allocSize = HcclSymWindowRegisterBytes(static_cast<uint64_t>(size));
  const char *allocatorMode = std::getenv("TRITON_ASCEND_GIN_HCCL_ALLOCATOR_MODE");
  const bool forceVmm = allocatorMode != nullptr && std::strcmp(allocatorMode, "vmm") == 0;
  const bool forceHccl = allocatorMode != nullptr && std::strcmp(allocatorMode, "hccl") == 0;
  if (forceHccl && !forceVmm && EnsureHcclAllocatorSymbols()) {
    void *ptr = nullptr;
    int ret = g_hcclAllocatorSymbols.memAlloc(&ptr, allocSize);
    if (ret == 0 && ptr != nullptr) {
      {
        std::lock_guard<std::mutex> lock(g_hcclVmmAllocationMutex);
        g_hcclVmmAllocations[ptr] = HcclVmmAllocation{nullptr, allocSize, true};
      }
      if (DebugEnabled()) {
        std::fprintf(stderr,
                     "[triton_ascend_gin] hccl allocator HcclMemAlloc ptr=%p "
                     "requested=%lld alloc_size=%llu device=%d\n",
                     ptr, static_cast<long long>(size),
                     static_cast<unsigned long long>(allocSize), device);
      }
      return ptr;
    }
    SetLastError("HcclMemAlloc failed with code " + std::to_string(ret));
    if (DebugEnabled()) {
      std::fprintf(stderr, "TritonAscendGinHcclAllocatorAlloc: %s\n", g_lastError.c_str());
    }
    return nullptr;
  } else if (forceHccl) {
    if (DebugEnabled()) {
      std::fprintf(stderr, "TritonAscendGinHcclAllocatorAlloc: %s\n", g_lastError.c_str());
    }
    return nullptr;
  }

  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.id = static_cast<uint32_t>(device);
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.reserve = 0;

  size_t granularity = 0;
  aclError granularityErr =
      aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity);
  if (granularityErr != ACL_SUCCESS || granularity == 0) {
    SetAclError("aclrtMemGetAllocationGranularity", granularityErr);
    if (DebugEnabled()) {
      std::fprintf(stderr, "TritonAscendGinHcclAllocatorAlloc: %s\n", g_lastError.c_str());
    }
    return nullptr;
  }

  allocSize = ((allocSize + granularity - 1) / granularity) * granularity;
  void *ptr = nullptr;
  aclError reserveErr = aclrtReserveMemAddress(&ptr, static_cast<size_t>(allocSize), 0, nullptr, 1);
  if (reserveErr != ACL_SUCCESS || ptr == nullptr) {
    SetAclError("aclrtReserveMemAddress", reserveErr);
    if (DebugEnabled()) {
      std::fprintf(stderr, "TritonAscendGinHcclAllocatorAlloc: %s\n", g_lastError.c_str());
    }
    return nullptr;
  }

  aclrtDrvMemHandle handle = nullptr;
  aclError mallocErr = aclrtMallocPhysical(&handle, static_cast<size_t>(allocSize), &prop, 0);
  if (mallocErr != ACL_SUCCESS || handle == nullptr) {
    SetAclError("aclrtMallocPhysical", mallocErr);
    (void)aclrtReleaseMemAddress(ptr);
    if (DebugEnabled()) {
      std::fprintf(stderr, "TritonAscendGinHcclAllocatorAlloc: %s\n", g_lastError.c_str());
    }
    return nullptr;
  }

  aclError mapErr = aclrtMapMem(ptr, static_cast<size_t>(allocSize), 0, handle, 0);
  if (mapErr != ACL_SUCCESS) {
    SetAclError("aclrtMapMem", mapErr);
    (void)aclrtFreePhysical(handle);
    (void)aclrtReleaseMemAddress(ptr);
    if (DebugEnabled()) {
      std::fprintf(stderr, "TritonAscendGinHcclAllocatorAlloc: %s\n", g_lastError.c_str());
    }
    return nullptr;
  }

  {
    std::lock_guard<std::mutex> lock(g_hcclVmmAllocationMutex);
    g_hcclVmmAllocations[ptr] = HcclVmmAllocation{handle, allocSize, false};
  }
  if (DebugEnabled()) {
    std::fprintf(stderr,
                 "[triton_ascend_gin] hccl allocator alloc ptr=%p requested=%lld alloc_size=%llu "
                 "granularity=%zu device=%d\n",
                 ptr, static_cast<long long>(size), static_cast<unsigned long long>(allocSize),
                 granularity, device);
  }
  return ptr;
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API void TritonAscendGinHcclAllocatorFree(
    void *ptr, uint64_t size, void *stream) {
  (void)size;
  (void)stream;
  if (ptr == nullptr) {
    return;
  }

  HcclVmmAllocation allocation;
  {
    std::lock_guard<std::mutex> lock(g_hcclVmmAllocationMutex);
    auto it = g_hcclVmmAllocations.find(ptr);
    if (it == g_hcclVmmAllocations.end()) {
      if (DebugEnabled()) {
        std::fprintf(stderr, "TritonAscendGinHcclAllocatorFree: unknown ptr=%p\n", ptr);
      }
      return;
    }
    allocation = it->second;
    g_hcclVmmAllocations.erase(it);
  }

  if (allocation.hcclOwned) {
    if (!EnsureHcclAllocatorSymbols()) {
      if (DebugEnabled()) {
        std::fprintf(stderr, "TritonAscendGinHcclAllocatorFree: %s\n", g_lastError.c_str());
      }
      return;
    }
    int ret = g_hcclAllocatorSymbols.memFree(ptr);
    if (ret != 0) {
      SetLastError("HcclMemFree failed with code " + std::to_string(ret));
      if (DebugEnabled()) {
        std::fprintf(stderr, "TritonAscendGinHcclAllocatorFree: %s\n", g_lastError.c_str());
      }
      return;
    }
    if (DebugEnabled()) {
      std::fprintf(stderr, "[triton_ascend_gin] hccl allocator HcclMemFree ptr=%p alloc_size=%llu\n",
                   ptr, static_cast<unsigned long long>(allocation.bytes));
    }
    return;
  }

  aclError unmapErr = aclrtUnmapMem(ptr);
  aclError freeErr = aclrtFreePhysical(allocation.handle);
  aclError releaseErr = aclrtReleaseMemAddress(ptr);
  if (unmapErr != ACL_SUCCESS || freeErr != ACL_SUCCESS || releaseErr != ACL_SUCCESS) {
    SetLastError("HCCL VMM free failed unmap=" + std::to_string(static_cast<int>(unmapErr)) +
                 " free=" + std::to_string(static_cast<int>(freeErr)) +
                 " release=" + std::to_string(static_cast<int>(releaseErr)));
    if (DebugEnabled()) {
      std::fprintf(stderr, "TritonAscendGinHcclAllocatorFree: %s\n", g_lastError.c_str());
    }
    return;
  }
  if (DebugEnabled()) {
    std::fprintf(stderr, "[triton_ascend_gin] hccl allocator free ptr=%p alloc_size=%llu\n",
                 ptr, static_cast<unsigned long long>(allocation.bytes));
  }
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API const char *TritonAscendGinGetLastError(void) {
  return g_lastError.c_str();
}

extern "C" TRITON_ASCEND_GIN_RUNTIME_API void TritonAscendGinDestroy(TritonAscendGinHandle handle) {
  auto *runtime = AsHandle(handle);
  if (runtime == nullptr) {
    return;
  }
  ClearRegisteredUserWindow(runtime);
  if (runtime->devDesc != nullptr) {
    aclrtFree(runtime->devDesc);
    runtime->devDesc = nullptr;
  }
  (void)DeregisterHcclSymWindow(runtime, &runtime->hcclDataSymWindow, "window");
  (void)DeregisterHcclSymWindow(runtime, &runtime->hcclSignalSymWindow, "signal_window");
  CloseTileXRSymbols(&runtime->tilexr);
  CloseHcclPeerMemSymbols(&runtime->hccl);
#if defined(__linux__)
  ReleaseHcclAivOpbaseBuffers(runtime);
  CloseHcclPlfIpcSymbols(&runtime->hcclPlfIpc);
  CloseHcclChannelSymbols(&runtime->hcclChannel);
#endif
  delete runtime;
}

