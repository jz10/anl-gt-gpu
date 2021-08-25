#include <list>
#include <map>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <vector>

#include <pthread.h>

#define CL_TARGET_OPENCL_VERSION 210
#define CL_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 210
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#include <CL/cl_ext_intel.h>
#include <CL/opencl.hpp>

#include "hip/hipcl.hh"

/************************************************************/

#if !defined(SPDLOG_ACTIVE_LEVEL)
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#endif

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#define HIP_LOG_ERROR(msg, status) logError("{} ({}) in {}:{}:{}\n", msg, hipGetErrorName(status), __FILE__, __LINE__, __func__)

#define HIP_PROCESS_ERROR_MSG(msg, status) do {\
  if (status != hipSuccess && status != hipErrorNotReady) { \
    HIP_LOG_ERROR(msg, status); \
    throw status; \
  } \
} while(0)

#define HIP_PROCESS_ERROR(status) HIP_PROCESS_ERROR_MSG("HIP Error", status)

#define HIP_RETURN_ERROR(status) HIP_RETURN_ERROR_MSG("HIP Error", status) \
  if (status != hipSuccess && status != hipErrorNotReady) { \
    HIP_LOG_ERROR(msg, status); \
    return status; \
  } \
} while(0)

void setupSpdlog();

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
template <typename... Args>
void logDebug(const char *fmt, const Args &... args) {
  setupSpdlog();
  spdlog::debug(fmt, std::forward<const Args>(args)...);
}
#else
#define logDebug(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
template <typename... Args>
void logInfo(const char *fmt, const Args &... args) {
  setupSpdlog();
  spdlog::info(fmt, std::forward<const Args>(args)...);
}
#else
#define logInfo(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN
template <typename... Args>
void logWarn(const char *fmt, const Args &... args) {
  setupSpdlog();
  spdlog::warn(fmt, std::forward<const Args>(args)...);
}
#else
#define logWarn(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR
template <typename... Args>
void logError(const char *fmt, const Args &... args) {
  setupSpdlog();
  spdlog::error(fmt, std::forward<const Args>(args)...);
}
#else
#define logError(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_CRITICAL
template <typename... Args>
void logCritical(const char *fmt, const Args &... args) {
  setupSpdlog();
  spdlog::critical(fmt, std::forward<const Args>(args)...);
}
#else
#define logCritical(...) void(0)
#endif

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
/************************************************************/

#include "common.hh"

#ifdef __GNUC__
#define INIT_PRIORITY(x) __attribute__((init_priority(x)))
#else
#define INIT_PRIORITY(x)
#endif

#define SVM_ALIGNMENT 128

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif


class ClEvent {
protected:
  std::mutex EventMutex;
  cl::Event *Event;
  hipStream_t Stream;
  event_status_e Status;
  unsigned Flags;
  cl::Context Context;

public:
  ClEvent(cl::Context &c, unsigned flags)
      : Event(), Stream(nullptr), Status(EVENT_STATUS_INIT), Flags(flags),
        Context(c) {}

  ClEvent() : Event(0) {
    // TODO:
  }

  virtual  ~ClEvent() {
    if (Event)
      delete Event;
  }

  virtual uint64_t getFinishTime();
  virtual cl::Event getEvent() { return *Event; }
  virtual bool isFromContext(cl::Context &Other) { return (Context == Other); }
  virtual bool isFromStream(hipStream_t &Other) { return (Stream == Other); }
  virtual bool isFinished() const { return (Status == EVENT_STATUS_RECORDED); }
  virtual bool isRecordingOrRecorded() const { return (Status >= EVENT_STATUS_RECORDING); }
  virtual bool recordStream(hipStream_t S, cl_event E);
  virtual bool updateFinishStatus();
  virtual bool wait();
};

typedef std::map<const void *, std::vector<hipFunction_t>> hipFunctionMap;

class ExecItem;

class ClKernel {
protected:
  cl::Kernel Kernel;
  std::string Name;
  OCLFuncInfo *FuncInfo;
  size_t TotalArgSize;
  cl::Context Context;
  //  hipFuncAttributes attributes;

public:
  ClKernel(cl::Context &C, cl::Kernel &&K)
      : Kernel(K), Name(), FuncInfo(nullptr), TotalArgSize(0), Context(C) {}
  ClKernel() : Kernel(nullptr), Name(), FuncInfo(nullptr), TotalArgSize(0), Context(nullptr) {
  }
  ~ClKernel() {}
  bool setup(size_t Index, OpenCLFunctionInfoMap &FuncInfoMap);

  bool isNamed(const std::string &arg) const { return Name == arg; }
  bool isFromContext(const cl::Context &arg) const { return Context == arg; }
  cl::Kernel get() const { return Kernel; }
  OCLFuncInfo *getFuncInfo() const { return FuncInfo; }

  int setAllArgs(void **args, size_t shared);
  int setAllArgs(void *args, size_t size, size_t shared);
  size_t getTotalArgSize() const { return TotalArgSize; };

  // If this kernel object support HipLZ
  virtual bool SupportLZ() { return false; };
};


/********************************/

class ClProgram {
protected:
  cl::Program Program;
  cl::Context Context;
  cl::Device Device;
  std::vector<hipFunction_t> Kernels;
  OpenCLFunctionInfoMap FuncInfos;

public:
  ClProgram(cl::Context C, cl::Device D) : Program(), Context(C), Device(D) {}
  ~ClProgram();

  bool setup(std::string &binary);
  hipFunction_t getKernel(const char *name);
  hipFunction_t getKernel(std::string &name);
};

class SVMemoryRegion {
  // ContextMutex should be enough

  std::map<void *, size_t> SvmAllocations;
  cl::Context Context;

public:
  void init(cl::Context &C) { Context = C; }
  SVMemoryRegion &operator=(SVMemoryRegion &&rhs) {
    SvmAllocations = std::move(rhs.SvmAllocations);
    Context = std::move(rhs.Context);
    return *this;
  }

  void *allocate(size_t size);
  bool free(void *p, size_t *size);
  bool hasPointer(const void *p);
  bool pointerSize(void *ptr, size_t *size);
  bool pointerInfo(void *ptr, void **pbase, size_t *psize);
  int memCopy(void *dst, const void *src, size_t size, cl::CommandQueue &queue);
  int memFill(void *dst, size_t size, const void *pattern, size_t patt_size,
              cl::CommandQueue &queue);
  void clear();
};

class ClQueue {
protected:
  std::mutex QueueMutex;
  cl::CommandQueue Queue;
  cl_event LastEvent;
  unsigned int Flags;
  int Priority;

public:
  ClQueue(cl::CommandQueue q, unsigned int f, int p)
    : Queue(q), LastEvent(nullptr), Flags(f), Priority(p) {}

  // Here we add default constructor to enable sub-class for HipLZ
  ClQueue() : LastEvent(nullptr), Flags(0), Priority(0) {}

  ~ClQueue() {
    if (LastEvent) {
      logDebug("~ClQueue: Releasing last event {}", (void*)LastEvent);
      clReleaseEvent(LastEvent);
    }
  }

  ClQueue(ClQueue &&rhs) {
    Flags = rhs.Flags;
    Priority = rhs.Priority;
    LastEvent = rhs.LastEvent;
    Queue = std::move(rhs.Queue);
  }

  virtual cl::CommandQueue &getQueue() { return Queue; }
  virtual unsigned int getFlags() const { return Flags; }
  virtual int getPriority() const { return Priority; }

  virtual bool finish();
  virtual bool enqueueBarrierForEvent(hipEvent_t event);
  virtual bool addCallback(hipStreamCallback_t callback, void *userData);
  virtual bool recordEvent(hipEvent_t e);

  virtual hipError_t memCopy(void *dst, const void *src, size_t size);
  virtual hipError_t memCopyAsync(void *dst, const void *src, size_t sizeBytes);
  virtual hipError_t memFill(void *dst, size_t size, const void *pattern, size_t pat_size);
  virtual hipError_t memFillAsync(void *dst, size_t size, const void *pattern, size_t pattern_size);
  virtual hipError_t launch3(ClKernel *Kernel, dim3 grid, dim3 block);
  virtual hipError_t launch(ClKernel *Kernel, ExecItem *Arguments);

  virtual hipError_t memCopy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
			       size_t width, size_t height);
  virtual hipError_t memCopy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
				    size_t width, size_t height);
  virtual hipError_t memCopy3D(void *dst, size_t dpitch, size_t dspitch,
		       const void *src, size_t spitch, size_t sspitch,
                       size_t width, size_t height, size_t depth);
  virtual hipError_t memCopy3DAsync(void *dst, size_t dpitch, size_t dspitch,
				    const void *src, size_t spitch, size_t sspitch,
				    size_t width, size_t height, size_t depth);
  // Make meory prefetch
  virtual hipError_t memPrefetch(const void* ptr, size_t size);

  // Make the advise for the managed memory (i.e. unified shared memory)
  virtual hipError_t memAdvise(const void* ptr, size_t count, hipMemoryAdvise advice);

  // If this queue object support HipLZ
  virtual bool SupportLZ() { return false; };

  // Get the native information
  virtual bool getNativeInfo(unsigned long* nativeInfo, int* size);

  // Was stream created with the non-blocking flag
  bool isNonBlocking() { return 0 != (Flags & hipStreamNonBlocking);}
};

class ExecItem {
protected:
  // Only one HIP launch API can be active. The old API is active if
  // ArgsPointer is nullptr. Otherwise, the new API is active.

  // Structures for old HIP launch API.
  std::vector<uint8_t> ArgData;
  std::vector<std::tuple<size_t, size_t>> OffsetsSizes;

  // Structures for New HIP launch API.
  void** ArgsPointer = nullptr;

public:
  const dim3 GridDim;
  const dim3 BlockDim;
  const size_t SharedMem;
  const hipStream_t Stream;

  ExecItem(dim3 grid, dim3 block, size_t shared, hipStream_t q)
      : SharedMem(shared), Stream(q), GridDim(grid), BlockDim(block) {}

  // Records an argument for the old HIP launch API.
  void setArg(const void *arg, size_t size, size_t offset);
  // Records argument pointer for the new HIP launch API.
  void setArgsPointer(void** args);
  int setupAllArgs(ClKernel *kernel);

  virtual hipError_t launch(ClKernel *Kernel);

  // If this execution item object support HipLZ
  virtual bool SupportLZ() { return false; };
};

class ClDevice;

class ClContext;

class GlobalPtrs {
protected:
  // The map between global pointer and its size
  std::map<void *, size_t> GlobalPointers;

  // The context reference
  ClContext* context;

public:
  GlobalPtrs() : context(nullptr) {};
  GlobalPtrs(ClContext* ctx) : context(ctx) {};

  // Set the context for global pointers
  void setupContext(ClContext* ctx) {
    this->context = ctx;
  }

  // Assign operator
  GlobalPtrs &operator=(GlobalPtrs &&rhs) {
    this->GlobalPointers = std::move(rhs.GlobalPointers);
    return *this;
  }

  // Add global pointer
  void addGlobalPtr(void *ptr, size_t size) {
    this->GlobalPointers.emplace(ptr, size);
  }

  // Remove global pointer
  void removeGlobalPtr(void *ptr) {
    auto it = this->GlobalPointers.find(ptr);
    if (it != this->GlobalPointers.end())
      GlobalPointers.erase(it);
  }

  // Get the given pointer's size, i.e. the length of pointer memory region, and the given pointer
  // has to be the base pointer
  bool pointerSize(void *ptr, size_t *size) {
    auto it = this->GlobalPointers.find(ptr);
    if (it != this->GlobalPointers.end()) {
      *size = it->second;
      return true;
    }

    return false;
  }

  // Get the given pointer's based pointer and lenght of pointer memory region
  bool pointerInfo(void *ptr, void **pbase, size_t *psize) {
    for (auto it : this->GlobalPointers) {
    if ((it.first <= ptr) && (ptr < ((const char *)it.first + it.second))) {
      if (pbase)
        *pbase = it.first;
      if (psize)
        *psize = it.second;
      return true;
    }
  }

    return false;
  }

  // Check if the given pointer has been registered
  bool hasPointer(const void *ptr) {
    return this->GlobalPointers.find((void *)ptr) != this->GlobalPointers.end();
  }
};

class ClContext {
protected:
  std::mutex ContextMutex;
  unsigned Flags;
  ClDevice *Device;
  cl::Context Context;

  SVMemoryRegion Memory;

  std::set<hipStream_t> Queues;
  hipStream_t DefaultQueue;
  std::stack<ExecItem *> ExecStack;

  std::map<const void *, ClProgram *> BuiltinPrograms;
  std::set<ClProgram *> Programs;

  // The reference for global pointer table
  GlobalPtrs globalPtrs;

  hipStream_t findQueue(hipStream_t stream);

public:
  ClContext(ClDevice *D, unsigned f);
  ~ClContext();

  ClDevice *getDevice() const { return Device; }
  unsigned getFlags() const { return Flags; }
  hipStream_t getDefaultQueue() { return DefaultQueue; }
  const std::set<hipStream_t> & getQueues() { return Queues; };
  virtual void reset();

  virtual hipError_t eventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop);
  ClEvent *createEvent(unsigned Flags);
  virtual bool createQueue(hipStream_t *stream, unsigned int Flags, int priority);
  hipStream_t createRTSpecificQueue(cl::CommandQueue q, unsigned int f, int p);
  bool releaseQueue(hipStream_t stream);
  hipError_t recordEvent(hipStream_t stream, hipEvent_t event);
  virtual bool finishAll();

  void *allocate(size_t size);
  bool free(void *p);
  bool hasPointer(const void *p);
  bool getPointerSize(void *ptr, size_t *size);
  bool findPointerInfo(hipDeviceptr_t dptr, hipDeviceptr_t *pbase,
                       size_t *psize);

  hipError_t configureCall(dim3 grid, dim3 block, size_t shared, hipStream_t q);
  hipError_t setArg(const void *arg, size_t size, size_t offset);
  hipError_t launchHostFunc(const void *HostFunction);
  hipError_t createProgramBuiltin(std::string *module, const void *HostFunction,
                                  std::string &FunctionName);
  hipError_t destroyProgramBuiltin(const void *HostFunction);

  hipError_t launchWithKernelParams(dim3 grid, dim3 block, size_t shared,
                                    hipStream_t stream, void **kernelParams,
                                    hipFunction_t kernel);
  hipError_t launchWithExtraParams(dim3 grid, dim3 block, size_t shared,
                                   hipStream_t stream, void **extraParams,
                                   hipFunction_t kernel);

  ClProgram *createProgram(std::string &binary);
  hipError_t destroyProgram(ClProgram *prog);

  virtual bool getSymbolAddressSize(const char *name, hipDeviceptr_t *dptr, size_t *bytes);
  void synchronizeQueues(hipStream_t queue);
};

class ClDevice {
protected:
  std::mutex DeviceMutex;

  hipDevice_t Index;
  hipDeviceProp_t Properties;
  bool SupportsIntelDiag;
  std::map<hipDeviceAttribute_t, int> Attributes;
  size_t TotalUsedMem, GlobalMemSize, MaxUsedMem;

  std::vector<std::string *> Modules;
  std::map<const void *, std::string *> HostPtrToModuleMap;
  std::map<const void *, std::string> HostPtrToNameMap;
  cl::Device Device;
  cl::Platform Platform;
  ClContext *PrimaryContext;
  std::set<ClContext *> Contexts;

  void setupProperties(int Index);

public:
  ClDevice(cl::Device d, cl::Platform p, hipDevice_t index);
  ClDevice(ClDevice &&rhs);
  void setPrimaryCtx();
  ~ClDevice();
  void reset();
  cl::Device &getDevice() { return Device; }
  hipDevice_t getHipDeviceT() const { return Index; }
  ClContext *getPrimaryCtx() const { return PrimaryContext; }

  ClContext *newContext(unsigned int flags);
  bool addContext(ClContext *ctx);
  bool removeContext(ClContext *ctx);
  bool supportsIntelDiag() const { return SupportsIntelDiag; }

  void registerModule(std::string *module);
  void unregisterModule(std::string *module);
  bool registerFunction(std::string *module, const void *HostFunction,
                        const char *FunctionName);
  bool getModuleAndFName(const void *HostFunction, std::string &FunctionName,
                         std::string **module);

  const char *getName() const { return Properties.name; }
  int getAttr(int *pi, hipDeviceAttribute_t attr);
  void copyProperties(hipDeviceProp_t *prop);

  size_t getGlobalMemSize() const { return GlobalMemSize; }
  size_t getUsedGlobalMem() const { return TotalUsedMem; }
  bool reserveMem(size_t bytes);
  bool releaseMem(size_t bytes);
};

extern size_t NumDevices;

ClDevice &CLDeviceById(int deviceId);

/********************************************************************/

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
