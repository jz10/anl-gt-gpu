
#include <list>
#include <map>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <vector>

#define CL_HPP_TARGET_OPENCL_VERSION 210
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#include <CL/cl2.hpp>
#include <CL/cl_ext_intel.h>

#include "hip/hipcl.hh"

#include "ze_api.h"

/************************************************************/

#if !defined(SPDLOG_ACTIVE_LEVEL)
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#endif

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

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

  ~ClEvent() {
    if (Event)
      delete Event;
  }

  uint64_t getFinishTime();
  cl::Event getEvent() { return *Event; }
  bool isFromContext(cl::Context &Other) { return (Context == Other); }
  bool isFromStream(hipStream_t &Other) { return (Stream == Other); }
  bool isFinished() const { return (Status == EVENT_STATUS_RECORDED); }
  bool isRecordingOrRecorded() const { return (Status >= EVENT_STATUS_RECORDING); }
  bool recordStream(hipStream_t S, cl_event E);
  bool updateFinishStatus();
  bool wait();
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
  ~ClKernel() {}
  bool setup(size_t Index, OpenCLFunctionInfoMap &FuncInfoMap);

  bool isNamed(const std::string &arg) const { return Name == arg; }
  bool isFromContext(const cl::Context &arg) const { return Context == arg; }
  cl::Kernel get() const { return Kernel; }
  OCLFuncInfo *getFuncInfo() const { return FuncInfo; }

  int setAllArgs(void **args, size_t shared);
  int setAllArgs(void *args, size_t size, size_t shared);
  size_t getTotalArgSize() const { return TotalArgSize; }

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

struct hipStreamCallbackData {
  hipStream_t Stream;
  hipError_t Status;
  void *UserData;
  hipStreamCallback_t Callback;
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
  int memFill(void *dst, size_t size, void *pattern, size_t patt_size,
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

  cl::CommandQueue &getQueue() { return Queue; }
  unsigned int getFlags() const { return Flags; }
  int getPriority() const { return Priority; }

  bool finish();
  bool enqueueBarrierForEvent(hipEvent_t event);
  bool addCallback(hipStreamCallback_t callback, void *userData);
  bool recordEvent(hipEvent_t e);

  hipError_t memCopy(void *dst, const void *src, size_t size);
  hipError_t memFill(void *dst, size_t size, void *pattern, size_t pat_size);
  hipError_t launch3(ClKernel *Kernel, dim3 grid, dim3 block);
  hipError_t launch(ClKernel *Kernel, ExecItem *Arguments);
};

class ExecItem {
protected:
  size_t SharedMem;
  hipStream_t Stream;
  std::vector<uint8_t> ArgData;
  std::vector<std::tuple<size_t, size_t>> OffsetsSizes;

public:
  dim3 GridDim;
  dim3 BlockDim;

  ExecItem(dim3 grid, dim3 block, size_t shared, hipStream_t q)
      : SharedMem(shared), Stream(q), GridDim(grid), BlockDim(block) {}

  void setArg(const void *arg, size_t size, size_t offset);
  int setupAllArgs(ClKernel *kernel);

  hipError_t launch(ClKernel *Kernel) { return Stream->launch(Kernel, this); }
};

class ClDevice;
class LZExecItem;

class ClContext {
protected:
  std::mutex ContextMutex;
  unsigned Flags;
  ClDevice *Device;
  cl::Context Context;

  SVMemoryRegion Memory;

  std::set<hipStream_t> Queues;
  hipStream_t DefaultQueue;
  std::stack<LZExecItem *> ExecStack;

  std::map<const void *, ClProgram *> BuiltinPrograms;
  std::set<ClProgram *> Programs;

  hipStream_t findQueue(hipStream_t stream);

public:
  ClContext(ClDevice *D, unsigned f);
  ~ClContext();

  ClDevice *getDevice() const { return Device; }
  unsigned getFlags() const { return Flags; }
  hipStream_t getDefaultQueue() { return DefaultQueue; }
  void reset();

  hipError_t eventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop);
  ClEvent *createEvent(unsigned Flags);
  bool createQueue(hipStream_t *stream, unsigned int Flags, int priority);
  bool releaseQueue(hipStream_t stream);
  hipError_t memCopy(void *dst, const void *src, size_t size,
                     hipStream_t stream);
  hipError_t memFill(void *dst, size_t size, void *pattern, size_t pat_size,
                     hipStream_t stream);
  hipError_t recordEvent(hipStream_t stream, hipEvent_t event);
  bool finishAll();

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

class LZKernel;

class LZExecItem : public ExecItem {
public:
  dim3 GridDim;
  dim3 BlockDim;

  LZExecItem(dim3 grid, dim3 block, size_t shared) : ExecItem(grid, block, shared, nullptr) {}

  // Setup all arguments for HipLZ kernel funciton invocation
  int setupAllArgs(LZKernel *kernel);

  bool launch(LZKernel *Kernel) { 
	  // return Stream->launch(Kernel, this); 
	  return false;
  	}
};

class LZContext;

class LZDevice {
protected:
  std::mutex mtx;
  LZContext* lzContext;
  ze_device_handle_t hDevice;
  ze_driver_handle_t hDriver;

  std::vector<std::string *> Modules;
  std::map<const void *, std::string *> HostPtrToModuleMap;
  std::map<const void *, std::string> HostPtrToNameMap;

public:
  LZDevice(ze_device_handle_t hDevice_, ze_driver_handle_t hDriver_);
  
  ze_device_handle_t GetDeviceHandle() { return this->hDevice; };
  ze_driver_handle_t GetDriverHandle() { return this->hDriver; }
  
  // Register HipLZ module which is presented as IL
  void registerModule(std::string* module);
  // Regsiter HipLZ module, kernel function name with host function which is a wrapper
  bool registerFunction(std::string *module, const void *HostFunction, const char *FunctionName);

  // Get host function pointer's corresponding name
  std::string GetHostFunctionName(const void* HostFunction);

  // Get primary context
  LZContext* getPrimaryCtx() { return this->lzContext; };
};

class LZKernel {
protected:
  // HipLZ kernel handle
  ze_kernel_handle_t hKernel;
  // The function info
  OCLFuncInfo *FuncInfo;

public: 
  LZKernel(ze_kernel_handle_t hKernel_) : FuncInfo(nullptr) {
    this->hKernel = hKernel_;
  }

  ze_kernel_handle_t GetKernelHandle() { return this->hKernel; }

  OCLFuncInfo *getFuncInfo() const { return FuncInfo; }
};

class LZModule {
protected:
  // HipLZ module handle
  ze_module_handle_t hModule;
  // The name --> HipLZ kernel map
  std::map<std::string, LZKernel* > kernels;

public:
  LZModule(ze_module_handle_t hModule_) {
    this->hModule = hModule_;
  };

  // Get HipLZ module handle  
  ze_module_handle_t GethModuleHandle() { return this->hModule; }

  // Create HipLZ kernel via function name
  void CreateKernel(std::string funcName);

  // Get HipLZ kernel via funciton name
  LZKernel* GetKernel(std::string funcName);
};

class LZCommandList;

class LZContext : public ClContext {
protected:
  // Lock for thread safeness
  std::mutex mtx;
  // Reference to HipLZ device
  LZDevice* lzDevice;
  // Reference to HipLZ module
  LZModule* lzModule;
  // Reference to HipLZ command list
  LZCommandList* lzCommandList;
  // HipLZ context handle
  ze_context_handle_t hContext;
  // OpenCL function information map, this is used for presenting SPIR-V kernel funcitons' arguments
  OpenCLFunctionInfoMap FuncInfos;

public:
  LZContext(ClDevice* D, unsigned f) : ClContext(D, f), lzDevice(0), lzModule(0) {}
  LZContext(LZDevice* D, ze_context_handle_t hContext_);  

  bool CreateModule(uint8_t* moduleIL, size_t ilSize, std::string funcName);
  ze_context_handle_t GetContextHandle() { return this->hContext; }

  // Configure the call for kernel
  bool configureCall(dim3 grid, dim3 block, size_t shared);
  
  // Set argument
  bool setArg(const void *arg, size_t size, size_t offset);

  // Launch HipLZ kernel
  bool launchHostFunc(const void* HostFunction);

  void *allocate(size_t size);
  bool free(void *p);
};

class LZCommandList {
protected:
  LZContext* lzContext;
  ze_command_list_handle_t hCommandList;

public:
  LZCommandList(LZContext* lzContext_, ze_command_list_handle_t hCommandList_) {
    this->lzContext = lzContext_;
    this->hCommandList = hCommandList_;
  };

  ze_command_list_handle_t GetCommandListHandle() { return this->hCommandList; }
};

LZDevice &HipLZDeviceById(int deviceId);

extern size_t NumLZDevices;

void InitializeHipLZ();


void InitializeOpenCL();
void UnInitializeOpenCL();

extern size_t NumDevices;

ClDevice &CLDeviceById(int deviceId);

/********************************************************************/

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
