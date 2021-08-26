#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

#include "backend.hh"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#define FIND_QUEUE(stream)                                                     \
  ClQueue *Queue = findQueue(stream);                                          \
  if (Queue == nullptr)                                                        \
    return hipErrorInvalidResourceHandle;

#define FIND_QUEUE_LOCKED(stream)                                              \
  std::lock_guard<std::mutex> Lock(ContextMutex);                              \
  ClQueue *Queue = findQueue(stream);                                          \
  if (Queue == nullptr)                                                        \
    return hipErrorInvalidResourceHandle;

size_t NumDevices = 0;

static std::vector<ClDevice *> OpenCLDevices INIT_PRIORITY(120);
static std::vector<cl::Platform> Platforms INIT_PRIORITY(120);

/********************************/

ClKernel::~ClKernel() {}

/********************************/

ClProgram::~ClProgram() {
  for (auto x: kernels)
    delete x.second;

  std::set<OCLFuncInfo *> PtrsToDelete;
  for (auto &kv : FuncInfos)
    PtrsToDelete.insert(kv.second);
  for (auto &Ptr : PtrsToDelete)
    delete Ptr;
}

hipFunction_t ClProgram::getKernel(std::string &name) {
  auto it = kernels.find(name);
  if (it == kernels.end())
    return nullptr;
  else
    return it->second;
}

hipFunction_t ClProgram::getKernel(const char *name) {
  std::string SName(name);
  return this->getKernel(SName);
}

/********************************/

void *SVMemoryRegion::allocate(size_t size) {
  void *Ptr = ::clSVMAlloc(Context(), CL_MEM_READ_WRITE, size, SVM_ALIGNMENT);
  if (Ptr) {
    logDebug("clSVMAlloc allocated: {} / {}\n", Ptr, size);
    SvmAllocations.emplace(Ptr, size);
  } else
    logError("clSVMAlloc of {} bytes failed\n", size);
  return Ptr;
}

bool SVMemoryRegion::free(void *p, size_t *size) {
  auto I = SvmAllocations.find(p);
  if (I != SvmAllocations.end()) {
    void *Ptr = I->first;
    *size = I->second;
    logDebug("clSVMFree on: {}\n", Ptr);
    SvmAllocations.erase(I);
    ::clSVMFree(Context(), Ptr);
    return true;
  } else {
    logError("clSVMFree on unknown memory: {}\n", p);
    return false;
  }
}

bool SVMemoryRegion::hasPointer(const void *p) {
  logDebug("hasPointer on: {}\n", p);
  return (SvmAllocations.find((void *)p) != SvmAllocations.end());
}

bool SVMemoryRegion::pointerSize(void *ptr, size_t *size) {
  logDebug("pointerSize on: {}\n", ptr);
  auto I = SvmAllocations.find(ptr);
  if (I != SvmAllocations.end()) {
    *size = I->second;
    return true;
  } else {
    return false;
  }
}

bool SVMemoryRegion::pointerInfo(void *ptr, void **pbase, size_t *psize) {
  logDebug("pointerInfo on: {}\n", ptr);
  for (auto I : SvmAllocations) {
    if ((I.first <= ptr) && (ptr < ((const char *)I.first + I.second))) {
      if (pbase)
        *pbase = I.first;
      if (psize)
        *psize = I.second;
      return true;
    }
  }
  return false;
}

void SVMemoryRegion::clear() {
  for (auto I : SvmAllocations) {
    ::clSVMFree(Context(), I.first);
  }
  SvmAllocations.clear();
}

/***********************************************************************/

void ExecItem::setArg(const void *arg, size_t size, size_t offset) {
  assert(!ArgsPointer && "New HIP Launch API is active!");

  if ((offset + size) > ArgData.size())
    ArgData.resize(offset + size + 1024);

  std::memcpy(ArgData.data() + offset, arg, size);
  logDebug("setArg on {} size {} offset {}\n", (void *)this, size, offset);
  OffsetsSizes.push_back(std::make_tuple(offset, size));
}

void ExecItem::setArgsPointer(void** args) {
  assert(ArgData.empty() && "Old HIP launch API is active!");
  ArgsPointer = args;
}

/***********************************************************************/

/* errinfo is a pointer to an error string.
 * private_info and cb represent a pointer to binary data that is
 * returned by the OpenCL implementation that can be used
 * to log additional information helpful in debugging the error.
 * user_data is a pointer to user supplied data.
 */

static void intel_driver_cb(
    const char *errinfo,
    const void *private_info,
    size_t cb,
    void *user_data) {

    logDebug("INTEL DIAG: {}\n", errinfo);
}

hipStream_t ClContext::createRTSpecificQueue(cl::CommandQueue q, unsigned int f, int p) {
  HIP_PROCESS_ERROR_MSG("HipLZ should not use ClContext to createRTSpecificQueue", hipErrorNotSupported);
}

ClContext::ClContext(ClDevice *D, unsigned f) {
  Device = D;
  Flags = f;
  int err;

  if (!D) {
    logDebug("CL CONTEXT WAS NOT INITIALIZED");
    return;
  }
  
  if (D->supportsIntelDiag()) {
    logDebug("creating context with Intel Debugging\n");
    cl_bitfield vl =
            CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL
            | CL_CONTEXT_DIAGNOSTICS_LEVEL_GOOD_INTEL
            | CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL;
    cl_context_properties props[] = {
        CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL,
        (cl_context_properties)vl,
        0 };
    Context = cl::Context(D->getDevice(), props,
                          intel_driver_cb, this,
                          &err);
  } else {
    logDebug("creating context for dev: {}\n", D->getName());
    Context = cl::Context(D->getDevice(), NULL, NULL, NULL, &err);
  }
  assert(err == CL_SUCCESS);

  cl::CommandQueue CmdQueue(Context, Device->getDevice(),
                            CL_QUEUE_PROFILING_ENABLE, &err);
  assert(err == CL_SUCCESS);

  DefaultQueue = createRTSpecificQueue(CmdQueue, 0, 0); 

  Memory.init(Context);
}

void ClContext::reset() {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  int err;

  while (!this->ExecStack.empty()) {
    ExecItem *Item = ExecStack.top();
    delete Item;
    this->ExecStack.pop();
  }

  this->Queues.clear();
  delete DefaultQueue;
  this->Memory.clear();

  cl::CommandQueue CmdQueue(Context, Device->getDevice(),
                            CL_QUEUE_PROFILING_ENABLE, &err);
  assert(err == CL_SUCCESS);

  DefaultQueue = createRTSpecificQueue(CmdQueue, 0, 0); 
}

ClContext::~ClContext() {

  while (!this->ExecStack.empty()) {
    ExecItem *Item = ExecStack.top();
    delete Item;
    this->ExecStack.pop();
  }

  for (ClQueue *Q : Queues) {
    delete Q;
  }
  Queues.clear();
  delete DefaultQueue;
  Memory.clear();

  for (ClProgram *P : Programs) {
    delete P;
  }
  Programs.clear();

  for (auto It : BuiltinPrograms) {
    delete It.second;
  }
  BuiltinPrograms.clear();
}

hipStream_t ClContext::findQueue(hipStream_t stream) {
  if (stream == nullptr || stream == DefaultQueue)
    return DefaultQueue;

  auto I = Queues.find(stream);
  if (I == Queues.end())
    return nullptr;
  return *I;
}

ClEvent *ClContext::createEvent(unsigned flags) {
  HIP_PROCESS_ERROR_MSG("HipLZ should not use ClContext to createEvent", hipErrorNotSupported);
}

void *ClContext::allocate(size_t size) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  if (!Device->reserveMem(size))
    return nullptr;

  void *retval = Memory.allocate(size);
  if (retval == nullptr)
    Device->releaseMem(size);
  return retval;
}

bool ClContext::free(void *p) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  size_t size;

  bool retval = Memory.free(p, &size);
  if (retval)
    Device->releaseMem(size);
  return retval;
}

bool ClContext::hasPointer(const void *p) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  return Memory.hasPointer(p);
}

bool ClContext::getPointerSize(void *ptr, size_t *size) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  return Memory.pointerSize(ptr, size);
}

bool ClContext::findPointerInfo(hipDeviceptr_t dptr, hipDeviceptr_t *pbase,
                                size_t *psize) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  return Memory.pointerInfo(dptr, pbase, psize);
}

hipError_t ClContext::recordEvent(hipStream_t stream, hipEvent_t event) {
  FIND_QUEUE_LOCKED(stream);
  
  return Queue->recordEvent(event) ? hipSuccess : hipErrorInvalidContext;
}

#define NANOSECS 1000000000

hipError_t ClContext::eventElapsedTime(float *ms, hipEvent_t start,
                                       hipEvent_t stop) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  assert(start->isFromContext(Context));
  assert(stop->isFromContext(Context));

  if (!start->isRecordingOrRecorded() || !stop->isRecordingOrRecorded())
    return hipErrorInvalidResourceHandle;

  start->updateFinishStatus();
  stop->updateFinishStatus();
  if (!start->isFinished() || !stop->isFinished())
    return hipErrorNotReady;

  uint64_t Started = start->getFinishTime();
  uint64_t Finished = stop->getFinishTime();

  logDebug("EventElapsedTime: STARTED {} / {} FINISHED {} / {} \n",
           (void *)start, Started, (void *)stop, Finished);

  // apparently fails for Intel NEO, god knows why
  // assert(Finished >= Started);
  uint64_t Elapsed;
  if (Finished < Started) {
    logWarn("Finished < Started\n");
    Elapsed = Started - Finished;
  } else
    Elapsed = Finished - Started;
  uint64_t MS = (Elapsed / NANOSECS)*1000;
  uint64_t NS = Elapsed % NANOSECS;
  float FractInMS = ((float)NS) / 1000000.0f;
  *ms = (float)MS + FractInMS;
  return hipSuccess;
}

bool ClContext::createQueue(hipStream_t *stream, unsigned flags, int priority) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  int err;
  cl::CommandQueue NewQueue(Context, Device->getDevice(),
                            CL_QUEUE_PROFILING_ENABLE, &err);
  assert(err == CL_SUCCESS);

  hipStream_t Ptr = createRTSpecificQueue(NewQueue, flags, priority); 
  Queues.insert(Ptr);
  *stream = Ptr;
  return true;
}

bool ClContext::releaseQueue(hipStream_t stream) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  auto I = Queues.find(stream);
  if (I == Queues.end())
    return false;
  hipStream_t QueuePtr = *I;
  delete QueuePtr;
  Queues.erase(I);
  return true;
}

bool ClContext::finishAll() {
  HIP_PROCESS_ERROR_MSG("HipLZ should not use ClContext to finishAll", hipErrorNotSupported);
}

hipError_t ClContext::configureCall(dim3 grid, dim3 block, size_t shared,
                                    hipStream_t stream) {
  HIP_PROCESS_ERROR_MSG("HipLZ should not use ClContext to configureCall", hipErrorNotSupported);
}

hipError_t ClContext::setArg(const void *arg, size_t size, size_t offset) {
  // Can't do a size check here b/c we don't know the kernel yet
  std::lock_guard<std::mutex> Lock(ContextMutex);
  ExecStack.top()->setArg(arg, size, offset);
  return hipSuccess;
}

hipError_t ClContext::createProgramBuiltin(std::string *module,
                                           const void *HostFunction,
                                           std::string &FunctionName) {
  HIP_PROCESS_ERROR_MSG("HipLZ should not use ClContext to createProgramBuiltin", hipErrorNotSupported);

/*  std::lock_guard<std::mutex> Lock(ContextMutex);

  std::cout << "call cl create program builtin" << std::endl;
  
  logDebug("createProgramBuiltin: {}\n", FunctionName);

  ClProgram *p = new ClProgram(Context, Device->getDevice());
  if (p == nullptr)
    return hipErrorOutOfMemory;

  if (!p->setup(*module)) {
    logCritical("Failed to build program for '{}'", FunctionName);
    delete p;
    return hipErrorInitializationError;
  }

  BuiltinPrograms[HostFunction] = p;
  return hipSuccess;*/
}

hipError_t ClContext::destroyProgramBuiltin(const void *HostFunction) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  auto it = BuiltinPrograms.find(HostFunction);
  if (it == BuiltinPrograms.end())
    return hipErrorUnknown;
  delete it->second;
  BuiltinPrograms.erase(it);
  return hipSuccess;
}

hipError_t ClContext::launchHostFunc(const void *HostFunction) {

  std::string FunctionName;
  std::string *module;

  if (!Device->getModuleAndFName(HostFunction, FunctionName, &module)) {
    logCritical("can NOT find kernel with stub address {} for device {}\n",
                HostFunction, Device->getHipDeviceT());
    return hipErrorLaunchFailure;
  }

  std::lock_guard<std::mutex> Lock(ContextMutex);

  ClKernel *Kernel = nullptr;
  // TODO can this happen ?
  if (BuiltinPrograms.find(HostFunction) != BuiltinPrograms.end())
    Kernel = BuiltinPrograms[HostFunction]->getKernel(FunctionName);

  if (Kernel == nullptr) {
    logCritical("can NOT find kernel with stub address {} for device {}\n",
                HostFunction, Device->getHipDeviceT());
    return hipErrorLaunchFailure;
  }

  ExecItem *Arguments;
  Arguments = ExecStack.top();
  ExecStack.pop();

  return Arguments->launch(Kernel);
}

hipError_t ClContext::launchWithKernelParams(dim3 grid, dim3 block,
                                             size_t shared, hipStream_t stream,
                                             void **kernelParams,
                                             hipFunction_t kernel) {
  HIP_PROCESS_ERROR_MSG("HipLZ should not use ClContext to call launchWithKernelParams", hipErrorNotSupported);
}

hipError_t ClContext::launchWithExtraParams(dim3 grid, dim3 block,
                                            size_t shared, hipStream_t stream,
                                            void **extraParams,
                                            hipFunction_t kernel) {
  HIP_PROCESS_ERROR_MSG("HipLZ should not use ClContext to call launchWithExtraParams", hipErrorNotSupported);
}

ClProgram *ClContext::createProgram(std::string &binary) {
  HIP_PROCESS_ERROR_MSG("HipLZ should not use ClContext to call createProgram", hipErrorNotSupported);
}

hipError_t ClContext::destroyProgram(ClProgram *prog) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  auto it = Programs.find(prog);
  if (it == Programs.end())
    return hipErrorInvalidHandle;

  Programs.erase(it);
  return hipSuccess;
}

// Get the address and size for the given symbol's name
bool ClContext::getSymbolAddressSize(const char *name, hipDeviceptr_t *dptr, size_t *bytes) {
  // TODO: no OpenCL support yet
  
  return false;
}

/***********************************************************************/

void ClDevice::setupProperties(int index) {
  cl_int err;
  std::string Temp;
  cl::Device Dev = this->Device;

  Temp = Dev.getInfo<CL_DEVICE_NAME>(&err);
  strncpy(Properties.name, Temp.c_str(), 255);
  Properties.name[255] = 0;

  Properties.totalGlobalMem = Dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&err);

  Properties.sharedMemPerBlock = Dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&err);

  Properties.maxThreadsPerBlock =
      Dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);

  std::vector<size_t> wi = Dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

  Properties.maxThreadsDim[0] = wi[0];
  Properties.maxThreadsDim[1] = wi[1];
  Properties.maxThreadsDim[2] = wi[2];

  // Maximum configured clock frequency of the device in MHz.
  Properties.clockRate = 1000 * Dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  Properties.multiProcessorCount = Dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  Properties.l2CacheSize = Dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  Properties.totalConstMem = Dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // totally made up
  Properties.regsPerBlock = 64;

  // The minimum subgroup size on an intel GPU
  if (Dev.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
    std::vector<uint> sg = Dev.getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
    if (sg.begin() != sg.end())
      Properties.warpSize = *std::min_element(sg.begin(), sg.end());
  }
  Properties.maxGridSize[0] = Properties.maxGridSize[1] =
      Properties.maxGridSize[2] = 65536;
  Properties.memoryClockRate = 1000;
  Properties.memoryBusWidth = 256;
  Properties.major = 2;
  Properties.minor = 0;

  Properties.maxThreadsPerMultiProcessor = 10;

  Properties.computeMode = 0;
  Properties.arch = {};

  Temp = Dev.getInfo<CL_DEVICE_EXTENSIONS>();
  if (Temp.find("cl_khr_global_int32_base_atomics") != std::string::npos)
    Properties.arch.hasGlobalInt32Atomics = 1;
  else
    Properties.arch.hasGlobalInt32Atomics = 0;

  if (Temp.find("cl_khr_local_int32_base_atomics") != std::string::npos)
    Properties.arch.hasSharedInt32Atomics = 1;
  else
    Properties.arch.hasSharedInt32Atomics = 0;

  if (Temp.find("cl_khr_int64_base_atomics") != std::string::npos) {
    Properties.arch.hasGlobalInt64Atomics = 1;
    Properties.arch.hasSharedInt64Atomics = 1;
  }
  else {
    Properties.arch.hasGlobalInt64Atomics = 1;
    Properties.arch.hasSharedInt64Atomics = 1;
  }

  if (Temp.find("cl_khr_fp64") != std::string::npos) 
    Properties.arch.hasDoubles = 1;
  else
    Properties.arch.hasDoubles = 0;

  Properties.clockInstructionRate = 2465;
  Properties.concurrentKernels = 1;
  Properties.pciDomainID = 0;
  Properties.pciBusID = 0x10;
  Properties.pciDeviceID = 0x40 + index;
  Properties.isMultiGpuBoard = 0;
  Properties.canMapHostMemory = 1;
  Properties.gcnArch = 0;
  Properties.integrated = 0;
  Properties.maxSharedMemoryPerMultiProcessor = 0;
}

ClDevice::ClDevice(cl::Device d, cl::Platform p, hipDevice_t index) {
  Device = d;
  Platform = p;
  Index = index;
  SupportsIntelDiag = false;

  setupProperties(index);

  std::string extensions = d.getInfo<CL_DEVICE_EXTENSIONS>();
  if (extensions.find("cl_intel_driver_diag") != std::string::npos) {
      logDebug("Intel debug extension supported\n");
      SupportsIntelDiag = true;
  }

  TotalUsedMem = 0;
  MaxUsedMem = 0;
  GlobalMemSize = Properties.totalGlobalMem;
  PrimaryContext = nullptr;

  logDebug("Device {} is {}: name \"{}\" \n",
           index, (void *)this, Properties.name);
}

void ClDevice::setPrimaryCtx() {
  PrimaryContext = new ClContext(this, 0);
}

void ClDevice::reset() {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  // TODO should we remove all contexts ?
  PrimaryContext->reset();
  for (ClContext *C : Contexts) {
    C->reset();
  }
}

ClDevice::~ClDevice() {
  delete PrimaryContext;
  logInfo("Max used memory on device {}: {} MB\n", Properties.name, (MaxUsedMem >> 20));
  logDebug("Destroy device {}\n", Properties.name);
  for (ClContext *C : Contexts) {
    delete C;
  }
  Contexts.clear();
}

ClDevice::ClDevice(ClDevice &&rhs) {
  Index = rhs.Index;
  Properties = rhs.Properties;
  Attributes = std::move(rhs.Attributes);

  Device = std::move(rhs.Device);
  Platform = std::move(rhs.Platform);
  PrimaryContext = std::move(rhs.PrimaryContext);
  Contexts = std::move(rhs.Contexts);
  TotalUsedMem = rhs.TotalUsedMem;
  MaxUsedMem = rhs.MaxUsedMem;
  GlobalMemSize = rhs.GlobalMemSize;
}

bool ClDevice::reserveMem(size_t bytes) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  if (bytes <= (GlobalMemSize - TotalUsedMem)) {
    TotalUsedMem += bytes;
    if (TotalUsedMem > MaxUsedMem)
      MaxUsedMem = TotalUsedMem;
    logDebug("Currently used memory on dev {}: {} M\n", Index, (TotalUsedMem >> 20));
    return true;
  } else {
    logError("Can't allocate {} bytes of memory\n", bytes);
    return false;
  }
}

bool ClDevice::releaseMem(size_t bytes) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  if (TotalUsedMem >= bytes) {
    TotalUsedMem -= bytes;
    return true;
  } else {
    return false;
  }
}

int ClDevice::getAttr(int *pi, hipDeviceAttribute_t attr) {
  auto I = Attributes.find(attr);
  if (I != Attributes.end()) {
    *pi = I->second;
    return 0;
  } else {
    return 1;
  }
}

void ClDevice::copyProperties(hipDeviceProp_t *prop) {
  if (prop)
    std::memcpy(prop, &this->Properties, sizeof(hipDeviceProp_t));
}

bool ClDevice::addContext(ClContext *ctx) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  auto it = Contexts.find(ctx);
  if (it != Contexts.end())
    return false;
  Contexts.emplace(ctx);
  return true;
}

bool ClDevice::removeContext(ClContext *ctx) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  auto I = std::find(Contexts.begin(), Contexts.end(), ctx);
  if (I == Contexts.end())
    return false;

  Contexts.erase(I);
  delete ctx;
  // TODO:
  // As per CUDA docs , attempting to access ctx from those threads which has
  // this ctx as current, will result in the error
  // HIP_ERROR_CONTEXT_IS_DESTROYED.
  return true;
}

ClContext *ClDevice::newContext(unsigned int flags) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);

  ClContext *ctx = new ClContext(this, flags);
  if (ctx != nullptr)
    Contexts.emplace(ctx);
  return ctx;
}

void ClDevice::registerModule(std::string *module) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  Modules.push_back(module);
}

void ClDevice::unregisterModule(std::string *module) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);

  auto it = std::find(Modules.begin(), Modules.end(), module);
  if (it == Modules.end()) {
    logCritical("unregisterModule: couldn't find {}\n", (void *)module);
    return;
  } else
    Modules.erase(it);

  const void *HostFunction = nullptr;
  std::map<const void *, std::string *>::iterator it2, e;

  for (it2 = HostPtrToModuleMap.begin(), e = HostPtrToModuleMap.end(); it2 != e;
       ++it2) {

    if (it2->second == module) {
      HostFunction = it2->first;
      HostPtrToModuleMap.erase(it2);
      auto it3 = HostPtrToNameMap.find(HostFunction);
      HostPtrToNameMap.erase(it3);
      PrimaryContext->destroyProgramBuiltin(HostFunction);
      for (ClContext *C : Contexts) {
        C->destroyProgramBuiltin(HostFunction);
      }
      break;
    }

  }
}

bool ClDevice::registerFunction(std::string *module, const void *HostFunction,
                                const char *FunctionName) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);

  auto it = std::find(Modules.begin(), Modules.end(), module);
  if (it == Modules.end()) {
    logError("Module PTR not FOUND: {}\n", (void *)module);
    return false;
  }

  HostPtrToModuleMap.emplace(std::make_pair(HostFunction, module));
  HostPtrToNameMap.emplace(std::make_pair(HostFunction, FunctionName));

  std::string temp(FunctionName);
  return (PrimaryContext->createProgramBuiltin(module, HostFunction, temp) ==
          hipSuccess);
}

bool ClDevice::getModuleAndFName(const void *HostFunction,
                                 std::string &FunctionName,
                                 std::string **module) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);

  auto it1 = HostPtrToModuleMap.find(HostFunction);
  auto it2 = HostPtrToNameMap.find(HostFunction);

  if ((it1 == HostPtrToModuleMap.end()) || (it2 == HostPtrToNameMap.end()))
    return false;

  FunctionName.assign(it2->second);
  *module = it1->second;
  return true;
}

ClDevice &CLDeviceById(int deviceId) { return *OpenCLDevices.at(deviceId); }

class InvalidDeviceType : public std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

class InvalidPlatformOrDeviceNumber : public std::out_of_range {
  using std::out_of_range::out_of_range;
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
