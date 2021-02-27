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

bool ClEvent::updateFinishStatus() {
  std::lock_guard<std::mutex> Lock(EventMutex);
  if (Status != EVENT_STATUS_RECORDING)
    return false;

  int Stat = Event->getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
  if (Stat <= CL_COMPLETE) {
    Status = EVENT_STATUS_RECORDED;
    return true;
  }
  return false;
}

bool ClEvent::recordStream(hipStream_t S, cl_event E) {
  std::lock_guard<std::mutex> Lock(EventMutex);

  Stream = S;
  Status = EVENT_STATUS_RECORDING;

  if (Event != nullptr) {
    cl_uint refc = Event->getInfo<CL_EVENT_REFERENCE_COUNT>();
    logDebug("removing old event, refc: {}\n", refc);

    delete Event;
  }

  Event = new cl::Event(E, true);
  return true;
}

bool ClEvent::wait() {
  std::lock_guard<std::mutex> Lock(EventMutex);
  if (Status != EVENT_STATUS_RECORDING)
    return false;

  Event->wait();
  Status = EVENT_STATUS_RECORDED;
  return true;
}

uint64_t ClEvent::getFinishTime() {
  std::lock_guard<std::mutex> Lock(EventMutex);
  int err;
  uint64_t ret = Event->getProfilingInfo<CL_PROFILING_COMMAND_END>(&err);
  assert(err == CL_SUCCESS);
  return ret;
}

/********************************/

static int setLocalSize(size_t shared, OCLFuncInfo *FuncInfo,
                        cl_kernel kernel) {

  int err = CL_SUCCESS;

  if (shared > 0) {
    logDebug("setLocalMemSize to {}\n", shared);
    size_t LastArgIdx = FuncInfo->ArgTypeInfo.size() - 1;
    if (FuncInfo->ArgTypeInfo[LastArgIdx].space != OCLSpace::Local) {
      // this can happen if for example the llvm optimizes away
      // the dynamic local variable
      logWarn("Can't set the dynamic local size, "
              "because the kernel doesn't use any local memory.\n");
    } else {
      err = ::clSetKernelArg(kernel, LastArgIdx, shared, nullptr);
      if (err != CL_SUCCESS) {
        logError("clSetKernelArg() failed to set dynamic local size!\n");
      }
    }
  }

  return err;
}

bool ClKernel::setup(size_t Index, OpenCLFunctionInfoMap &FuncInfoMap) {
  int err = 0;
  Name = Kernel.getInfo<CL_KERNEL_FUNCTION_NAME>(&err);
  if (err != CL_SUCCESS) {
    logError("clGetKernelInfo(CL_KERNEL_FUNCTION_NAME) failed: {}\n", err);
    return false;
  }

  logDebug("Kernel {} is: {} \n", Index, Name);

  auto it = FuncInfoMap.find(Name);
  assert(it != FuncInfoMap.end());
  FuncInfo = it->second;

  // TODO attributes
  cl_uint NumArgs = Kernel.getInfo<CL_KERNEL_NUM_ARGS>(&err);
  if (err != CL_SUCCESS) {
    logError("clGetKernelInfo(CL_KERNEL_NUM_ARGS) failed: {}\n", err);
    return false;
  }
  assert(FuncInfo->ArgTypeInfo.size() == NumArgs);

  if (NumArgs > 0) {
    logDebug("Kernel {} numArgs: {} \n", Name, NumArgs);
    logDebug("  RET_TYPE: {} {} {}\n", FuncInfo->retTypeInfo.size,
             (unsigned)FuncInfo->retTypeInfo.space,
             (unsigned)FuncInfo->retTypeInfo.type);
    for (auto &argty : FuncInfo->ArgTypeInfo) {
      logDebug("  ARG: SIZE {} SPACE {} TYPE {}\n", argty.size,
               (unsigned)argty.space, (unsigned)argty.type);
      TotalArgSize += argty.size;
    }
  }
  return true;
}

int ClKernel::setAllArgs(void **args, size_t shared) {
  void *p;
  int err;

  for (cl_uint i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];

    if (ai.type == OCLType::Pointer) {
      // TODO other than global AS ?
      assert(ai.size == sizeof(void *));
      p = *(void **)(args[i]);
      logDebug("setArg SVM {} to PTR {}\n", i, p);
      err = ::clSetKernelArgSVMPointer(Kernel(), i, p);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArgSVMPointer failed with error {}\n", err);
        return err;
      }
    } else {
      logDebug("setArg {} SIZE {}\n", i, ai.size);
      err = ::clSetKernelArg(Kernel(), i, ai.size, args[i]);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArg failed with error {}\n", err);
        return err;
      }
    }
  }

  return setLocalSize(shared, FuncInfo, Kernel());
}

int ClKernel::setAllArgs(void *args, size_t size, size_t shared) {
  void *p = args;
  int err;

  for (cl_uint i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];

    if (ai.type == OCLType::Pointer) {
      // TODO other than global AS ?
      assert(ai.size == sizeof(void *));
      void *pp = *(void **)p;
      logDebug("setArg SVM {} to PTR {}\n", i, pp);
      err = ::clSetKernelArgSVMPointer(Kernel(), i, pp);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArgSVMPointer failed with error {}\n", err);
        return err;
      }
    } else {
      logDebug("setArg {} SIZE {}\n", i, ai.size);
      err = ::clSetKernelArg(Kernel(), i, ai.size, p);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArg failed with error {}\n", err);
        return err;
      }
    }

    p = (char *)p + ai.size;
  }

  return setLocalSize(shared, FuncInfo, Kernel());
}

/********************************/

bool ClProgram::setup(std::string &binary) {

  size_t numWords = binary.size() / 4;
  int32_t *bindata = new int32_t[numWords + 1];
  std::memcpy(bindata, binary.data(), binary.size());
  bool res = parseSPIR(bindata, numWords, FuncInfos);
  delete[] bindata;
  if (!res) {
    logError("SPIR-V parsing failed\n");
    return false;
  }

  int err;
  std::vector<char> binary_vec(binary.begin(), binary.end());
  Program = cl::Program(Context, binary_vec, false, &err);
  if (err != CL_SUCCESS) {
    logError("CreateProgramWithIL Failed: {}\n", err);
    return false;
  }

  std::string name = Device.getInfo<CL_DEVICE_NAME>();

  int build_failed = Program.build("-x spir -cl-kernel-arg-info");

  std::string log = Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(Device, &err);
  if (err != CL_SUCCESS) {
    logError("clGetProgramBuildInfo() Failed: {}\n", err);
    return false;
  }
  logDebug("Program BUILD LOG for device {}:\n{}\n", name, log);
  if (build_failed != CL_SUCCESS) {
    logError("clBuildProgram() Failed: {}\n", build_failed);
    return false;
  }

  std::vector<cl::Kernel> kernels;
  err = Program.createKernels(&kernels);
  if (err != CL_SUCCESS) {
    logError("clCreateKernels() Failed: {}\n", err);
    return false;
  }
  logDebug("Kernels in program: {} \n", kernels.size());
  Kernels.resize(kernels.size());

  for (size_t i = 0; i < kernels.size(); ++i) {
    ClKernel *k = new ClKernel(Context, std::move(kernels[i]));
    if (k == nullptr)
      return false; // TODO memleak
    if (!k->setup(i, FuncInfos))
      return false;
    Kernels[i] = k;
  }
  return true;
}

ClProgram::~ClProgram() {
  for (hipFunction_t K : Kernels) {
    delete K;
  }
  Kernels.clear();

  std::set<OCLFuncInfo *> PtrsToDelete;
  for (auto &kv : FuncInfos)
    PtrsToDelete.insert(kv.second);
  for (auto &Ptr : PtrsToDelete)
    delete Ptr;
}

hipFunction_t ClProgram::getKernel(std::string &name) {
  for (hipFunction_t It : Kernels) {
    if (It->isNamed(name)) {
      return It;
    }
  }
  return nullptr;
}

hipFunction_t ClProgram::getKernel(const char *name) {
  std::string SearchName(name);
  return getKernel(SearchName);
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

hipError_t ClQueue::memCopy(void *dst, const void *src, size_t size) {
  std::lock_guard<std::mutex> Lock(QueueMutex);

  logDebug("clSVMmemcpy {} -> {} / {} B\n", src, dst, size);
  cl_event ev = nullptr;
  int retval =
      ::clEnqueueSVMMemcpy(Queue(), CL_FALSE, dst, src, size, 0, nullptr, &ev);
  if (retval == CL_SUCCESS) {
    if (LastEvent != nullptr) {
      logDebug("memCopy: LastEvent == {}, will be: {}", (void *)LastEvent,
               (void *)ev);
      clReleaseEvent(LastEvent);
    } else
      logDebug("memCopy: LastEvent == NULL, will be: {}\n", (void *)ev);
    LastEvent = ev;
  } else {
    logError("clEnqueueSVMMemCopy() failed with error {}\n", retval);
  }
  return (retval == CL_SUCCESS) ? hipSuccess : hipErrorLaunchFailure;
}

hipError_t ClQueue::memFill(void *dst, size_t size, const void *pattern,
                            size_t patt_size) {
  std::lock_guard<std::mutex> Lock(QueueMutex);

  logDebug("clSVMmemfill {} / {} B\n", dst, size);
  cl_event ev = nullptr;
  int retval = ::clEnqueueSVMMemFill(Queue(), dst, pattern, patt_size, size, 0,
                                     nullptr, &ev);
  if (retval == CL_SUCCESS) {
    if (LastEvent != nullptr) {
      logDebug("memFill: LastEvent == {}, will be: {}", (void *)LastEvent,
               (void *)ev);
      clReleaseEvent(LastEvent);
    } else
      logDebug("memFill: LastEvent == NULL, will be: {}\n", (void *)ev);
    LastEvent = ev;
  } else {
    logError("clEnqueueSVMMemFill() failed with error {}\n", retval);
  }

  return (retval == CL_SUCCESS) ? hipSuccess : hipErrorLaunchFailure;
}

bool ClQueue::finish() {
  int err = Queue.finish();
  if (err != CL_SUCCESS)
    logError("clFinish() failed with error {}\n", err);
  return err == CL_SUCCESS;
}

static void notifyOpenCLevent(cl_event event, cl_int status, void *data) {
  hipStreamCallbackData *Data = (hipStreamCallbackData *)data;
  Data->Callback(Data->Stream, Data->Status, Data->UserData);
  delete Data;
}

bool ClQueue::addCallback(hipStreamCallback_t callback, void *userData) {
  HIP_PROCESS_ERROR_MSG("Supported in LZQueue::addCallback!", hipErrorNotSupported);   
}

bool ClQueue::enqueueBarrierForEvent(hipEvent_t ProvidedEvent) {
  std::lock_guard<std::mutex> Lock(QueueMutex);
  // CUDA API cudaStreamWaitEvent:
  // event may be from a different device than stream.

  cl::Event MarkerEvent;
  logDebug("Queue is: {}\n", (void *)(Queue()));
  int err = Queue.enqueueMarkerWithWaitList(nullptr, &MarkerEvent);
  if (err != CL_SUCCESS)
    return false;

  cl::vector<cl::Event> Events = {MarkerEvent, ProvidedEvent->getEvent()};
  cl::Event barrier;
  err = Queue.enqueueBarrierWithWaitList(&Events, &barrier);
  if (err != CL_SUCCESS) {
    logError("clEnqueueBarrierWithWaitList failed with error {}\n", err);
    return false;
  }

  if (LastEvent)
    clReleaseEvent(LastEvent);
  LastEvent = barrier();

  return true;
}

bool ClQueue::recordEvent(hipEvent_t event) {
  HIP_PROCESS_ERROR_MSG("Supported in LZQueue::recordEvent!", hipErrorNotSupported);
}

hipError_t ClQueue::launch(ClKernel *Kernel, ExecItem *Arguments) {
  std::lock_guard<std::mutex> Lock(QueueMutex);
  
  if (Arguments->setupAllArgs(Kernel) != CL_SUCCESS) {
    logError("Failed to set kernel arguments for launch! \n");
    return hipErrorLaunchFailure;
  }

  dim3 GridDim = Arguments->GridDim;
  dim3 BlockDim = Arguments->BlockDim;

  const cl::NDRange global(GridDim.x * BlockDim.x, GridDim.y * BlockDim.y,
                           GridDim.z * BlockDim.z);
  const cl::NDRange local(BlockDim.x, BlockDim.y, BlockDim.z);

  cl::Event ev;
  int err = Queue.enqueueNDRangeKernel(Kernel->get(), cl::NullRange, global,
                                       local, nullptr, &ev);

  if (err != CL_SUCCESS)
    logError("clEnqueueNDRangeKernel() failed with: {}\n", err);
  hipError_t retval = (err == CL_SUCCESS) ? hipSuccess : hipErrorLaunchFailure;

  if (retval == hipSuccess) {
    if (LastEvent != nullptr) {
      logDebug("Launch: LastEvent == {}, will be: {}", (void *)LastEvent,
               (void *)ev.get());
      clReleaseEvent(LastEvent);
    } else
      logDebug("launch: LastEvent == NULL, will be: {}\n", (void *)ev.get());
    LastEvent = ev.get();
    clRetainEvent(LastEvent);
  }

  delete Arguments;
  return retval;
}

hipError_t ClQueue::launch3(ClKernel *Kernel, dim3 grid, dim3 block) {
  std::lock_guard<std::mutex> Lock(QueueMutex);

  dim3 GridDim = grid;
  dim3 BlockDim = block;

  const cl::NDRange global(GridDim.x * BlockDim.x, GridDim.y * BlockDim.y,
                           GridDim.z * BlockDim.z);
  const cl::NDRange local(BlockDim.x, BlockDim.y, BlockDim.z);

  cl::Event ev;
  int err = Queue.enqueueNDRangeKernel(Kernel->get(), cl::NullRange, global,
                                       local, nullptr, &ev);

  if (err != CL_SUCCESS)
    logError("clEnqueueNDRangeKernel() failed with: {}\n", err);
  hipError_t retval = (err == CL_SUCCESS) ? hipSuccess : hipErrorLaunchFailure;

  if (retval == hipSuccess) {
    if (LastEvent != nullptr) {
      logDebug("Launch3: LastEvent == {}, will be: {}", (void *)LastEvent,
               (void *)ev.get());
      clReleaseEvent(LastEvent);
    } else
      logDebug("launch3: LastEvent == NULL, will be: {}\n", (void *)ev.get());
    LastEvent = ev.get();
    clRetainEvent(LastEvent);
  }

  return retval;
}

/***********************************************************************/

void ExecItem::setArg(const void *arg, size_t size, size_t offset) {
  if ((offset + size) > ArgData.size())
    ArgData.resize(offset + size + 1024);

  std::memcpy(ArgData.data() + offset, arg, size);
  logDebug("setArg on {} size {} offset {}\n", (void *)this, size, offset);
  OffsetsSizes.push_back(std::make_tuple(offset, size));
}

int ExecItem::setupAllArgs(ClKernel *kernel) {
  OCLFuncInfo *FuncInfo = kernel->getFuncInfo();
  size_t NumLocals = 0;
  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    if (FuncInfo->ArgTypeInfo[i].space == OCLSpace::Local)
      ++NumLocals;
  }
  // there can only be one dynamic shared mem variable, per cuda spec
  assert (NumLocals <= 1);

  if ((OffsetsSizes.size()+NumLocals) != FuncInfo->ArgTypeInfo.size()) {
      logError("Some arguments are still unset\n");
      return CL_INVALID_VALUE;
  }

  if (OffsetsSizes.size() == 0)
    return CL_SUCCESS;

  std::sort(OffsetsSizes.begin(), OffsetsSizes.end());
  if ((std::get<0>(OffsetsSizes[0]) != 0) ||
      (std::get<1>(OffsetsSizes[0]) == 0)) {
          logError("Invalid offset/size\n");
          return CL_INVALID_VALUE;
      }

  // check args are set
  if (OffsetsSizes.size() > 1) {
    for (size_t i = 1; i < OffsetsSizes.size(); ++i) {
      if ( (std::get<0>(OffsetsSizes[i]) == 0) ||
           (std::get<1>(OffsetsSizes[i]) == 0) ||
           (
           (std::get<0>(OffsetsSizes[i - 1]) + std::get<1>(OffsetsSizes[i - 1])) >
            std::get<0>(OffsetsSizes[i]))
           ) {
          logError("Invalid offset/size\n");
          return CL_INVALID_VALUE;
        }
    }
  }

  const unsigned char *start = ArgData.data();
  void *p;
  int err;
  for (cl_uint i = 0; i < OffsetsSizes.size(); ++i) {
    OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];
    logDebug("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n", i,
             std::get<0>(OffsetsSizes[i]), std::get<1>(OffsetsSizes[i]),
             (unsigned)ai.type, (unsigned)ai.space, ai.size);

    if (ai.type == OCLType::Pointer) {

      // TODO other than global AS ?
      assert(ai.size == sizeof(void *));
      assert(std::get<1>(OffsetsSizes[i]) == ai.size);
      p = *(void **)(start + std::get<0>(OffsetsSizes[i]));
      logDebug("setArg SVM {} to {}\n", i, p);
      err = ::clSetKernelArgSVMPointer(kernel->get().get(), i, p);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArgSVMPointer failed with error {}\n", err);
        return err;
      }
    } else {
      size_t size = std::get<1>(OffsetsSizes[i]);
      size_t offs = std::get<0>(OffsetsSizes[i]);
      void* value = (void*)(start + offs);
      logDebug("setArg {} size {} offs {}\n", i, size, offs);
      err =
          ::clSetKernelArg(kernel->get().get(), i, size, value);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArg failed with error {}\n", err);
        return err;
      }
    }
  }

  return setLocalSize(SharedMem, FuncInfo, kernel->get().get());
}

inline hipError_t ExecItem::launch(ClKernel *Kernel) {
  return Stream->launch(Kernel, this);
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

  DefaultQueue = new LZQueue(CmdQueue, 0, 0); // new ClQueue(CmdQueue, 0, 0);

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

  DefaultQueue = new LZQueue(CmdQueue, 0, 0); // ClQueue(CmdQueue, 0, 0);
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
  std::lock_guard<std::mutex> Lock(ContextMutex);
  return new ClEvent(Context, flags);
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

hipError_t ClContext::memCopy(void *dst, const void *src, size_t size,
                              hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);

  if (Memory.hasPointer(dst) || Memory.hasPointer(src))
    return Queue->memCopy(dst, src, size);
  else
    return hipErrorInvalidDevicePointer;
}

hipError_t ClContext::memFill(void *dst, size_t size, const void *pattern,
                              size_t pat_size, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);

  if (!Memory.hasPointer(dst))
    return hipErrorInvalidDevicePointer;

  return Queue->memFill(dst, size, pattern, pat_size);
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

  hipStream_t Ptr = new LZQueue(NewQueue, flags, priority); // new ClQueue(NewQueue, flags, priority);
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
  std::vector<cl::CommandQueue> Copies;
  {
    std::lock_guard<std::mutex> Lock(ContextMutex);
    for (hipStream_t I : Queues) {
      Copies.push_back(I->getQueue());
    }
    // Note that this does not really go through due to the subclass : LZQueue
    Copies.push_back(DefaultQueue->getQueue());
  }

  for (cl::CommandQueue &I : Copies) {
    int err = I.finish();
    if (err != CL_SUCCESS) {
      logError("clFinish() failed with error {}\n", err);
      return false;
    }
  }
  return true;
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
  std::lock_guard<std::mutex> Lock(ContextMutex);

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
  return hipSuccess;
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
  FIND_QUEUE_LOCKED(stream);

  if (!kernel->isFromContext(Context))
    return hipErrorLaunchFailure;

  int err = kernel->setAllArgs(kernelParams, shared);
  if (err != CL_SUCCESS) {
    logError("Failed to set kernel arguments for launch! \n");
    return hipErrorLaunchFailure;
  }

  return stream->launch3(kernel, grid, block);
}

hipError_t ClContext::launchWithExtraParams(dim3 grid, dim3 block,
                                            size_t shared, hipStream_t stream,
                                            void **extraParams,
                                            hipFunction_t kernel) {
  FIND_QUEUE_LOCKED(stream);

  if (!kernel->isFromContext(Context))
    return hipErrorLaunchFailure;

  void *args = nullptr;
  size_t size = 0;

  void **p = extraParams;
  while (*p && (*p != HIP_LAUNCH_PARAM_END)) {
    if (*p == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
      args = (void *)p[1];
      p += 2;
      continue;
    } else if (*p == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
      size = (size_t)p[1];
      p += 2;
      continue;
    } else {
      logError("Unknown parameter in extraParams: {}\n", *p);
      return hipErrorLaunchFailure;
    }
  }

  if (args == nullptr || size == 0) {
    logError("extraParams doesn't contain all required parameters\n");
    return hipErrorLaunchFailure;
  }

  // TODO This only accepts structs with no padding.
  if (size != kernel->getTotalArgSize()) {
    logError("extraParams doesn't have correct size\n");
    return hipErrorLaunchFailure;
  }

  int err = kernel->setAllArgs(args, size, shared);
  if (err != CL_SUCCESS) {
    logError("Failed to set kernel arguments for launch! \n");
    return hipErrorLaunchFailure;
  }
  return stream->launch3(kernel, grid, block);
}

ClProgram *ClContext::createProgram(std::string &binary) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  ClProgram *prog = new ClProgram(Context, Device->getDevice());
  if (prog == nullptr)
    return nullptr;

  if (!prog->setup(binary)) {
    delete prog;
    return nullptr;
  }

  Programs.emplace(prog);
  return prog;
}

hipError_t ClContext::destroyProgram(ClProgram *prog) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  auto it = Programs.find(prog);
  if (it == Programs.end())
    return hipErrorInvalidHandle;

  Programs.erase(it);
  return hipSuccess;
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

/***********************************************************************/
// HipLZ support
static std::vector<LZDevice *> HipLZDevices INIT_PRIORITY(120);
// The drivers are managed globally
static std::vector<LZDriver *> HipLZDrivers INIT_PRIORITY(120);

size_t NumLZDevices = 1;

size_t NumLZDrivers = 1;

LZDevice::LZDevice(hipDevice_t id, ze_device_handle_t hDevice_, LZDriver* driver_) {
  this->deviceId = id;
  this->hDevice = hDevice_;
  this->driver = driver_;
  ze_result_t status = ZE_RESULT_SUCCESS;

  // Query device properties
  status = zeDeviceGetProperties(this->hDevice, &(this->deviceProps));
  LZ_PROCESS_ERROR_MSG("HipLZ zeDeviceGetProperties Failed with return code ", status);

  // Query device memory properties
  uint32_t count = 1;
  status = zeDeviceGetMemoryProperties(this->hDevice, &count, &(this->deviceMemoryProps));
  this->TotalUsedMem = 0;

  // Query device computation properties
  status = zeDeviceGetComputeProperties(this->hDevice, &(this->deviceComputeProps));

  // Query device cache properties
  count = 1;
  status = zeDeviceGetCacheProperties(this->hDevice, &count, &(this->deviceCacheProps));

  // Create HipLZ context  
  ze_context_desc_t ctxtDesc = {
    ZE_STRUCTURE_TYPE_CONTEXT_DESC,
    nullptr,
    0
  };
  ze_context_handle_t hContext;
  status = zeContextCreate(this->driver->GetDriverHandle(), &ctxtDesc, &hContext);
  LZ_PROCESS_ERROR_MSG("HipLZ zeContextCreate Failed with return code ", status);
  logDebug("LZ CONTEXT {} via calling zeContextCreate ", status);
  this->defaultContext = new LZContext(this, hContext);

  // Get the copute queue group ordinal
  retrieveCmdQueueGroupOrdinal(this->cmdQueueGraphOrdinal);
      
  // Setup HipLZ device properties
  setupProperties(id);
}

void LZDevice::registerModule(std::string* module) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  Modules.push_back(module);
}

bool LZDevice::registerFunction(std::string *module, const void *HostFunction,
				const char *FunctionName) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);

  logDebug("LZ REGISER FUCNTION {}", FunctionName);
  auto it = std::find(Modules.begin(), Modules.end(), module);
  if (it == Modules.end()) {
    logError("HipLZ Module PTR not FOUND: {}\n", (void *)module);
    return false;
  }

  HostPtrToModuleMap.emplace(std::make_pair(HostFunction, module));
  HostPtrToNameMap.emplace(std::make_pair(HostFunction, FunctionName));

  // Create HipLZ module
  std::string funcName(FunctionName);
  this->defaultContext->CreateModule((uint8_t* )module->data(), module->length(), funcName);

  return true;
}

// Get host function pointer's corresponding name 
std::string LZDevice::GetHostFunctionName(const void* HostFunction) {
  if (HostPtrToNameMap.find(HostFunction) == HostPtrToNameMap.end())
    HIP_PROCESS_ERROR_MSG("HipLZ no corresponding host function name found", hipErrorInitializationError);

  return HostPtrToNameMap[HostFunction];
}

// Get current device driver handle 
ze_driver_handle_t LZDevice::GetDriverHandle() { 
  return this->driver->GetDriverHandle(); 
}

// Reset current device
void LZDevice::reset() {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  this->defaultContext->reset();
}

// Setup HipLZ device properties
void LZDevice::setupProperties(int index) {
  // Copy device name
  if (255 < ZE_MAX_DEVICE_NAME) {
    strncpy(Properties.name, this->deviceProps.name, 255);
    Properties.name[255] = 0;
  } else {
    strncpy(Properties.name, this->deviceProps.name, ZE_MAX_DEVICE_NAME);
    Properties.name[ZE_MAX_DEVICE_NAME-1] = 0;
  }

  // Get total device memory
  Properties.totalGlobalMem = this->deviceMemoryProps.totalSize;

  Properties.sharedMemPerBlock = this->deviceComputeProps.maxSharedLocalMemory; 
  //??? Dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&err);

  Properties.maxThreadsPerBlock = this->deviceComputeProps.maxTotalGroupSize;
  //??? Dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);

  Properties.maxThreadsDim[0] = this->deviceComputeProps.maxGroupSizeX;
  Properties.maxThreadsDim[1] = this->deviceComputeProps.maxGroupSizeY;
  Properties.maxThreadsDim[2] = this->deviceComputeProps.maxGroupSizeZ;

  // Maximum configured clock frequency of the device in MHz. 
  Properties.clockRate = 1000 * this->deviceMemoryProps.maxClockRate;
  // Dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  Properties.multiProcessorCount = this->deviceComputeProps.maxTotalGroupSize;
  //??? Dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  Properties.l2CacheSize = this->deviceCacheProps.cacheSize;
  // Dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  Properties.totalConstMem = 0;
  // ??? Dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // totally made up 
  Properties.regsPerBlock = 64;

  Properties.warpSize = this->deviceComputeProps.maxTotalGroupSize;

  // Replicate from OpenCL implementation
  Properties.maxGridSize[0] = 65536;
  Properties.maxGridSize[1] = 65536;
  Properties.maxGridSize[2] = 65536;
  Properties.memoryClockRate = this->deviceMemoryProps.maxClockRate;
  Properties.memoryBusWidth = this->deviceMemoryProps.maxBusWidth;
  Properties.major = 2;
  Properties.minor = 0;

  Properties.maxThreadsPerMultiProcessor = 10;

  Properties.computeMode = 0;
  Properties.arch = {};

  Properties.arch.hasGlobalInt32Atomics = 1;
  Properties.arch.hasSharedInt32Atomics = 1;

  Properties.arch.hasGlobalInt64Atomics = 1;
  Properties.arch.hasSharedInt64Atomics = 1;

  Properties.arch.hasDoubles = 1;

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

// Copy device properties to given property data structure 
void LZDevice::copyProperties(hipDeviceProp_t *prop) {
  if (prop)
    std::memcpy(prop, &this->Properties, sizeof(hipDeviceProp_t));
}

// Get Hip attribute from attribute enum ID 
int LZDevice::getAttr(int *pi, hipDeviceAttribute_t attr) {
  auto I = Attributes.find(attr);
  if (I != Attributes.end()) {
    *pi = I->second;
    return 0;
  } else {
    return 1;
  }
}

bool LZDevice::retrieveCmdQueueGroupOrdinal(uint32_t& computeQueueGroupOrdinal) {
  // Discover all command queue groups
  uint32_t cmdqueueGroupCount = 0;
  zeDeviceGetCommandQueueGroupProperties(hDevice, &cmdqueueGroupCount, nullptr);

  if (cmdqueueGroupCount > 32)
    return false;
  ze_command_queue_group_properties_t cmdqueueGroupProperties[32];
  zeDeviceGetCommandQueueGroupProperties(hDevice, &cmdqueueGroupCount, cmdqueueGroupProperties);
  
  // Find a command queue type that support compute
  computeQueueGroupOrdinal = cmdqueueGroupCount;
  for (uint32_t i = 0; i < cmdqueueGroupCount; ++i ) {
    if (cmdqueueGroupProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE ) {
      computeQueueGroupOrdinal = i;
      break;
    }
  }
  
  if (computeQueueGroupOrdinal == cmdqueueGroupCount)
    return false; // no compute queues found

  return true;
}

hipError_t LZContext::memCopy(void *dst, const void *src, size_t sizeBytes, hipStream_t stream) {
  if (stream == nullptr) {
    // Here we use default queue in  LZ context to do the synchronous copy
    ze_result_t status = zeCommandListAppendMemoryCopy(lzCommandList->GetCommandListHandle(), dst, src,
						       sizeBytes, NULL, 0, NULL);
    LZ_RETURN_ERROR_MSG("HipLZ zeCommandListAppendMemoryCopy FAILED with return code ", status);
    logDebug("LZ MEMCPY {} via calling zeCommandListAppendMemoryCopy ", status);

    // Execute memory copy asynchronously via lz context's default command list
    if (!lzCommandList->Execute(lzQueue))
      return hipErrorInvalidDevice;
  } else {
    if (!stream->memCopy(dst, src, sizeBytes))
      return hipErrorInvalidDevice;
  }

  return hipSuccess;
}

hipError_t LZContext::memFill(void *dst, size_t size, const void *pattern, size_t pattern_size, hipStream_t stream) {
  if (stream == nullptr) {
    // Here we use default queue in  LZ context to do the synchronous copy
    ze_result_t status = zeCommandListAppendMemoryFill(lzCommandList->GetCommandListHandle(), dst, pattern, pattern_size, size, NULL, 0, NULL);
    LZ_RETURN_ERROR_MSG("HipLZ zeCommandListAppendMemoryFill FAILED with return code ", status);
    logDebug("LZ MEMFILL {} via calling zeCommandListAppendMemoryFill ", status);

    if (!lzCommandList->Execute(lzQueue))
      return hipErrorInvalidDevice;
  } else {
    if (!stream->memFill(dst, size, pattern, pattern_size))
      return hipErrorInvalidDevice;
  }
  return hipSuccess;
}

hipError_t LZContext::memFill(void *dst, size_t size, const void *pattern, size_t pattern_size) {
  lzCommandList->ExecuteMemFill(lzQueue, dst, size, pattern, pattern_size);
  return hipSuccess;
}

hipError_t LZContext::memFillAsync(void *dst, size_t size, const void *pattern, size_t pattern_size, hipStream_t stream) {
  if (stream == nullptr) {
    // Here we use default queue in  LZ context to do the asynchronous copy
    ze_result_t status = zeCommandListAppendMemoryFill(lzCommandList->GetCommandListHandle(), dst, pattern, pattern_size, size, NULL, 0, NULL);
    logDebug("LZ MEMFILL {} via calling zeCommandListAppendMemoryFill ", status);

    // Execute memory copy asynchronously via lz context's default command list
    if (!lzCommandList->ExecuteAsync(lzQueue))
      return hipErrorInvalidDevice;
  } else {
    if (!stream->memFillAsync(dst, size, pattern, pattern_size))
      return hipErrorInvalidDevice;
  }
  return hipSuccess;
}

hipError_t LZContext::memCopyAsync(void *dst, const void *src, size_t sizeBytes, hipStream_t stream) {
  if (stream == nullptr) {
    // Here we use default queue in  LZ context to do the asynchronous copy
    ze_result_t status = zeCommandListAppendMemoryCopy(lzCommandList->GetCommandListHandle(), dst, src,
                                                       sizeBytes, NULL, 0, NULL);
    LZ_RETURN_ERROR_MSG("HipLZ zeCommandListAppendMemoryCopy FAILED with return code ", status);
    logDebug("LZ MEMCPY {} via calling zeCommandListAppendMemoryCopy ", status);

    // Execute memory copy asynchronously via lz context's default command list     
    if (!lzCommandList->ExecuteAsync(lzQueue))
      return hipErrorInvalidDevice;
  } else {
    if (!stream->memCopyAsync(dst, src, sizeBytes))
      return hipErrorInvalidDevice;
  }
  return hipSuccess;
}

hipError_t LZContext::memCopy(void *dst, const void *src, size_t sizeBytes) {
  // Execute memory copy
  lzCommandList->ExecuteMemCopy(lzQueue, dst, src, sizeBytes);
  return hipSuccess;
}

LZContext::LZContext(LZDevice* D, ze_context_handle_t hContext_) : ClContext(0, 0) {
  this->lzDevice = D;
  this->hContext = hContext_;
  this->lzModule = 0;
  
  // Create command list 
  // ze_command_queue_desc_t cqDesc;
  // cqDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;                                    
  // cqDesc.pNext = nullptr;
  // cqDesc.ordinal = 0;
  // cqDesc.index = 0;
  //  cqDesc.flags = 0; // default hehaviour
  // cqDesc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
  // cqDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  
  // ze_command_list_handle_t hCommandList;
  // ze_result_t status = zeCommandListCreateImmediate(this->hContext, lzDevice->GetDeviceHandle(), &cqDesc, &hCommandList);
  // if (status != ZE_RESULT_SUCCESS) {
  //   throw InvalidLevel0Initialization("HipLZ zeCommandListCreate FAILED with return code " + std::to_string(status));
    // }
  // Create the Level-0 queue
  // ze_command_queue_handle_t hQueue;
  // status = zeCommandQueueCreate(this->hContext, lzDevice->GetDeviceHandle(), &cqDesc, &hQueue);

  // TODO: just use DefaultQueue to maintain the context local queu and command list?
  // Create a command list for default command queue
  this->lzCommandList = new LZCommandList(this);
  // Create the default command queue
  this->lzQueue = this->DefaultQueue = new LZQueue(this, this->lzCommandList);
  // Create the default event pool
  this->defaultEventPool = new LZEventPool(this);

  // if (status != ZE_RESULT_SUCCESS) {
  //   throw InvalidLevel0Initialization("HipLZ zeCommandQueueCreate with return code " + std::to_string(status));
  //  }
  // logDebug("LZ COMMAND LIST {} ", status);
  // this->lzCommandList = new LZCommandList(this, hCommandList);
}

bool LZContext::CreateModule(uint8_t* funcIL, size_t ilSize, std::string funcName) {
  logDebug("LZ CREATE MODULE {} ", funcName);
  // Parse the SPIR-V fat binary to retrieve kernel function information
  size_t numWords = ilSize / 4;
  int32_t * binarydata = new int32_t[numWords + 1];
  std::memcpy(binarydata, funcIL, ilSize);
  // Extract kernel function information 
  bool res = parseSPIR(binarydata, numWords, FuncInfos);
  delete[] binarydata;
  if (!res) {
    logError("SPIR-V parsing failed\n");
    return false;
  }

  logDebug("LZ PARSE SPIR {} ", funcName);
 
  if (!this->lzModule) {
    // Create module 
    ze_module_desc_t moduleDesc = {
      ZE_STRUCTURE_TYPE_MODULE_DESC,
      nullptr,
      ZE_MODULE_FORMAT_IL_SPIRV,
      ilSize,
      funcIL,
      nullptr,
      nullptr
    };
    ze_module_handle_t hModule;
    ze_result_t status = zeModuleCreate(hContext, lzDevice->GetDeviceHandle(), &moduleDesc, &hModule, nullptr);
    LZ_PROCESS_ERROR_MSG("Hiplz zeModuleCreate FAILED with return code  ", status);

    logDebug("LZ CREATE MODULE via calling zeModuleCreate {} ", status);
    // Create module object
    this->lzModule = new LZModule(hModule);
  }
  
  // Create kernel object
  this->lzModule->CreateKernel(funcName, FuncInfos);

  return true;
}

// Configure the call to LZ kernel, here we ignore OpenCL queue but using LZ command list
hipError_t LZContext::configureCall(dim3 grid, dim3 block, size_t shared, hipStream_t stream) {
  // TODO: make thread safeness
  if (stream == nullptr)
    stream = this->DefaultQueue;
  LZExecItem *NewItem = new LZExecItem(grid, block, shared, stream);
  // Here we reuse the execution item stack from super class, i.e. OpenCL context 
  ExecStack.push(NewItem);
  
  return hipSuccess;
}

// Set argument
hipError_t LZContext::setArg(const void *arg, size_t size, size_t offset) {
  std::lock_guard<std::mutex> Lock(this->mtx);
  LZExecItem* lzExecItem = (LZExecItem* )this->ExecStack.top();
  lzExecItem->setArg(arg, size, offset);

  return hipSuccess;
}

// Launch HipLZ kernel 
bool LZContext::launchHostFunc(const void* HostFunction) {
  std::lock_guard<std::mutex> Lock(this->mtx);
  LZKernel* Kernel = 0;
  logDebug("LAUNCH HOST FUNCTION {} ",  this->lzModule != nullptr);
  if (!this->lzModule) {
    HIP_PROCESS_ERROR_MSG("Hiplz LZModule was not created before invoking kernel?", hipErrorInitializationError);
  }

  std::string HostFunctionName = this->lzDevice->GetHostFunctionName(HostFunction);
  Kernel = this->lzModule->GetKernel(HostFunctionName);
  logDebug("LAUNCH HOST FUNCTION {} - {} ", HostFunctionName,  Kernel != nullptr);
  
  if (!Kernel)
    HIP_PROCESS_ERROR_MSG("Hiplz no LZkernel found?", hipErrorInitializationError);

  LZExecItem *Arguments;
  Arguments = ExecStack.top();
  ExecStack.pop();

  // ze_result_t status = ZE_RESULT_SUCCESS;
  // status = zeKernelSetGroupSize(Kernel->GetKernelHandle(),
  // 				Arguments->BlockDim.x, Arguments->BlockDim.y, Arguments->BlockDim.z);
  // if (status != ZE_RESULT_SUCCESS) {
  //    throw InvalidLevel0Initialization("could not set group size!");
  //  }
  
  
  // Arguments->setupAllArgs(Kernel);
 
  // Launch kernel via Level-0 command list
  // uint32_t numGroupsX = Arguments->GridDim.x;
  // uint32_t numGroupsY = Arguments->GridDim.y;
  // uint32_t numGroupsz = Arguments->GridDim.z;
  // ze_group_count_t hLaunchFuncArgs = { numGroupsX, numGroupsY, numGroupsz };
  // ze_event_handle_t hSignalEvent = nullptr;
  // status = zeCommandListAppendLaunchKernel(this->lzCommandList->GetCommandListHandle(), 
  // 					   Kernel->GetKernelHandle(), 
  //					   &hLaunchFuncArgs, 
  // 					   hSignalEvent, 
  // 					   0, 
  // 					   nullptr);
  // if (status != ZE_RESULT_SUCCESS) {
  //   throw InvalidLevel0Initialization("Hiplz zeCommandListAppendLaunchKernel FAILED with return code  " + std::to_string(status));
  // } 

  // Execute kernel
  // lzCommandList->Execute(lzQueue);

  return Arguments->launch(Kernel);
}

// Allocate memory via Level-0 runtime  
void* LZContext::allocate(size_t size, size_t alignment, LZMemoryType memTy) {
  void *ptr = 0;
  if (memTy == LZMemoryType::Shared) {
    ze_device_mem_alloc_desc_t dmaDesc;
    dmaDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    dmaDesc.pNext = NULL;
    dmaDesc.flags = 0;
    dmaDesc.ordinal = 0;
    ze_host_mem_alloc_desc_t hmaDesc;
    hmaDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    hmaDesc.pNext = NULL;
    hmaDesc.flags = 0;
    ze_result_t status = zeMemAllocShared(this->hContext, &dmaDesc, &hmaDesc, size, alignment,
					  this->lzDevice->GetDeviceHandle(), &ptr);
    LZ_PROCESS_ERROR_MSG("HipLZ could not allocate shared memory with error code: ", status);
    logDebug("LZ MEMORY ALLOCATE via calling zeMemAllocShared {} ", status);
    
    return ptr;
  } else if (memTy == LZMemoryType::Device) {
    ze_device_mem_alloc_desc_t dmaDesc;
    dmaDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    dmaDesc.pNext = NULL;
    dmaDesc.flags = 0;
    dmaDesc.ordinal = 0;
    ze_result_t status = zeMemAllocDevice(this->hContext, &dmaDesc, size, alignment,
					  this->lzDevice->GetDeviceHandle(), &ptr);
    LZ_PROCESS_ERROR_MSG("HipLZ could not allocate device memory with error code: ", status);
    logDebug("LZ MEMORY ALLOCATE via calling zeMemAllocDevice {} ", status);
    
    return ptr;
  }

  HIP_PROCESS_ERROR_MSG("HipLZ could not recognize allocation options", hipErrorNotSupported);
}

bool LZContext::getPointerSize(void *ptr, size_t *size) {
  void *pBase;
  size_t pSize;
  ze_result_t status = zeMemGetAddressRange(this->hContext, ptr, &pBase, &pSize);
  LZ_PROCESS_ERROR_MSG("HipLZ could not get pointer info with error code: ", status);
  logDebug("LZ MEMORY GET INFO via calling zeMemGetAddressRange {} ", status);
  *size = pSize;
  return true;
}

bool LZContext::findPointerInfo(hipDeviceptr_t dptr, hipDeviceptr_t *pbase, size_t *psize) {
  ze_result_t status = zeMemGetAddressRange(this->hContext, dptr, pbase, psize);
  LZ_PROCESS_ERROR_MSG("HipLZ could not get pointer info with error code: ", status);
  logDebug("LZ MEMORY GET INFO via calling zeMemGetAddressRange {} ", status);
  return true;
}

void * LZContext::allocate(size_t size) {
  return allocate(size, 0x1000, LZMemoryType::Device); // Shared);
}

bool LZContext::free(void *p) {
  ze_result_t status = zeMemFree(this->hContext, p);
  LZ_PROCESS_ERROR_MSG("HipLZ could not free memory with error code: ", status);

  return true;
}

// Create stream/queue
bool LZContext::createQueue(hipStream_t *stream, unsigned int Flags, int priority) {
  hipStream_t Ptr = new LZQueue(this, true);
  Queues.insert(Ptr);
  *stream = Ptr;
  
  return true;
}

// Create HipLZ event 
LZEvent* LZContext::createEvent(unsigned flags) {
  if (!this->defaultEventPool)
    HIP_PROCESS_ERROR_MSG("HipLZ could not get event pool in current context", hipErrorInitializationError);

  // std::lock_guard<std::mutex> Lock(ContextMutex); 

  return this->defaultEventPool->createEvent(flags);
}

// Get the elapse between two events 
hipError_t LZContext::eventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  // assert(start->isFromContext(this));
  // assert(stop->isFromContext(this));

  // if (!start->isRecordingOrRecorded() || !stop->isRecordingOrRecorded())
  //   return hipErrorInvalidResourceHandle;

  // start->updateFinishStatus();
  // stop->updateFinishStatus();
  // if (!start->isFinished() || !stop->isFinished())
  //   return hipErrorNotReady;

  uint64_t Started = start->getFinishTime();
  uint64_t Finished = stop->getFinishTime();

  uint64_t timerResolution    = this->GetDevice()->GetDeviceProps()->timerResolution;
  uint32_t timestampValidBits = this->GetDevice()->GetDeviceProps()->timestampValidBits;

  logDebug("EventElapsedTime: Started {} Finished {} timerResolution {} timestampValidBits {}\n", Started, Finished, timerResolution, timestampValidBits);

  Started = (Started & (((uint64_t)1 << timestampValidBits) - 1));
  Finished = (Finished & (((uint64_t)1 << timestampValidBits) - 1));
  if (Started > Finished)
    Finished = Finished + ((uint64_t)1 << timestampValidBits) - Started;
  Started *= timerResolution;
  Finished *= timerResolution;

  logDebug("EventElapsedTime: STARTED {} / {} FINISHED {} / {} \n",
           (void *)start, Started, (void *)stop, Finished);

  // apparently fails for Intel NEO, god knows why 
  // assert(Finished >= Started);   
  uint64_t Elapsed;
  if (Finished < Started) {
    HIP_PROCESS_ERROR_MSG("HipLZ Invalid timmestamp values ", hipErrorInitializationError);
  } else
    Elapsed = Finished - Started;
  uint64_t MS = (Elapsed / NANOSECS)*1000;
  uint64_t NS = Elapsed % NANOSECS;
  float FractInMS = ((float)NS) / 1000000.0f;
  *ms = (float)MS + FractInMS;
  
  return hipSuccess;
}

// The synchronious among all HipLZ queues
bool LZContext::finishAll() {
  std::set<hipStream_t> Copies;
  {
    std::lock_guard<std::mutex> Lock(ContextMutex);
    for (hipStream_t I : Queues) {
      Copies.insert(I);
    }
    Copies.insert(DefaultQueue);   
  }

  for (hipStream_t I : Copies) {
    bool err = I->finish();
    if (!err) {
      logError("HipLZ Finish() failed with error {}\n", err);
      return false;
    }
  }
  return true;
}

// Reset current context 
void LZContext::reset() {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  int err;

  // Cleanup the execution item stack
  while (!this->ExecStack.empty()) {
    ExecItem *Item = ExecStack.top();
    delete Item;
    this->ExecStack.pop();
  }

  this->Queues.clear();
  delete DefaultQueue;
  // this->Memory.clear();

  // TODO: check if the default queue still need to be initialized?
}

// Initialize HipLZ drivers 
bool LZDriver::InitDrivers(std::vector<LZDriver* >& drivers, const ze_device_type_t deviceType) {
  const ze_device_type_t type = ZE_DEVICE_TYPE_GPU;
  ze_device_handle_t pDevice = nullptr;

  // Get driver count 
  uint32_t driverCount = 0;
  ze_result_t status = zeDriverGet(&driverCount, nullptr);
  LZ_PROCESS_ERROR(status);
  logDebug("HipLZ GET DRIVER via calling zeDriverGet {}\n", status);

  // Get drivers 
  std::vector<ze_driver_handle_t> driver_handles(driverCount);
  status = zeDriverGet(&driverCount, driver_handles.data());
  LZ_PROCESS_ERROR(status);
  logDebug("HipLZ GET DRIVER COUNT via calling zeDriverGet {}\n", status);

  // Create driver object and find the level-0 devices for each driver 
  for (uint32_t driverId = 0; driverId < driverCount; ++ driverId) {
    ze_driver_handle_t hDriver = driver_handles[driverId];
    LZDriver* driver = new LZDriver(hDriver, deviceType);
    drivers.push_back(driver);
    
    // Count the number of devices
    NumLZDevices += driver->GetNumOfDevices();
 }

  logDebug("LZ DEVICES {}", NumLZDevices);

  if (NumLZDevices == 0) {
    HIP_PROCESS_ERROR(hipErrorNoDevice);
  }

  // Set the number of drivers
  NumLZDrivers = driverCount;

  return true;
}

// Collect HipLZ device that belongs to this driver
bool LZDriver::FindHipLZDevices() {
  // get all devices 
  uint32_t deviceCount = 0;
  zeDeviceGet(this->hDriver, &deviceCount, nullptr);
  logDebug("GET DRIVER'S DEVICE COUNT {} ", deviceCount);

  std::vector<ze_device_handle_t> device_handles(deviceCount); 
  ze_result_t status = zeDeviceGet(this->hDriver, &deviceCount, device_handles.data());
  LZ_PROCESS_ERROR_MSG("HipLZ zeDeviceGet FAILED with return code ", status);
  logDebug("GET DRIVER'S DEVICE COUNT (via calling zeDeviceGet) -  {} ", deviceCount);

  ze_device_handle_t found = nullptr;
  // For each device, find the first one matching the type 
  for (uint32_t deviceId = 0; deviceId < deviceCount; ++ deviceId) {
    auto hDevice = device_handles[deviceId];
    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    status = zeDeviceGetProperties(hDevice, &device_properties);
    LZ_PROCESS_ERROR_MSG("HipLZ zeDeviceGetProperties FAILED with return code " ,status);
    logDebug("GET DEVICE PROPERTY (via calling zeDeviceGetProperties) {} ", this->deviceType == device_properties.type);

    if (this->deviceType == device_properties.type) {
      // Register HipLZ device in driver
      this->devices.push_back(new LZDevice(deviceId, hDevice, this));
    }
  }

  return !this->devices.empty();
}

// Get HipLZ driver via integer ID 
LZDriver& LZDriver::HipLZDriverById(int id) {
  return * HipLZDrivers.at(id);
}

// Get the HipLZ device driver by ID                                                                    
LZDevice& LZDriver::GetDeviceById(int id) {
  if (id >= this->devices.size() || this->devices.size() == 0 || id < 0)
    HIP_PROCESS_ERROR_MSG("Hiplz devices were not initialized in this driver?", hipErrorInitializationError);

  return * this->devices[id];
}

// Execute the callback function  
static void * CallbackExecutor(void* data) {
  hipStreamCallbackData* callback_data = (hipStreamCallbackData*)data;

  if (callback_data != nullptr) {
    // Invoke the callback function   
    callback_data->Callback(callback_data->Stream, hipSuccess, callback_data->UserData);

    // Destroy callback data   
    delete callback_data;
  }

  return 0;
}

// Monitor event status and invoke callbacks  
static void * EventMonitor(void* data) {
  LZQueue* lzQueue = (LZQueue* )data;
  while (lzQueue->GetContext() != nullptr) {
    // Invoke callbacks 
    hipStreamCallbackData callback_data;
    while (lzQueue->GetCallback(&callback_data)) {
      ze_result_t status;
      status = zeEventHostSynchronize(callback_data.waitEvent, UINT64_MAX );
      //LZ_PROCESS_ERROR_MSG("HipLZ zeEventHostSynchronize FAILED with return code ", status);
      callback_data.Callback(callback_data.Stream, lzConvertResult(status), callback_data.UserData);
      status = zeEventHostSignal(callback_data.signalEvent);
      LZ_PROCESS_ERROR_MSG("HipLZ zeEventHostSignal FAILED with return code ", status);
      status = zeEventHostSynchronize(callback_data.waitEvent2, UINT64_MAX );
      LZ_PROCESS_ERROR_MSG("HipLZ zeEventHostSynchronize FAILED with return code ", status);
      status = zeEventDestroy(callback_data.waitEvent);
      LZ_PROCESS_ERROR_MSG("HipLZ zeEventDestroy FAILED with return code ", status);
      status = zeEventDestroy(callback_data.signalEvent);
      LZ_PROCESS_ERROR_MSG("HipLZ zeEventDestroy FAILED with return code ", status);
      status = zeEventDestroy(callback_data.waitEvent2);
      LZ_PROCESS_ERROR_MSG("HipLZ zeEventDestroy FAILED with return code ", status);
      status = zeEventPoolDestroy(callback_data.eventPool);
      LZ_PROCESS_ERROR_MSG("HipLZ zeEventPoolDestroy FAILED with return code ", status);
    }
    // Release processor 
    pthread_yield();
  }

  return 0;
}

LZQueue::LZQueue(LZContext* lzContext_, bool needDefaultCmdList) {
  // Initialize super class fields, i.e. ClQueue
  this->LastEvent = nullptr;
  this->Flags = 0;
  this->Priority = 0;
  
  // Initialize Level-0 related class fields
  this->lzContext = lzContext_;
  this->defaultCmdList = nullptr;
  this->monitorThreadId = 0;

  // Initialize Level-0 queue
  initializeQueue(lzContext, needDefaultCmdList);
}

// Initialize Level-0 queue
void LZQueue::initializeQueue(LZContext* lzContext, bool needDefaultCmdList) {
  // Create a Level-0 command queue
  ze_command_queue_desc_t cqDesc;
  cqDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  cqDesc.pNext = nullptr;
  cqDesc.ordinal = 0;
  cqDesc.index = 0;
  cqDesc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY; // 0; // default hehaviour 
  cqDesc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
  cqDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

  // Create the Level-0 queue  
  ze_result_t status = zeCommandQueueCreate(lzContext->GetContextHandle(),
					    lzContext->GetDevice()->GetDeviceHandle(), &cqDesc, &hQueue);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandQueueCreate FAILED with return code ", status);
  logDebug("LZ QUEUE INITIALIZATION via calling zeCommandQueueCreate {} ", status);

  if (needDefaultCmdList) {
    this->defaultCmdList = new LZCommandList(this->lzContext);
  }
}

LZQueue::LZQueue(LZContext* lzContext_, LZCommandList* lzCmdList) {
   // Initialize super class fields, i.e. ClQueue
  this->LastEvent = nullptr;
  this->Flags = 0;
  this->Priority = 0;

  // Initialize Level-0 related class fields
  this->lzContext = lzContext_;
  this->defaultCmdList = lzCmdList;
  this->monitorThreadId = 0;

  // Initialize Level-0 queue
  initializeQueue(lzContext);
}

// Queue synchronous support                                                                           
bool LZQueue::finish() {
  std::lock_guard<std::mutex> Lock(QueueMutex);
  if (this->lzContext == nullptr) {
    HIP_PROCESS_ERROR_MSG("HipLZ LZQueue was not associated with a LZContext!", hipErrorInitializationError);
  }
  return defaultCmdList->finish();
  //ze_result_t status = zeCommandQueueSynchronize(hQueue, UINT64_MAX);
  //LZ_PROCESS_ERROR_MSG("HipLZ zeCommandQueueSynchronize FAILED with return code ", status);
  //logDebug("LZ COMMAND EXECUTION FINISH via calling zeCommandQueueSynchronize {} ", status);
  
  //return true;
}

// Get OpenCL command queue  
cl::CommandQueue& LZQueue::getQueue() {
  HIP_PROCESS_ERROR_MSG("Not support LZQueue::getQueue!", hipErrorNotSupported);
}

// Enqueue barrier for event 
bool LZQueue::enqueueBarrierForEvent(hipEvent_t event) {
  HIP_PROCESS_ERROR_MSG("Not support LZQueue::enqueueBarrierForEvent yet!", hipErrorNotSupported);
}

// Add call back     
bool LZQueue::addCallback(hipStreamCallback_t callback, void *userData) {
  std::lock_guard<std::mutex> Lock(QueueMutex);

  hipStreamCallbackData Data; //  = new hipStreamCallbackData{};
  ze_result_t status;
  ze_event_pool_desc_t ep_desc = {};
  ep_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  ep_desc.count = 3;
  ep_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  ze_event_desc_t ev_desc = {};
  ev_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  ev_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  ev_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
/*  if (LastEvent == nullptr) {
    callback(this, hipSuccess, userData);
    return true;
  }*/

  ze_device_handle_t dev = GetContext()->GetDevice()->GetDeviceHandle();
  status = zeEventPoolCreate(GetContext()->GetContextHandle(), &ep_desc, 1, &dev, &(Data.eventPool));
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventPoolCreate FAILED with return code ", status);
  status = zeEventCreate(Data.eventPool, &ev_desc, &(Data.waitEvent));
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
  status = zeEventCreate(Data.eventPool, &ev_desc, &(Data.signalEvent));
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
  status = zeEventCreate(Data.eventPool, &ev_desc, &(Data.waitEvent2));
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
  Data.Stream = this;
  Data.Callback = callback;
  Data.UserData = userData;
  Data.Status = hipSuccess;
  ze_command_list_handle_t list = defaultCmdList->GetCommandListHandle();
  status = zeCommandListAppendSignalEvent(list, Data.waitEvent);
  LZ_PROCESS_ERROR_MSG("HipLZ  zeCommandListAppendSignalEvent FAILED with return code ", status);
  status = zeCommandListAppendBarrier(list, NULL, 1, &(Data.signalEvent));
  LZ_PROCESS_ERROR_MSG("HipLZ  zeCommandListAppendBarrier FAILED with return code ", status);
  status = zeCommandListAppendSignalEvent(list, Data.waitEvent2);
  LZ_PROCESS_ERROR_MSG("HipLZ  zeCommandListAppendSignalEvent FAILED with return code ", status);
  // err = ::clSetEventCallback(LastEvent, CL_COMPLETE, notifyOpenCLevent, Data);
  // if (err != CL_SUCCESS)
  //   logError("clSetEventCallback failed with error {}\n", err);

  // return (err == CL_SUCCESS);
  
  // Add event callback to callback list 
  {
    std::lock_guard<std::mutex> Lock(CallbacksMutex);
    callbacks.push_back(Data);
  }
  
  // Create event monitor on demand  
  CheckAndCreateMonitor();

  return true;

  // LZ_PROCESS_ERROR_MSG("Not support LZQueue::addCallback yet!", hipErrorNotSupported);
}

// Get callback from lock protected callback list  
bool LZQueue::GetCallback(hipStreamCallbackData* data) {
  bool res = false;
  {
    std::lock_guard<std::mutex> Lock(CallbacksMutex);
    if (!this->callbacks.empty()) {
      * data = callbacks.front();
      callbacks.pop_front();

      res = true;
    }
  }

  return res;
}

// Create the callback monitor on-demand  
bool LZQueue::CheckAndCreateMonitor() {
  // This operation is protected by event mutex  
  std::lock_guard<std::mutex> Lock(EventsMutex);
  if (this->monitorThreadId != 0)
    return false;

  // If the event monotor thread was not created yet, spawn it 
  return 0 != pthread_create(&(this->monitorThreadId), 0, EventMonitor, (void* )this);
}

// Synchronize on the event monitor thread 
void LZQueue::WaitEventMonitor() {
  if  (this->monitorThreadId == 0)
    return;

  // Join the event monitor thread   
  pthread_join(this->monitorThreadId, NULL);
}

// Record event
bool LZQueue::recordEvent(hipEvent_t event) {
  // std::lock_guard<std::mutex> Lock(QueueMutex);

  /* slightly tricky WRT refcounts.
   * if LastEvents != NULL, it should have refcount 1.
   *  if NULL, enqueue a marker here; 
   * libOpenCL will process it & decrease refc to 1;
   * we retain it here because d-tor is called at } and releases it.  
   * 
   * in both cases, event->recordStream should Retain */

  // if (LastEvent == nullptr) {
  //   cl::Event MarkerEvent;
  //   int err = Queue.enqueueMarkerWithWaitList(nullptr, &MarkerEvent);
  //   if (err) {
  //     logError ("enqueueMarkerWithWaitList FAILED with {}\n", err);
  //     return false;
  //   } else {
  //     LastEvent = MarkerEvent();
  //     clRetainEvent(LastEvent);
  //   }
  // }

  // logDebug("record Event: {} on Queue: {}\n", (void *)(LastEvent),
  //          (void *)(Queue()));

  // cl_uint refc1, refc2;
  // int err =
  //     ::clGetEventInfo(LastEvent, CL_EVENT_REFERENCE_COUNT, 4, &refc1, NULL);
  // assert(err == CL_SUCCESS);
  // can be >1 because recordEvent can be called >1 on the same event
  // assert(refc1 >= 1);

  // return event->recordStream(this, LastEvent);

  // err = ::clGetEventInfo(LastEvent, CL_EVENT_REFERENCE_COUNT, 4, &refc2, NULL);
  // assert(err == CL_SUCCESS);
  // assert(refc2 >= 2);
  // assert(refc2 == (refc1 + 1));

  if (event == nullptr)
    HIP_PROCESS_ERROR_MSG("HipLZ get null Event recorded?", hipErrorInitializationError);

  if (this->defaultCmdList == nullptr) 
    HIP_PROCESS_ERROR_MSG("Invalid command list during event recording", hipErrorInitializationError);
  
  // Record event to stream
  event->recordStream(this, nullptr);
  
  // Record timestamp got from execution write global timestamp from command list
  event->recordTimeStamp(this->defaultCmdList->ExecuteWriteGlobalTimeStamp(this));

  return true;
}

// Memory copy support   
hipError_t LZQueue::memCopy(void *dst, const void *src, size_t size) {
  if (this->defaultCmdList == nullptr)
    HIP_PROCESS_ERROR_MSG("HipLZ Invalid command list ", hipErrorInitializationError);
  this->defaultCmdList->ExecuteMemCopy(this, dst, src, size); 
  return hipSuccess;
}

// Memory fill support  
hipError_t LZQueue::memFill(void *dst, size_t size, const void *pattern, size_t pattern_size) {
  if (this->defaultCmdList == nullptr)
    HIP_PROCESS_ERROR_MSG("HipLZ Invalid command list ", hipErrorInitializationError);
  this->defaultCmdList->ExecuteMemFill(this, dst, size, pattern, pattern_size);
  return hipSuccess;
}

// Launch kernel support 
hipError_t LZQueue::launch3(ClKernel *Kernel, dim3 grid, dim3 block) {
  HIP_PROCESS_ERROR_MSG("Not support LZQueue::launch3 yet!", hipErrorNotSupported);
}

// Launch kernel support 
hipError_t LZQueue::launch(ClKernel *Kernel, ExecItem *Arguments) {
  if (this->defaultCmdList == nullptr) {
    HIP_PROCESS_ERROR_MSG("Invalid command list", hipErrorInitializationError);
  } else {
    if (Kernel->SupportLZ() && Arguments->SupportLZ()) {
      this->defaultCmdList->ExecuteKernel(this, (LZKernel* )Kernel, (LZExecItem* )Arguments);
    } else
      HIP_PROCESS_ERROR_MSG("Not support LZQueue::launch yet!", hipErrorNotSupported); 
  }

  return hipSuccess;
}

// The asynchronously memory copy support 
bool LZQueue::memCopyAsync(void *dst, const void *src, size_t sizeBytes) {
  if (this->defaultCmdList == nullptr)
    HIP_PROCESS_ERROR_MSG("No default command list setup in current HipLZ queue yet!", hipErrorInitializationError);
  
  return this->defaultCmdList->ExecuteMemCopyAsync(this, dst, src, sizeBytes);
}

// The asynchronously memory fill support
bool LZQueue::memFillAsync(void *dst, size_t size, const void *pattern, size_t pattern_size) {
  if (this->defaultCmdList == nullptr)
    HIP_PROCESS_ERROR_MSG("No default command list setup in current HipLZ queue yet!", hipErrorInitializationError);

  return this->defaultCmdList->ExecuteMemFillAsync(this, dst, size, pattern, pattern_size);
}

// The set the current event  
bool LZQueue::SetEvent(LZEvent* event) {
  // if (this->currentEvent != nullptr) 
  HIP_PROCESS_ERROR_MSG("No current event here!", hipErrorInitializationError);

  // this->currentEvent = event;
  
  return true;
}

// Get and clear current event 
LZEvent* LZQueue::GetAndClearEvent() {
  // LZEvent* res = this->currentEvent;
  // this->currentEvent = nullptr;

  // return res;

  HIP_PROCESS_ERROR_MSG("No current event there to get and clear!", hipErrorInitializationError);

  return nullptr;
}

// Create and monior event  
LZEvent* LZQueue::CreateAndMonitorEvent(LZEvent* event) {
  if (!event)
    event = this->lzContext->createEvent(0);
  {
    std::lock_guard<std::mutex> Lock(EventsMutex);
    // Put event into local event list to enable monitor    
    this->localEvents.push_back(event);
  }

  return event;
}

LZEvent* LZQueue::GetPendingEvent() {
  LZEvent* res = nullptr;
  {
    // Check if there is pending event in list
    std::lock_guard<std::mutex> Lock(EventsMutex);
    if (!this->localEvents.empty()) {
      res = this->localEvents.front();
      this->localEvents.pop_front();
    }
  }

  return res;
}
    
// Get the potential signal event 
LZEvent* LZCommandList::GetSignalEvent(LZQueue* lzQueue) {
  // LZEvent* lzEvent = lzQueue->GetAndClearEvent();
  // if (lzEvent != nullptr) {
  //   lzEvent->GetEventHandler();
  // }

  return lzQueue->CreateAndMonitorEvent(nullptr);
}

// Execute the Level-0 kernel
bool LZCommandList::ExecuteKernel(LZQueue* lzQueue, LZKernel* Kernel, LZExecItem* Arguments) {
  // Set group size
  ze_result_t status = zeKernelSetGroupSize(Kernel->GetKernelHandle(),
					    Arguments->BlockDim.x, Arguments->BlockDim.y, Arguments->BlockDim.z);
  LZ_PROCESS_ERROR_MSG("could not set group size! ", status);

  logDebug("LZ KERNEL EXECUTION via calling zeKernelSetGroupSize {} ", status);
  
  // Set all kernel function arguments
  Arguments->setupAllArgs(Kernel);
  
  // Launch kernel via Level-0 command list
  uint32_t numGroupsX = Arguments->GridDim.x;
  uint32_t numGroupsY = Arguments->GridDim.y;
  uint32_t numGroupsz = Arguments->GridDim.z;
  ze_group_count_t hLaunchFuncArgs = { numGroupsX, numGroupsY, numGroupsz };
  
  // Get the potential signal event 
  ze_event_handle_t hSignalEvent = GetSignalEvent(lzQueue)->GetEventHandler();

  status = zeCommandListAppendLaunchKernel(hCommandList,
                                           Kernel->GetKernelHandle(),
                                           &hLaunchFuncArgs,
                                           hSignalEvent,
                                           0,
                                           nullptr);
  LZ_PROCESS_ERROR_MSG("Hiplz zeCommandListAppendLaunchKernel FAILED with return code  ", status);

  logDebug("LZ KERNEL EXECUTION via calling zeCommandListAppendLaunchKernel {} ", status);

  // Execute kernel  
  return Execute(lzQueue);
}

// Execute HipLZ memory copy command
bool LZCommandList::ExecuteMemCopy(LZQueue* lzQueue, void *dst, const void *src, size_t sizeBytes) {
  // Get the potential signal event
  ze_event_handle_t hSignalEvent = GetSignalEvent(lzQueue)->GetEventHandler();

  ze_result_t status = zeCommandListAppendMemoryCopy(hCommandList, dst, src, sizeBytes,
                                                     hSignalEvent, 0, NULL);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendMemoryCopy FAILED with return code ", status);
  // Execute memory copy  
  return Execute(lzQueue);
}

// Execute HipLZ memory copy command asynchronously
bool LZCommandList::ExecuteMemCopyAsync(LZQueue* lzQueue, void *dst, const void *src, size_t sizeBytes) {
  // Get the potential signal event  
  ze_event_handle_t hSignalEvent = GetSignalEvent(lzQueue)->GetEventHandler();

  ze_result_t status = zeCommandListAppendMemoryCopy(hCommandList, dst, src, sizeBytes,
                                                     hSignalEvent, 0, NULL);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendMemoryCopy FAILED with return code ", status);
  // Execute memory copy asynchronously
  return ExecuteAsync(lzQueue);
}

// Execute HipLZ memory fill command
bool LZCommandList::ExecuteMemFill(LZQueue* lzQueue, void *dst, size_t size, const void *pattern, size_t pattern_size) {
  ze_event_handle_t hSignalEvent = GetSignalEvent(lzQueue)->GetEventHandler();

  ze_result_t status = zeCommandListAppendMemoryFill(hCommandList, dst, pattern, pattern_size, size, hSignalEvent, 0, NULL);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendMemoryFill FAILED with return code ", status);
  return Execute(lzQueue);
}

// Execute HipLZ memory fill command asynchronously
bool LZCommandList::ExecuteMemFillAsync(LZQueue* lzQueue, void *dst, size_t size, const void *pattern, size_t pattern_size) {
  ze_event_handle_t hSignalEvent = GetSignalEvent(lzQueue)->GetEventHandler();

  ze_result_t status = zeCommandListAppendMemoryFill(hCommandList, dst, pattern, pattern_size, size, hSignalEvent, 0, NULL);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendMemoryFill FAILED with return code ", status);
  return ExecuteAsync(lzQueue);
}

// Execute HipLZ write global timestamp
uint64_t LZCommandList::ExecuteWriteGlobalTimeStamp(LZQueue* lzQueue) {
  // Get the event for recording time stamp
  // LZEvent* tsEvent = GetTimeStampEvent(lzQueue);
  // if (!tsEvent) {
  //   logError("LZ Time Elapse Event was not set ");

  //   return false;
  // }
		 
  ze_result_t status = zeCommandListAppendWriteGlobalTimestamp(hCommandList, (uint64_t*)(shared_buf), nullptr, 0, nullptr);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendWriteGlobalTimestamp FAILED with return code ", status);
  Execute(lzQueue);

  uint64_t ret = * (uint64_t*)(shared_buf);

  return ret;
}

bool LZCommandList::finish() {
  ze_result_t status = zeCommandListAppendSignalEvent(hCommandList, finishEvent);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendSignalEvent FAILED with return code ", status);
  status = zeEventHostSynchronize(finishEvent, UINT64_MAX);
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventHostSynchronize FAILED with return code ", status);
  status = zeEventHostReset(finishEvent);
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventHostReset FAILED with return code ", status);
  return true;
 }

// Execute HipLZ command list  
bool LZCommandList::Execute(LZQueue* lzQueue) {
  // Finished appending commands (typically done on another thread)
//  ze_result_t status = zeCommandListClose(hCommandList);
//  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListClose FAILED with return code ", status);

//  logDebug("LZ KERNEL EXECUTION via calling zeCommandListClose {} ", status);
  
  // Execute command list in command queue
//  status = zeCommandQueueExecuteCommandLists(lzQueue->GetQueueHandle(), 1, &hCommandList, nullptr);
//  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandQueueExecuteCommandLists FAILED with return code ", status);

//  logDebug("LZ KERNEL EXECUTION via calling zeCommandQueueExecuteCommandLists {} ", status);

  // Synchronize host with device kernel execution
  return finish();
 
//  logDebug("LZ KERNEL EXECUTION via calling zeCommandQueueSynchronize {} ", status);
  
  // Reset (recycle) command list for new commands
//  status = zeCommandListReset(hCommandList);
//  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListReset FAILED with return code ", status);

//  logDebug("LZ KERNEL EXECUTION via calling zeCommandListReset {} ", status);
  
//  return true;
}

// Execute HipLZ command list asynchronously
bool LZCommandList::ExecuteAsync(LZQueue* lzQueue) {
  // Finished appending commands (typically done on another thread)  
//  ze_result_t status = zeCommandListClose(hCommandList);
//  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListClose FAILED with return code ", status);

//  logDebug("LZ KERNEL EXECUTION via calling zeCommandListClose {} ", status);

  // Execute command list in command queue  
//  status = zeCommandQueueExecuteCommandLists(lzQueue->GetQueueHandle(), 1, &hCommandList, nullptr);
//  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandQueueExecuteCommandLists FAILED with return code ", status);

//  logDebug("LZ KERNEL EXECUTION via calling zeCommandQueueExecuteCommandLists {} ", status);

//  status = zeCommandListReset(hCommandList);
//  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListReset FAILED with return code ", status);
  
  return true;
}

LZCommandList::LZCommandList(LZContext* lzContext_, bool immediate) {
  this->lzContext = lzContext_;

  // Initialize the shared memory buffer
  this->shared_buf = this->lzContext->allocate(32, 8, LZMemoryType::Shared);
  // Initialize the uint64_t part as 0 
  * (uint64_t* )this->shared_buf = 0;

  if (immediate) {
    // Create command list via immidiately associated with a queue
    ze_command_queue_desc_t cqDesc;
    cqDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    cqDesc.pNext = nullptr;
    cqDesc.ordinal = 0;
    cqDesc.index = 0;
    cqDesc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY; // 0;
    cqDesc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    cqDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL; 

    ze_result_t status = zeCommandListCreateImmediate(this->lzContext->GetContextHandle(),
						      this->lzContext->GetDevice()->GetDeviceHandle(),
						      &cqDesc, &hCommandList);
    LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListCreate FAILED with return code ", status);
    logDebug("LZ COMMAND LIST CREATION via calling zeCommandListCreateImmediate {} ", status);
  } else {
    // Default command list creation, i.e. w/o immediately associated with a  queue
    
    // Create the command list                                                
    ze_command_list_desc_t clDesc;
    clDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    clDesc.flags = ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY; // default hehaviour 
    clDesc.commandQueueGroupOrdinal = 0;
    clDesc.pNext = nullptr;
    ze_result_t status = zeCommandListCreate(lzContext->GetContextHandle(), lzContext->GetDevice()->GetDeviceHandle(),
					     &clDesc, &hCommandList);
    
    LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListCreate FAILED with return code ", status);
    logDebug("LZ COMMAND LIST CREATION via calling zeCommandListCreate {} ", status);
  }
  ze_event_pool_desc_t ep_desc = {};
  ep_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  ep_desc.count = 1;
  ep_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  ze_event_desc_t ev_desc = {};
  ev_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  ev_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  ev_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  ze_result_t status;
  ze_device_handle_t dev = lzContext_->GetDevice()->GetDeviceHandle();
  status = zeEventPoolCreate(lzContext_->GetContextHandle(), &ep_desc, 1, &dev, &(this->eventPool));
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventPoolCreate FAILED with return code ", status);
  status = zeEventCreate(this->eventPool, &ev_desc, &(this->finishEvent));
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
}

int LZExecItem::setupAllArgs(LZKernel *kernel) {
  OCLFuncInfo *FuncInfo = kernel->getFuncInfo();
  size_t NumLocals = 0;
 
  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    if (FuncInfo->ArgTypeInfo[i].space == OCLSpace::Local)
      ++ NumLocals;
  }
  // there can only be one dynamic shared mem variable, per cuda spec 
  assert(NumLocals <= 1);
  
  if ((OffsetsSizes.size()+NumLocals) != FuncInfo->ArgTypeInfo.size()) {
    logError("Some arguments are still unset\n");
    return CL_INVALID_VALUE;
  }

  if (OffsetsSizes.size() == 0)
    return CL_SUCCESS;

  std::sort(OffsetsSizes.begin(), OffsetsSizes.end());
  if ((std::get<0>(OffsetsSizes[0]) != 0) ||
      (std::get<1>(OffsetsSizes[0]) == 0)) {
    logError("Invalid offset/size\n");
    return CL_INVALID_VALUE;
  }
  
  // check args are set  
  if (OffsetsSizes.size() > 1) {
    for (size_t i = 1; i < OffsetsSizes.size(); ++i) {
      if ( (std::get<0>(OffsetsSizes[i]) == 0) ||
           (std::get<1>(OffsetsSizes[i]) == 0) ||
           (
	    (std::get<0>(OffsetsSizes[i - 1]) + std::get<1>(OffsetsSizes[i - 1])) >
            std::get<0>(OffsetsSizes[i]))
           ) {
	logError("Invalid offset/size\n");
	return CL_INVALID_VALUE;
      }
    }
  }

  const unsigned char *start = ArgData.data();
  void *p;
  int err;
  for (cl_uint i = 0; i < OffsetsSizes.size(); ++i) {
    OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];
    logDebug("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n", i,
             std::get<0>(OffsetsSizes[i]), std::get<1>(OffsetsSizes[i]),
             (unsigned)ai.type, (unsigned)ai.space, ai.size);

    if (ai.type == OCLType::Pointer) {
      // TODO: sync with ExecItem's solution   
      assert(ai.size == sizeof(void *));
      assert(std::get<1>(OffsetsSizes[i]) == ai.size);
      size_t size = std::get<1>(OffsetsSizes[i]);
      size_t offs = std::get<0>(OffsetsSizes[i]);
      const void* value = (void*)(start + offs);
      logDebug("setArg SVM {} to {}\n", i, p);
      ze_result_t status = zeKernelSetArgumentValue(kernel->GetKernelHandle(), i, size, value);

      if (status != ZE_RESULT_SUCCESS) {
        logDebug("zeKernelSetArgumentValue failed with error {}\n", err);
        return CL_INVALID_VALUE;
      }

      logDebug("LZ SET ARGUMENT VALUE via calling zeKernelSetArgumentValue {} ", status);
    } else {
      size_t size = std::get<1>(OffsetsSizes[i]);
      size_t offs = std::get<0>(OffsetsSizes[i]);
      const void* value = (void*)(start + offs);
      logDebug("setArg {} size {} offs {}\n", i, size, offs);
      ze_result_t status = zeKernelSetArgumentValue(kernel->GetKernelHandle(), i, size, value);

      if (status != ZE_RESULT_SUCCESS) {
        logDebug("zeKernelSetArgumentValue failed with error {}\n", err);
        return CL_INVALID_VALUE;
      }

      logDebug("LZ SET ARGUMENT VALUE via calling zeKernelSetArgumentValue {} ", status);
    }
  }

  return CL_SUCCESS;
}

bool LZExecItem::launch(LZKernel *Kernel) {
  return Stream->launch(Kernel, this) == hipSuccess;  
};

// Create Level-0 kernel 
void LZModule::CreateKernel(std::string funcName, OpenCLFunctionInfoMap& FuncInfos) {
  if (this->kernels.find(funcName) != this->kernels.end())
    return;

  // Create kernel
  ze_kernel_desc_t kernelDesc = {
    ZE_STRUCTURE_TYPE_KERNEL_DESC,
    nullptr,
    0, // flags 
    funcName.c_str()
  };
  ze_kernel_handle_t hKernel;
  ze_result_t status = zeKernelCreate(this->hModule, &kernelDesc, &hKernel);
  LZ_PROCESS_ERROR_MSG("HipLZ zeKernelCreate FAILED with return code ", status);

  logDebug("LZ KERNEL CREATION via calling zeKernelCreate {} ", status);
  
  // Register kernel
  if (FuncInfos.find(funcName) == FuncInfos.end())
    HIP_PROCESS_ERROR_MSG("HipLZ could not find function information ", hipErrorInitializationError);
  this->kernels[funcName] = new LZKernel(hKernel, FuncInfos[funcName]);
}

// Get Level-0 kernel
LZKernel* LZModule::GetKernel(std::string funcName) {
  if (kernels.find(funcName) == kernels.end())
    return nullptr;

  return kernels[funcName];
}

// Create event pool
LZEventPool::LZEventPool(LZContext* c) {
  this->lzContext = c;

  // Create event pool
  ze_event_pool_desc_t eventPoolDesc = {
    ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
    nullptr,
    ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP, // all events in pool are visible to Host
    1 // count
  };
  
  ze_result_t status = zeEventPoolCreate(this->lzContext->GetContextHandle(), &eventPoolDesc, 0, nullptr, &hEventPool);
  LZ_PROCESS_ERROR_MSG("HipLZ event pool creation fail! ", status);
}

// Create HipLZ event from event pool
LZEvent* LZEventPool::createEvent(unsigned flags) {
  return new LZEvent(this->lzContext, flags, this);
}

// Create HipLZ event
LZEvent::LZEvent(LZContext* c, unsigned flags, LZEventPool* eventPool) {
  this->Stream = nullptr;
  this->Status = EVENT_STATUS_INIT;
  this->Flags = flags;
  this->cont = c;
  * (uint64_t* )this->timestamp_buf = 0;

  ze_event_desc_t eventDesc = {
    ZE_STRUCTURE_TYPE_EVENT_DESC,
    nullptr,
    0, // index
    0, // no additional memory/cache coherency required on signal
    ZE_EVENT_SCOPE_FLAG_HOST  // ensure memory coherency across device and Host after event completes
  };
 
  ze_result_t status = zeEventCreate(eventPool->GetEventPoolHandler(), &eventDesc, &hEvent);
  LZ_PROCESS_ERROR_MSG("HipLZ event creation fail! ", status);
}

// Get the finish time of the event associated operation
uint64_t LZEvent::getFinishTime() {
  std::lock_guard<std::mutex> Lock(EventMutex);
  
  // ze_kernel_timestamp_result_t dst;
  // ze_result_t status = zeEventQueryKernelTimestamp(hEvent, &dst);
  // if (status != ZE_RESULT_SUCCESS)
  //   throw InvalidLevel0Initialization("HipLZ event queries timestamp error!");

  return getTimeStamp();
}

// Record event into stream
bool LZEvent::recordStream(hipStream_t S, cl_event E) {
  std::lock_guard<std::mutex> Lock(EventMutex);
 
  Stream = S;
  // if (((LZQueue* )Stream)->SetEvent(this)) {
  ((LZQueue* )Stream)->CreateAndMonitorEvent(this);
  Status = EVENT_STATUS_RECORDING;

  return true;
}

bool LZEvent::updateFinishStatus() {
  std::lock_guard<std::mutex> Lock(EventMutex);
  if (Status != EVENT_STATUS_RECORDING)
    return false;

  if (!Stream) {
    // Here we use this protocol: Stream == nullptr ==> event is associated with a queue operation 
    ze_result_t status = zeEventHostSynchronize(this->hEvent, UINT64_MAX);
    LZ_PROCESS_ERROR_MSG("HipLZ event synchronization error! ", status);
  }

  Status = EVENT_STATUS_RECORDED;
  return true;
}

bool LZEvent::wait() {
  std::lock_guard<std::mutex> Lock(EventMutex);
  if (Status != EVENT_STATUS_RECORDING)
    return false;

  if (!Stream) {
    // Here we use this protocol: Stream == nullptr ==> event is associated with a queue operation
    ze_result_t status = zeEventHostSynchronize(this->hEvent, UINT64_MAX);
    LZ_PROCESS_ERROR_MSG("HipLZ event synchronization error! ", status);
  }

  Status = EVENT_STATUS_RECORDED;
  return true;
}

// Get the event object? this is only for OpenCL  
cl::Event LZEvent::getEvent() { 
  HIP_PROCESS_ERROR_MSG("HipLZ does not support cl::Event! ", hipErrorNotSupported);
}

// Check if the event is from same cl::Context? this is only for OpenCL 
bool LZEvent::isFromContext(cl::Context &Other) {
  HIP_PROCESS_ERROR_MSG("HipLZ does not support cl::Context! ", hipErrorNotSupported);
}

hipError_t lzConvertResult(ze_result_t status) {
  switch (status) {
  case ZE_RESULT_SUCCESS:
    return hipSuccess;
  case ZE_RESULT_NOT_READY:
    return hipErrorNotReady;
  case ZE_RESULT_ERROR_DEVICE_LOST:
    return hipErrorOperatingSystem;
  case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return hipErrorMemoryAllocation;
  case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    return hipErrorMemoryAllocation;
  case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
    return hipErrorInvalidSource;
  case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
    return hipErrorInvalidImage;
  case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
    return hipErrorOperatingSystem;
  case ZE_RESULT_ERROR_NOT_AVAILABLE:
    return hipErrorAlreadyAcquired;
  case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
    return hipErrorInitializationError;
  case ZE_RESULT_ERROR_UNINITIALIZED:
    return hipErrorInitializationError;
  case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
    return hipErrorInsufficientDriver;
  case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
    return hipErrorNotSupported;
  case ZE_RESULT_ERROR_INVALID_ARGUMENT:
    return hipErrorInvalidValue;
  case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
    return hipErrorInvalidResourceHandle;
  case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
    return hipErrorInvalidResourceHandle;
  case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
    return hipErrorInvalidValue;
  case ZE_RESULT_ERROR_INVALID_SIZE:
    return hipErrorInvalidValue;
  case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
    return hipErrorInvalidValue;
  case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
    return hipErrorInvalidValue;
  case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
    return hipErrorInvalidResourceHandle;
  case ZE_RESULT_ERROR_INVALID_ENUMERATION:
    return hipErrorInvalidValue;
  case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
    return hipErrorNotSupported;
  case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
    return hipErrorNotSupported;
  case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
    return hipErrorInvalidImage;
  case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
    return hipErrorInvalidSymbol;
  case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
    return hipErrorInvalidDeviceFunction;
  case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
    return hipErrorInvalidDeviceFunction;
  case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
    return hipErrorInvalidConfiguration;
  case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
    return hipErrorInvalidConfiguration;
  case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
    return hipErrorInvalidConfiguration;
  case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
    return hipErrorInvalidConfiguration;
  case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
    return hipErrorInvalidValue;
  case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
    return hipErrorInvalidImage;
  case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
    return hipErrorInvalidResourceHandle;
  case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
    return hipErrorInvalidValue;
  case ZE_RESULT_ERROR_UNKNOWN:
    return hipErrorUnknown;
  default:
    return hipErrorUnknown;
  }
}

const char * lzResultToString(ze_result_t status) {
  switch (status) {
  case ZE_RESULT_SUCCESS:
    return "ZE_RESULT_SUCCESS";
  case ZE_RESULT_NOT_READY:
    return "ZE_RESULT_NOT_READY";
  case ZE_RESULT_ERROR_DEVICE_LOST:
    return "ZE_RESULT_ERROR_DEVICE_LOST";
  case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
  case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
  case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
    return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
  case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
    return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
  case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
    return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
  case ZE_RESULT_ERROR_NOT_AVAILABLE:
    return "ZE_RESULT_ERROR_NOT_AVAILABLE";
  case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
    return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
  case ZE_RESULT_ERROR_UNINITIALIZED:
    return "ZE_RESULT_ERROR_UNINITIALIZED";
  case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
    return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
  case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
    return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
  case ZE_RESULT_ERROR_INVALID_ARGUMENT:
    return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
  case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
    return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
  case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
    return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
  case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
    return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
  case ZE_RESULT_ERROR_INVALID_SIZE:
    return "ZE_RESULT_ERROR_INVALID_SIZE";
  case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
    return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
  case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
    return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
  case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
    return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
  case ZE_RESULT_ERROR_INVALID_ENUMERATION:
    return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
  case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
    return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
  case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
    return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
  case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
    return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
  case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
  case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
  case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
    return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
  case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
    return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
  case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
  case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
  case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
  case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
  case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
    return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
  case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
    return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
  case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
    return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
  case ZE_RESULT_ERROR_UNKNOWN:
    return "ZE_RESULT_ERROR_UNKNOWN";
  default:
    return "Unknown Error Code";
  }
}

static void InitializeHipLZCallOnce() {
  // Initialize the driver 
  ze_result_t status = zeInit(0);
  LZ_PROCESS_ERROR(status);
  logDebug("INITIALIZE LEVEL-0 (via calling zeInit) {}\n", status);
  
  // Initialize HipLZ device drivers and relevant devices
  LZDriver::InitDrivers(HipLZDrivers, ZE_DEVICE_TYPE_GPU);
}

void InitializeHipLZ() {
  static std::once_flag HipLZInitialized;
  std::call_once(HipLZInitialized, InitializeHipLZCallOnce);
}

LZDevice &HipLZDeviceById(int deviceId) {
  return *HipLZDevices.at(deviceId);
}

/***********************************************************************/


ClDevice &CLDeviceById(int deviceId) { return *OpenCLDevices.at(deviceId); }

class InvalidDeviceType : public std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

class InvalidPlatformOrDeviceNumber : public std::out_of_range {
  using std::out_of_range::out_of_range;
};

static void InitializeOpenCLCallOnce() {

  cl_int err = cl::Platform::get(&Platforms);
  std::string ver;
  if (err != CL_SUCCESS)
    return;

  OpenCLDevices.clear();
  NumDevices = 0;
  std::vector<cl::Device> Devices;
  const char *selected_platform_str = std::getenv("HIPCL_PLATFORM");
  const char *selected_device_str = std::getenv("HIPCL_DEVICE");
  const char *selected_device_type_str = std::getenv("HIPCL_DEVICE_TYPE");
  int selected_platform = -1;
  int selected_device = -1;
  cl_bitfield selected_dev_type = 0;
  try {
    if (selected_platform_str) {
      selected_platform = std::stoi(selected_platform_str);
      if ((selected_platform < 0) || (selected_platform >= Platforms.size()))
        throw InvalidPlatformOrDeviceNumber(
            "HIPCL_PLATFORM: platform number out of range");
    }

    if (selected_device_str) {
      selected_device = std::stoi(selected_device_str);
      Devices.clear();
      if (selected_platform < 0)
        selected_platform = 0;
      err =
          Platforms[selected_platform].getDevices(CL_DEVICE_TYPE_ALL, &Devices);
      if (err != CL_SUCCESS)
        throw InvalidPlatformOrDeviceNumber(
            "HIPCL_DEVICE: can't get devices for platform");
      if ((selected_device < 0) || (selected_device >= Devices.size()))
        throw InvalidPlatformOrDeviceNumber(
            "HIPCL_DEVICE: device number out of range");
    }

    if (selected_device_type_str) {
      std::string s(selected_device_type_str);
      if (s == "all")
        selected_dev_type = CL_DEVICE_TYPE_ALL;
      else if (s == "cpu")
        selected_dev_type = CL_DEVICE_TYPE_CPU;
      else if (s == "gpu")
        selected_dev_type = CL_DEVICE_TYPE_GPU;
      else if (s == "default")
        selected_dev_type = CL_DEVICE_TYPE_DEFAULT;
      else if (s == "accel")
        selected_dev_type = CL_DEVICE_TYPE_ACCELERATOR;
      else
        throw InvalidDeviceType(
            "Unknown value provided for HIPCL_DEVICE_TYPE\n");
    }
  } catch (const InvalidDeviceType &e) {
    logCritical("{}\n", e.what());
    return;
  } catch (const InvalidPlatformOrDeviceNumber &e) {
    logCritical("{}\n", e.what());
    return;
  } catch (const std::invalid_argument &e) {
    logCritical(
        "Could not convert HIPCL_PLATFORM or HIPCL_DEVICES to a number\n");
    return;
  } catch (const std::out_of_range &e) {
    logCritical("HIPCL_PLATFORM or HIPCL_DEVICES is out of range\n");
    return;
  }

  if (selected_dev_type == 0)
    selected_dev_type = CL_DEVICE_TYPE_ALL;
  for (auto Platform : Platforms) {
    Devices.clear();
    err = Platform.getDevices(selected_dev_type, &Devices);
    if (err != CL_SUCCESS)
      continue;
    if (Devices.size() == 0)
      continue;
    if (selected_platform >= 0 && (Platforms[selected_platform] != Platform))
      continue;

    for (cl::Device &Dev : Devices) {
      ver.clear();
      if (selected_device >= 0 && (Devices[selected_device] != Dev))
        continue;
      ver = Dev.getInfo<CL_DEVICE_IL_VERSION>(&err);
      if ((err == CL_SUCCESS) && (ver.rfind("SPIR-V_1.", 0) == 0)) {
        ClDevice *temp = new ClDevice(Dev, Platform, NumDevices);
        temp->setPrimaryCtx();
        OpenCLDevices.emplace_back(temp);
        ++NumDevices;
      }
    }
  }

  logDebug("DEVICES {}", NumDevices);
  assert(NumDevices == OpenCLDevices.size());
}

void InitializeOpenCL() {
  static std::once_flag OpenClInitialized;
  std::call_once(OpenClInitialized, InitializeOpenCLCallOnce);
}

static void UnInitializeOpenCLCallOnce() {
  logDebug("DEVICES UNINITALIZE \n");

  for (ClDevice *d : OpenCLDevices) {
    delete d;
  }

  for (auto Platform : Platforms) {
    Platform.unloadCompiler();
  }

  // spdlog::details::os::sleep_for_millis(18000);
}

void UnInitializeOpenCL() {
  static std::once_flag OpenClUnInitialized;
  std::call_once(OpenClUnInitialized, UnInitializeOpenCLCallOnce);
}

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
