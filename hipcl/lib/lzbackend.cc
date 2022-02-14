#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

#include "lzbackend.hh"

#include <cxxabi.h>

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#ifndef NANOSECS
#define NANOSECS 1000000000
#endif

#define FIND_QUEUE_LOCKED(stream)                                              \
  std::lock_guard<std::mutex> Lock(ContextMutex);                              \
  LZQueue *Queue = (LZQueue *)findQueue(stream);                               \
  if (Queue == nullptr)                                                        \
    return hipErrorInvalidResourceHandle

#define CHECK_QUEUE_COMMAND_LIST()                                             \
  if (this->defaultCmdList == nullptr)                                         \
    HIP_PROCESS_ERROR_MSG("HipLZ Invalid command list ", hipErrorInitializationError)

/***********************************************************************/
// HipLZ support
static std::vector<LZDevice *> HipLZDevices INIT_PRIORITY(120);
// The drivers are managed globally
static std::vector<LZDriver *> HipLZDrivers INIT_PRIORITY(120);

// The global storage for module binary
std::vector<std::string *> LZDriver::FatBinModules;

// The global storage for kernel functions, includig module, hostFunction and deviceName
std::vector<std::tuple<std::string *, const void *, const char* >> LZDriver::RegFunctions;

// The global storage for global variables include module, hostVar, deviceName and size
std::vector<std::tuple<std::string *, char *, const char *, int>> LZDriver::GlobalVars;

size_t NumLZDevices = 1;

size_t NumLZDrivers = 1;

static void notifyOpenCLevent(cl_event event, cl_int status, void *data) {
  hipStreamCallbackData *Data = (hipStreamCallbackData *)data;
  Data->Callback(Data->Stream, Data->Status, Data->UserData);
  delete Data;
}

LZDevice::LZDevice(hipDevice_t id, ze_device_handle_t hDevice_, LZDriver* driver_) {
  this->Index = id;
  this->hDevice = hDevice_;
  this->driver = driver_;

  // Retrieve device properties related data
  retrieveDeviceProperties();

  // Create HipLZ context
  this->PrimaryContext = new LZContext(this);

  // Get the copute queue group ordinal
  retrieveCmdQueueGroupOrdinal(this->cmdQueueGraphOrdinal);

  // Setup HipLZ device properties
  setupProperties(id);
}

LZDevice::LZDevice(hipDevice_t id,  ze_device_handle_t hDevice_, LZDriver* driver_,
		   ze_context_handle_t hContext, ze_command_queue_handle_t hQueue) {
  this->Index = id;
  this->hDevice = hDevice_;
  this->driver = driver_;

  // Retrieve device properties related data
  retrieveDeviceProperties();

  // Create HipLZ context
  this->PrimaryContext = new LZContext(this, hContext, hQueue);

  // Get the copute queue group ordinal
  retrieveCmdQueueGroupOrdinal(this->cmdQueueGraphOrdinal);

  // Setup HipLZ device properties.
  setupProperties(id);
}

// Retrieve device properties related data
void LZDevice::retrieveDeviceProperties() {
  ze_result_t status = ZE_RESULT_SUCCESS;

  // Initialize members used as input for zeDeviceGet*Properties() calls.
  this->deviceMemoryProps.pNext = nullptr;
  this->deviceComputeProps.pNext = nullptr;
  this->deviceCacheProps.pNext = nullptr;
  this->deviceModuleProps.pNext = nullptr;
  this->deviceProps.pNext = nullptr;

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

  // Query device module properties
  status = zeDeviceGetModuleProperties(this->hDevice, &(this->deviceModuleProps));
}

void LZDevice::registerModule(std::string* module) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);
  Modules.push_back(module);
}

// Register kernel function
bool LZDevice::registerFunction(std::string *module, const void *HostFunction,
				const char *FunctionName) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);

  logDebug("LZ REGISER FUCNTION {}", FunctionName);
  auto it = std::find(Modules.begin(), Modules.end(), module);
  if (it == Modules.end()) {
    logCritical("HipLZ Module PTR not FOUND: {}\n", (void *)module);
    return false;
  }

  HostPtrToModuleMap.emplace(std::make_pair(HostFunction, module));
  HostPtrToNameMap.emplace(std::make_pair(HostFunction, FunctionName));

  // std::cout << "Register function: " <<	FunctionName << "    " << (unsigned long)module->data() << std::endl;
  // Create HipLZ module
  std::string funcName(FunctionName);
  this->PrimaryContext->CreateModule((uint8_t* )module->data(), module->length(), funcName);

  return true;
}

// Register global variable
bool LZDevice::registerVar(std::string *module, const void *HostVar, const char *VarName) {
  std::string varName = (const char* )VarName;
  // std::cout << "Register variable: " <<	varName << "   module data address: " << (unsigned long)module->data() << std::endl;

  // Register global variable on primary context
  return getPrimaryCtx()->registerVar(module, HostVar, VarName);
}

bool LZDevice::registerVar(std::string *module, const void *HostVar, const char *VarName, size_t size) {
  // std::string varName = (const char* )VarName;
  // std::cout << "Register variable with size : " << varName << "   module data address: " << (unsigned long)module->data() << std::endl;

  // Register global variable on primary context
  return getPrimaryCtx()->registerVar(module, HostVar, VarName, size);
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
  this->PrimaryContext->reset();
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
  Properties.clockRate = 1000 * this->deviceProps.coreClockRate; // deviceMemoryProps.maxClockRate;
  // Dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  Properties.multiProcessorCount = this->deviceProps.numEUsPerSubslice * this->deviceProps.numSlices; // this->deviceComputeProps.maxTotalGroupSize;
  //??? Dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  Properties.l2CacheSize = this->deviceCacheProps.cacheSize;
  // Dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  Properties.totalConstMem = this->deviceMemoryProps.totalSize;
  // ??? Dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // as per gen architecture doc
  Properties.regsPerBlock = 4096;

  Properties.warpSize =
    this->deviceComputeProps.subGroupSizes[this->deviceComputeProps.numSubGroupSizes-1];

  // Replicate from OpenCL implementation

  // HIP and LZ uses int and uint32_t, respectively, for storing the
  // group count. Clamp the group count to INT_MAX to avoid 2^31+ size
  // being interpreted as negative number.
  constexpr unsigned int_max = std::numeric_limits<int>::max();
  Properties.maxGridSize[0] =
      std::min(this->deviceComputeProps.maxGroupCountX, int_max);
  Properties.maxGridSize[1] =
      std::min(this->deviceComputeProps.maxGroupCountY, int_max);
  Properties.maxGridSize[2] =
      std::min(this->deviceComputeProps.maxGroupCountZ, int_max);
  Properties.memoryClockRate = this->deviceMemoryProps.maxClockRate;
  Properties.memoryBusWidth = this->deviceMemoryProps.maxBusWidth;
  Properties.major = 2;
  Properties.minor = 0;

  Properties.maxThreadsPerMultiProcessor =
    this->deviceProps.numEUsPerSubslice * this->deviceProps.numThreadsPerEU; //  10;

  Properties.computeMode = hipComputeModeDefault;
  Properties.arch = {};

  Properties.arch.hasGlobalInt32Atomics = 1;
  Properties.arch.hasSharedInt32Atomics = 1;

  Properties.arch.hasGlobalInt64Atomics =
    (this->deviceModuleProps.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS) ? 1 : 0;
  Properties.arch.hasSharedInt64Atomics =
    (this->deviceModuleProps.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS) ? 1 : 0;

  Properties.arch.hasDoubles =
    (this->deviceModuleProps.flags & ZE_DEVICE_MODULE_FLAG_FP64) ? 1 : 0;

  Properties.clockInstructionRate = this->deviceProps.coreClockRate;
  Properties.concurrentKernels = 1;
  Properties.pciDomainID = 0;
  Properties.pciBusID = 0x10 + index;
  Properties.pciDeviceID = 0x40 + index;
  Properties.isMultiGpuBoard = 0;
  Properties.canMapHostMemory = 1;
  Properties.gcnArch = 0;
  Properties.integrated =
    (this->deviceProps.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) ? 1 : 0;
  Properties.maxSharedMemoryPerMultiProcessor =
    this->deviceComputeProps.maxSharedLocalMemory;
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

// Check if two devices can access peer from one to another
hipError_t LZDevice::CanAccessPeer(LZDevice& device, LZDevice& peerDevice, int* canAccessPeer) {
  ze_bool_t value;
  ze_result_t status = zeDeviceCanAccessPeer(device.GetDeviceHandle(), peerDevice.GetDeviceHandle(),
					     &value);

  LZ_RETURN_ERROR_MSG("HipLZ zeDeviceCanAccessPeer FAILED with return code ", status);

  if (value)
    * canAccessPeer = 1;
  else
    * canAccessPeer = 0;

  return hipSuccess;
}

// Check if the curren device can be accessed by another device
hipError_t LZDevice::CanBeAccessed(LZDevice& srcDevice, int* canAccessPeer) {
  int srcDeviceId = srcDevice.getHipDeviceT();
  if (peerAccessTable.find(srcDevice.getHipDeviceT()) == peerAccessTable.end()) {
    hipError_t res = CanAccessPeer(* this, srcDevice, canAccessPeer);
    if (res == hipSuccess && * canAccessPeer) {
      peerAccessTable[srcDeviceId] = PeerAccessState::Accessible;
    } else if (res == hipSuccess && !(* canAccessPeer))	{
      peerAccessTable[srcDeviceId] = PeerAccessState::UnAccessible;
    }

    return res;
  } else {
    PeerAccessState accessState = peerAccessTable[srcDeviceId];
    if (accessState == PeerAccessState::Accessible) {
      * canAccessPeer = 1;
    } else if (accessState == PeerAccessState::UnAccessible) {
      * canAccessPeer =	0;
    } else if (accessState == PeerAccessState::Accessible_Disabled) {
      * canAccessPeer = 0;
    } else {
       * canAccessPeer = 0;

       return hipErrorInvalidDevice;
    }

    return hipSuccess;
  }
}

// Enable/Disable the peer access from given device
hipError_t LZDevice::SetAccess(LZDevice& srcDevice, int flags, bool enableAccess) {
  int srcDeviceId = srcDevice.getHipDeviceT();
  if (peerAccessTable.find(srcDevice.getHipDeviceT()) == peerAccessTable.end()) {
    int canAccessPeer = 0;
    hipError_t res = CanAccessPeer(* this, srcDevice, &canAccessPeer);
    if (res == hipSuccess && canAccessPeer) {
      // The device can be accessed physically
      if (enableAccess)
	// Accessible
	peerAccessTable[srcDeviceId] = PeerAccessState::Accessible;
      else
	// Physically accessible but disabled
	peerAccessTable[srcDeviceId] = PeerAccessState::Accessible_Disabled;
    } else if (res == hipSuccess && !(canAccessPeer)) {
      peerAccessTable[srcDeviceId] = PeerAccessState::UnAccessible;
    }

    return res;
  } else {
    PeerAccessState accessState = peerAccessTable[srcDeviceId];
    if (accessState == PeerAccessState::Accessible) {
      if (!enableAccess)
	 peerAccessTable[srcDeviceId] = PeerAccessState::Accessible_Disabled;
    } else if (accessState == PeerAccessState::UnAccessible) {
      if (enableAccess)
	// Cand not set a physically unaccessible device to be accessible
	return hipErrorInvalidDevice;
    } else if (accessState == PeerAccessState::Accessible_Disabled) {
      if (enableAccess)
	peerAccessTable[srcDeviceId] = PeerAccessState::Accessible;
    } else {
       return hipErrorInvalidDevice;
    }

    return hipSuccess;
  }
}

// Check if the current device has same PCI bus ID as the one given by input
bool LZDevice::HasPCIBusId(int pciDomainID, int pciBusID, int pciDeviceID) {
  if (Properties.pciDomainID == pciDomainID &&
      Properties.pciBusID == pciBusID &&
      Properties.pciDeviceID == pciDeviceID)
    return true;

  return false;
}

hipError_t LZContext::recordEvent(hipStream_t stream, hipEvent_t event) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);
  return Queue->recordEvent(event) ? hipSuccess : hipErrorInvalidContext;
}

hipError_t LZContext::memCopy(void *dst, const void *src, size_t sizeBytes, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);
  return Queue->memCopy(dst, src, sizeBytes);
}

hipError_t LZContext::memCopyAsync(void *dst, const void *src, size_t sizeBytes, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);
  return Queue->memCopyAsync(dst, src, sizeBytes);
}

hipError_t LZContext::memFill(void *dst, size_t size, const void *pattern, size_t pattern_size, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);
  return Queue->memFill(dst, size, pattern, pattern_size);
}

hipError_t LZContext::memFillAsync(void *dst, size_t size, const void *pattern, size_t pattern_size, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);
  return Queue->memFillAsync(dst, size, pattern, pattern_size);
}

// The memory copy 2D support
hipError_t LZContext::memCopy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
				size_t width, size_t height, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);
  return Queue->memCopy2D(dst, dpitch, src, spitch, width, height);
}

hipError_t LZContext::memCopy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
				     size_t width, size_t height, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);
  return Queue->memCopy2DAsync(dst, dpitch, src, spitch, width, height);
}

// The memory copy 3D support
hipError_t LZContext::memCopy3D(void *dst, size_t dpitch, size_t dspitch,
				const void *src, size_t spitch, size_t sspitch,
				size_t width, size_t height, size_t depth, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);
  return Queue->memCopy3D(dst, dpitch, dspitch, src, spitch, sspitch, width, height, depth);
}

hipError_t LZContext::memCopy3DAsync(void *dst, size_t dpitch, size_t dspitch,
				     const void *src, size_t spitch, size_t sspitch,
				     size_t width, size_t height, size_t depth, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);
  return Queue->memCopy3DAsync(dst, dpitch, dspitch, src, spitch, sspitch, width, height, depth);
}

// Memory copy to texture object, i.e. image
hipError_t LZContext::memCopyToTexture(LZTextureObject* texObj, void* src, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);
  return Queue->memCopyToTexture(texObj, src);
}

// Make the advise for the managed memory (i.e. unified shared memory)
hipError_t LZContext::memAdvise(const void* ptr, size_t count, hipMemoryAdvise advise, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);
  return Queue->memAdvise(ptr, count, advise);
}

// Make meory prefetch
hipError_t LZContext::memPrefetch(const void* ptr, size_t size, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);
  return Queue->memPrefetch(ptr, size);
}

LZContext::LZContext(LZDevice* dev) : ClContext(0, 0) {
  this->lzDevice = dev;
  ze_context_desc_t ctxtDesc = {
    ZE_STRUCTURE_TYPE_CONTEXT_DESC,
    nullptr,
    0
  };
  ze_result_t status = zeContextCreate(dev->GetDriverHandle(), &ctxtDesc, &this->hContext);
  LZ_PROCESS_ERROR_MSG("HipLZ zeContextCreate Failed with return code ", status);
  logDebug("LZ CONTEXT {} via calling zeContextCreate ", status);

  // Create the default command queue
  this->DefaultQueue = new LZQueue(this, LZCommandList::CreateCmdList(this), 0, 0);
  // Initialize
  stopMonitor = false;
  monitorThreadId = 0;
  if (CreateMonitor())
    logError("LZ CONTEXT sync event monitor could not be created");
}

LZContext::LZContext(LZDevice* dev, ze_context_handle_t hContext_, ze_command_queue_handle_t hQueue)
  : ClContext(0, 0) {
  this->lzDevice = dev;
  this->hContext = hContext_;

  // Create the default command queue
  this->DefaultQueue = new LZQueue(this, hQueue, LZCommandList::CreateCmdList(this), 0, 0);
  // Initialize
  stopMonitor = false;
  monitorThreadId = 0;
  if (CreateMonitor())
    logError("LZ CONTEXT sync event monitor could not be created");
}

bool LZContext::CreateModule(uint8_t* funcIL, size_t ilSize, std::string funcName) {
  logDebug("LZ CREATE MODULE {} ", funcName);

  LZProgram* lzModule = 0;
  if (this->IL2Module.find(funcIL) == this->IL2Module.end()) {
    // Create HipLZ module and register it
    lzModule = new LZProgram(this, funcIL, ilSize);
    this->IL2Module[funcIL] = lzModule;
  } else
    lzModule = this->IL2Module[funcIL];
  
  // Create kernel object
  lzModule->CreateKernel(funcName);

  return true;
}

// Launch HipLZ kernel (old HIP launch API).
hipError_t LZContext::launchHostFunc(const void* HostFunction) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  synchronizeQueues(DefaultQueue);
  hipFunction_t Kernel = 0;
  // logDebug("LAUNCH HOST FUNCTION {} ",  this->lzModule != nullptr);
  // if (!this->lzModule) {
  //   HIP_PROCESS_ERROR_MSG("Hiplz LZProgram was not created before invoking kernel?", hipErrorInitializationError);
  // }

  std::string HostFunctionName = this->lzDevice->GetHostFunctionName(HostFunction);
  Kernel = GetKernelByFunctionName(HostFunctionName);
  if (!Kernel) {
    HIP_PROCESS_ERROR_MSG("Hiplz LZProgram was not created before invoking kernel?", hipErrorInitializationError);
  }

  logDebug("LAUNCH HOST FUNCTION {} - {} ", HostFunctionName,  Kernel != nullptr);

  if (!Kernel)
    HIP_PROCESS_ERROR_MSG("Hiplz no LZkernel found?", hipErrorInitializationError);

  LZExecItem *Arguments;
  Arguments = (LZExecItem* )ExecStack.top();
  ExecStack.pop();

  return Arguments->launch(Kernel);
}

// Launch HipLZ kernel (new HIP launch API).
hipError_t LZContext::launchHostFunc(const void *hostFunction, dim3 numBlocks,
                               dim3 dimBlocks, void **args,
                               size_t sharedMemBytes, hipStream_t stream) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);
  logDebug("Launch kernel via new HIP launch API.");

  std::string hostFunctionName =
      this->lzDevice->GetHostFunctionName(hostFunction);
  hipFunction_t kernel = GetKernelByFunctionName(hostFunctionName);
  if (!kernel) {
    HIP_PROCESS_ERROR_MSG(
        "Hiplz LZProgram was not created before invoking kernel?",
        hipErrorInitializationError);
  }

  LZExecItem Arguments(numBlocks, dimBlocks, sharedMemBytes, Queue);
  Arguments.setArgsPointer(args);
  return Arguments.launch(kernel);
}

hipError_t LZContext::launchWithKernelParams(dim3 grid, dim3 block, size_t shared,
                                             hipStream_t stream, void **kernelParams,
                                             hipFunction_t kernel) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);
  LZExecItem Arguments(grid, block, shared, Queue);
  Arguments.setArgsPointer(kernelParams);
  return Arguments.launch(kernel);
}

hipError_t LZContext::launchWithExtraParams(dim3 grid, dim3 block,
                                            size_t shared, hipStream_t stream,
                                            void **extraParams,
                                            hipFunction_t kernel) {
  FIND_QUEUE_LOCKED(stream);
  synchronizeQueues(Queue);

  void *args = nullptr;
  size_t size = 0;

  void **p = extraParams;
  while (*p && (*p != HIP_LAUNCH_PARAM_END)) {
    if (*p == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
      args = (void *)p[1];
      p += 2;
      continue;
    } else if (*p == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
      // size = (size_t)p[1];
      size = *(size_t *)p[1];

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

  /*x
    This check seems to be incorrect now?
  if (size != kernel->getTotalArgSize()) {
    logError("extraParams doesn't have correct size {} and total arg size is {}\n", size, kernel->getTotalArgSize());
    return hipErrorLaunchFailure;
  }
  */

  LZExecItem Arguments(grid, block, shared, Queue);
  OCLFuncInfo *FuncInfo = kernel->getFuncInfo();
  size_t offset = 0;
  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];
    if (ai.type == OCLType::Pointer) {
      // TODO other than global AS ?
      assert(ai.size == sizeof(void *));
    }
    Arguments.setArg(args, ai.size, offset);
    offset += ai.size;
    args = (char *)args + ai.size;
  }
  return Arguments.launch(kernel);
}

// Get HipLZ kernel via function name
hipFunction_t LZContext::GetKernelByFunctionName(std::string funcName) {
  hipFunction_t lzKernel = 0;
  // Go through all modules in current HipLZ context to find the kernel via function name
  for (auto mod : this->IL2Module) {
    LZProgram* lzModule = mod.second;
    lzKernel = lzModule->getKernel(funcName);
    if (lzKernel)
      break;
  }

  return lzKernel;
}

// Allocate memory via Level-0 runtime
void* LZContext::allocate(size_t size, size_t alignment, ClMemoryType memTy) {
  void *ptr = 0;
  if (memTy == ClMemoryType::Shared) {
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
  } else if (memTy == ClMemoryType::Device) {
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

void * LZContext::allocate(size_t size, ClMemoryType memTy) {
  return allocate(size, 0x1000, memTy);
}

void * LZContext::allocate(size_t size) {
  return allocate(size, 0x1000, ClMemoryType::Device);
}

bool LZContext::free(void *p) {
  ze_result_t status = zeMemFree(this->hContext, p);
  LZ_PROCESS_ERROR_MSG("HipLZ could not free memory with error code: ", status);

  return true;
}

// Register global variable
bool LZContext::registerVar(std::string *module, const void *HostVar, const char *VarName) {
  size_t VarSize = 0;
  void* VarPtr = 0;

  for (auto mod : this->IL2Module) {
    LZProgram* lzModule = mod.second;

    if (lzModule->getSymbolAddressSize(VarName, &VarPtr, &VarSize)) {
      // Register HipLZ module, address and size information based on symbol name
      GlobalVarsMap[VarName] = std::make_tuple(lzModule, VarPtr, VarSize);
      // Register size information based on devie pointer
      globalPtrs.addGlobalPtr(VarPtr, VarSize);

      break;
    }
  }

  return VarPtr != 0;
}

bool LZContext::registerVar(std::string *module, const void *HostVar, const char *VarName, size_t size) {
  size_t VarSize = size;
  void* VarPtr = 0;

  for (auto mod : this->IL2Module) {
    LZProgram* lzModule = mod.second;

    if (lzModule->getSymbolAddressSize(VarName, &VarPtr, &VarSize)) {
      // Register HipLZ module, address and size information based on symbol name
      GlobalVarsMap[VarName] = std::make_tuple(lzModule, VarPtr, VarSize);
      // Register size information based on devie pointer
      globalPtrs.addGlobalPtr(VarPtr, VarSize);

      break;
    }
  }

  return VarPtr != 0;
}

// Get the address and size for the given symbol's name
bool LZContext::getSymbolAddressSize(const char *name, hipDeviceptr_t *dptr, size_t *bytes) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  // Check if the global variable has been registered
  auto it = this->GlobalVarsMap.find(name);
  if (it != this->GlobalVarsMap.end()) {
    *dptr = std::get<1>(it->second);
    *bytes = std::get<2>(it->second);

    return true;
  }

  // Go through HipLZ modules and identify the relevant global pointer information
  for (auto mod : IL2Module) {
    LZProgram* lzModule = mod.second;
    if (lzModule->getSymbolAddressSize(name, dptr, bytes)) {
      // Register HipLZ module, address and size information based on symbol name
      GlobalVarsMap[(const char *)name] = std::make_tuple(lzModule, *dptr, *bytes);
      // Register size information based on devie pointer
      globalPtrs.addGlobalPtr(*dptr, *bytes);

      return true;
    }
  }

  return false;
}

// Create stream/queue
bool LZContext::createQueue(hipStream_t *stream, unsigned int flags, int priority) {
  hipStream_t Ptr = new LZQueue(this, flags, priority);
  Queues.insert(Ptr);
  *stream = Ptr;
  return true;
}

bool LZContext::releaseQueue(hipStream_t stream) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  auto I = Queues.find(stream);
  if (I == Queues.end())
    return false;
  hipStream_t QueuePtr = *I;
  delete QueuePtr;
  Queues.erase(I);
  return true;
}

// Create HipLZ event
hipEvent_t LZContext::createEvent(unsigned flags) {
  std::lock_guard<std::mutex> Lock(ContextMutex);
  return new LZEvent(this, flags);
}

#define NANOSECS 1000000000

// Get the elapse between two events
hipError_t LZContext::eventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  std::lock_guard<std::mutex> Lock(ContextMutex);

  // assert(start->isFromContext(this));
  // assert(stop->isFromContext(this));

  if (!start->isRecordingOrRecorded() || !stop->isRecordingOrRecorded())
    return hipErrorInvalidResourceHandle;

  start->updateFinishStatus();
  stop->updateFinishStatus();
  if (!start->isFinished() || !stop->isFinished())
    return hipErrorNotReady;

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

// Create an event pool for synchronization
bool LZContext::CreateSyncEventPool(uint32_t count, ze_event_pool_handle_t &pool) {
  ze_result_t status;
  ze_event_pool_desc_t desc = {
    .stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
    .pNext = nullptr,
    .flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
    .count = count
  };
  ze_device_handle_t dev = lzDevice->GetDeviceHandle();
  status = zeEventPoolCreate(hContext, &desc, 1, &dev, &pool);
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventPoolCreate FAILED with return code ", status);
  return true;
}

bool LZContext::GetSyncEvent(hipContextSyncData *syncEvent) {
  bool res = false;
  {
    std::lock_guard<std::mutex> Lock(syncDatasMutex);
    if (!this->syncDatas.empty()) {
      *syncEvent = syncDatas.front();
      syncDatas.pop_front();
      res = true;
    }
  }
  return res;
}

static void * LZContextEventMonitor(void* data) {
  LZContext *lzContext = (LZContext *)data;
  while (!lzContext->StopMonitor()) {
    hipContextSyncData syncEvent;
    while (lzContext->GetSyncEvent(&syncEvent)) {
      ze_result_t status;
      status = zeEventHostSynchronize(syncEvent.waitEvent, UINT64_MAX );
      LZ_PROCESS_ERROR_MSG("HipLZ zeEventHostSynchronize FAILED with return code ", status);
      status = zeEventDestroy(syncEvent.waitEvent);
      LZ_PROCESS_ERROR_MSG("HipLZ zeEventDestroy FAILED with return code ", status);
      for (ze_event_handle_t event : syncEvent.signaledEvents) {
        status = zeEventDestroy(event);
        LZ_PROCESS_ERROR_MSG("HipLZ zeEventDestroy FAILED with return code ", status);
      }
      status = zeEventPoolDestroy(syncEvent.eventPool);
      LZ_PROCESS_ERROR_MSG("HipLZ zeEventDestroy FAILED with return code ", status);
    }
    pthread_yield();
  }
  return 0;
}

bool LZContext::CreateMonitor() {
  return 0 != pthread_create(&(this->monitorThreadId), 0, LZContextEventMonitor, (void* )this);
}

void LZContext::WaitEventMonitor() {
  if  (this->monitorThreadId == 0)
    return;

  // Join the event monitor thread
  pthread_join(this->monitorThreadId, NULL);
}

LZContext::~LZContext() {
  stopMonitor = true;
  WaitEventMonitor();
}

// Enforce HIP correct stream synchronization (For now only Legacy is suported by HIP)
void LZContext::synchronizeQueues(hipStream_t queue) {
  ze_result_t status;

  if (queue == DefaultQueue) {
    // Enforce synchonization with all non NonBlocking streams
    // if there is any
    size_t blocking = 0;
    for (hipStream_t Q : Queues)
      if (!Q->isNonBlocking())
        blocking++;
    if (blocking != 0) {
      // Need an event pool of to track barriers and signal event
      hipContextSyncData syncData;

      CreateSyncEventPool(blocking + 2, syncData.eventPool);
      syncData.signaledEvents.reserve(blocking + 1);
      ze_event_handle_t event;
      ze_event_desc_t desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
        .pNext = nullptr,
        .index = 0,
        .signal = ZE_EVENT_SCOPE_FLAG_DEVICE,
        .wait = ZE_EVENT_SCOPE_FLAG_DEVICE
      };
      uint32_t i = 0;
      // barriers on all blocking queues
      for (hipStream_t Q : Queues)
        if (Q->isNonBlocking()) {
	  desc.index = i++;
          status = zeEventCreate(syncData.eventPool, &desc, &event);
          LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
          ((LZQueue*)Q)->enqueueZeBarrier(event);
          syncData.signaledEvents.push_back(event);
	}
      // barrier on the default queue
      desc.index = i++;
      status = zeEventCreate(syncData.eventPool, &desc, &event);
      LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
      ((LZQueue*)queue)->enqueueZeBarrier(event);
      syncData.signaledEvents.push_back(event);
      // barrier waiting for all previous barriers
      desc.index = i++;
      status = zeEventCreate(syncData.eventPool, &desc, &event);
      LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
      ((LZQueue*)queue)->enqueueZeBarrier(event, blocking + 1, &syncData.signaledEvents[0]);
      syncData.waitEvent = event;
      {
        std::lock_guard<std::mutex> Lock(syncDatasMutex);
        syncDatas.push_back(syncData);
      }
    } else {
      // Only need to synchronize queue
      ((LZQueue*)queue)->enqueueZeBarrier();
    }
  } else if (queue->isNonBlocking()) {
    // Stream is NonBlocking no outside synchronization but
    // sequential execution inside the stream
    ((LZQueue*)queue)->enqueueZeBarrier();
  } else {
    // This is a non NonBlocking stream, must be synchronized
    // with the NULL stream and itself
    hipContextSyncData syncData;

    CreateSyncEventPool(3, syncData.eventPool);
    syncData.signaledEvents.reserve(2);
    ze_event_handle_t event;
    ze_event_desc_t desc = {
      .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
      .pNext = nullptr,
      .index = 0,
      .signal = ZE_EVENT_SCOPE_FLAG_DEVICE,
      .wait = ZE_EVENT_SCOPE_FLAG_DEVICE
    };

    status = zeEventCreate(syncData.eventPool, &desc, &event);
    LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
    ((LZQueue*)DefaultQueue)->enqueueZeBarrier(event);
    syncData.signaledEvents.push_back(event);

    desc.index = 1;
    status = zeEventCreate(syncData.eventPool, &desc, &event);
    LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
    ((LZQueue*)queue)->enqueueZeBarrier(event);
    syncData.signaledEvents.push_back(event);

    desc.index = 2;
    status = zeEventCreate(syncData.eventPool, &desc, &event);
    LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
    ((LZQueue*)queue)->enqueueZeBarrier(event, 2, &syncData.signaledEvents[0]);
    syncData.waitEvent = event;
    {
      std::lock_guard<std::mutex> Lock(syncDatasMutex);
      syncDatas.push_back(syncData);
    }
  }
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

// Create HipLZ program that is derived from ClProgram
ClProgram *LZContext::createProgram(std::string &binary) {
  uint8_t* funcIL = (uint8_t* )binary.data();
  size_t ilSize = binary.length();

  LZProgram* lzProgram = 0;
  if (this->IL2Module.find(funcIL) == this->IL2Module.end()) {
    // Create HipLZ module and register it
    lzProgram = new LZProgram(this, funcIL, ilSize);
    this->IL2Module[funcIL] = lzProgram;
  } else
    lzProgram = this->IL2Module[funcIL];

  if (!lzProgram)
    HIP_PROCESS_ERROR_MSG("HipLZ counld not create a LZProgram ", hipErrorNotSupported);
  
  return lzProgram;
}

// Create Level-0 image object
LZImage* LZContext::createImage(hipResourceDesc* resDesc, hipTextureDesc* texDesc) {
  if (resDesc == nullptr || texDesc == nullptr)
    return nullptr;

  return new LZImage(this, resDesc, texDesc);
}

// Create HIP texture object
hipTextureObject_t LZContext::createTextureObject(const hipResourceDesc* pResDesc,
						  const hipTextureDesc* pTexDesc,
						  const struct hipResourceViewDesc* pResViewDesc) {
  // Create the Hip Texture struct here
  LZTextureObject*  texObj =
    LZTextureObject::CreateTextureObject(this, pResDesc, pTexDesc, pResViewDesc);

  return (hipTextureObject_t)texObj;
}

// Destroy HIP texture object 
bool LZContext::destroyTextureObject(hipTextureObject_t textureObject) {
  return LZTextureObject::DestroyTextureObject((LZTextureObject* )textureObject);
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

// Initialize HipLZ driver via pre-initialized resource
bool LZDriver::InitDriver(std::vector<LZDriver* >& drivers,
			  const ze_device_type_t deviceType,
			  ze_driver_handle_t hDriver,
			  ze_device_handle_t hDevice,
			  ze_context_handle_t hContext,
			  ze_command_queue_handle_t hQueue) {
  const ze_device_type_t type = ZE_DEVICE_TYPE_GPU;

  if (drivers.size() != 0) {
    // Clean the pre-initialized drives
    drivers.clear();
    // TODO: check the safeness
  }

  if (hDriver == nullptr || hDevice == nullptr || hContext == nullptr || hQueue == nullptr) {
    logDebug("HipLZ Initialize Driver, Device, Context and Queue from outside failed \n");
    return false;
  }

  LZDriver* driver = new LZDriver(hDriver, deviceType, hDevice, hContext, hQueue);
  drivers.push_back(driver);

  if (NumLZDevices == 0) {
    HIP_PROCESS_ERROR(hipErrorNoDevice);
  }

  // Set the number of drivers
  NumLZDrivers = 1;

  return true;
}

// Collect HipLZ device that belongs to this driver
bool LZDriver::FindHipLZDevices(ze_device_handle_t hDevice,
				ze_context_handle_t hContext,
				ze_command_queue_handle_t hQueue) {
  if (hDevice != nullptr && hContext != nullptr && hQueue != nullptr) {
    this->devices.push_back(new LZDevice(0, hDevice, this, hContext, hQueue));
    return true;
  } else if (hDevice != nullptr || hContext != nullptr || hQueue != nullptr) {
    logDebug("DEVICE OR CONTEXT OR QUEUE WAS MISTAKENLY INITIALIZED\n");
    return false;
  }

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
static void * LZQueueEventMonitor(void* data) {
  LZQueue* lzQueue = (LZQueue* )data;
  while (lzQueue->GetContext() != nullptr) {
    // Invoke callbacks
    hipStreamCallbackData callback_data;
    while (lzQueue->GetCallback(&callback_data)) {
      ze_result_t status;
      status = zeEventHostSynchronize(callback_data.waitEvent, UINT64_MAX );
      LZ_PROCESS_ERROR_MSG("HipLZ zeEventHostSynchronize FAILED with return code ", status);
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

LZQueue::LZQueue(LZContext* lzContext_, unsigned int f, int p) : ClQueue(f, p) {
  // Initialize Level-0 related class fields
  this->lzContext = lzContext_;
  this->defaultCmdList = nullptr;
  this->monitorThreadId = 0;

  // Initialize Level-0 queue
  initializeQueue(lzContext, true);
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
    this->defaultCmdList = LZCommandList::CreateCmdList(this->lzContext);
  }
}

LZQueue::LZQueue(LZContext* lzContext_, LZCommandList* lzCmdList, unsigned int f, int p) : ClQueue(f, p) {
  // Initialize Level-0 related class fields
  this->lzContext = lzContext_;
  this->defaultCmdList = lzCmdList;
  this->monitorThreadId = 0;

  // Initialize Level-0 queue
  initializeQueue(lzContext);
}

LZQueue::LZQueue(LZContext* lzContext_, ze_command_queue_handle_t hQueue_, LZCommandList* lzCmdList, unsigned int f, int p) : ClQueue(f, p) {
  // Initialize Level-0 related class fields
  this->lzContext = lzContext_;
  this->defaultCmdList = lzCmdList;
  this->monitorThreadId = 0;

  this->hQueue = hQueue_;
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
  ze_command_list_handle_t list = defaultCmdList->GetCommandListHandle();
  ze_event_handle_t ev = ((LZEvent *)event)->GetEventHandle();
  ze_result_t status = zeCommandListAppendBarrier(list, nullptr, 1, &ev);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendBarrier FAILED with return code ", status);
  return true;
}

bool LZQueue::enqueueZeBarrier(ze_event_handle_t event, uint32_t waitCount, ze_event_handle_t *waitList) {
  ze_command_list_handle_t list = defaultCmdList->GetCommandListHandle();
  ze_result_t status = zeCommandListAppendBarrier(list, event, waitCount, waitList);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendBarrier FAILED with return code ", status);
  return true;
}

// Add call back
bool LZQueue::addCallback(hipStreamCallback_t callback, void *userData) {
  std::lock_guard<std::mutex> Lock(QueueMutex);

  hipStreamCallbackData Data;
  ze_result_t status;

  ze_event_desc_t ev_desc = {};
  ev_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  ev_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  ev_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;

  lzContext->CreateSyncEventPool(3, Data.eventPool);
  ev_desc.index = 0;
  status = zeEventCreate(Data.eventPool, &ev_desc, &(Data.waitEvent));
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
  ev_desc.index = 1;
  status = zeEventCreate(Data.eventPool, &ev_desc, &(Data.signalEvent));
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
  ev_desc.index = 2;
  status = zeEventCreate(Data.eventPool, &ev_desc, &(Data.waitEvent2));
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);
  Data.Stream = this;
  Data.Callback = callback;
  Data.UserData = userData;
  Data.Status = hipSuccess;
  ze_command_list_handle_t list = defaultCmdList->GetCommandListHandle();
  status = zeCommandListAppendBarrier(list, Data.waitEvent, 0, NULL);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendBarrier FAILED with return code ", status);
  status = zeCommandListAppendBarrier(list, NULL, 1, &(Data.signalEvent));
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendBarrier FAILED with return code ", status);
  status = zeCommandListAppendSignalEvent(list, Data.waitEvent2);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendSignalEvent FAILED with return code ", status);

  // Add event callback to callback list
  {
    std::lock_guard<std::mutex> Lock(CallbacksMutex);
    callbacks.push_back(Data);
  }

  // Create event monitor on demand
  CheckAndCreateMonitor();

  return true;
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
  return 0 != pthread_create(&(this->monitorThreadId), 0, LZQueueEventMonitor, (void* )this);
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

  if (event == nullptr)
    HIP_PROCESS_ERROR_MSG("HipLZ get null Event recorded?", hipErrorInitializationError);

  CHECK_QUEUE_COMMAND_LIST();

  // Record event to stream
  return event->recordStream(this);
}

// Memory copy support
hipError_t LZQueue::memCopy(void *dst, const void *src, size_t size) {
  CHECK_QUEUE_COMMAND_LIST();
  this->defaultCmdList->ExecuteMemCopy(this, dst, src, size);
  return hipSuccess;
}

hipError_t LZQueue::memCopyAsync(void *dst, const void *src, size_t sizeBytes) {
  CHECK_QUEUE_COMMAND_LIST();
  this->defaultCmdList->ExecuteMemCopyAsync(this, dst, src, sizeBytes);
  return hipSuccess;
}

// Memory fill support
hipError_t LZQueue::memFill(void *dst, size_t size, const void *pattern, size_t pattern_size) {
  CHECK_QUEUE_COMMAND_LIST();
  this->defaultCmdList->ExecuteMemFill(this, dst, size, pattern, pattern_size);
  return hipSuccess;
}

hipError_t LZQueue::memFillAsync(void *dst, size_t size, const void *pattern, size_t pattern_size) {
  CHECK_QUEUE_COMMAND_LIST();
  this->defaultCmdList->ExecuteMemFillAsync(this, dst, size, pattern, pattern_size);
  return hipSuccess;
}

// The memory copy 2D support
hipError_t LZQueue::memCopy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
			      size_t width, size_t height) {
  CHECK_QUEUE_COMMAND_LIST();
  this->defaultCmdList->ExecuteMemCopyRegion(this, dst, dpitch, src, spitch, width, height);
  return hipSuccess;
}

hipError_t LZQueue::memCopy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
				   size_t width, size_t height) {
  CHECK_QUEUE_COMMAND_LIST();
  this->defaultCmdList->ExecuteMemCopyRegionAsync(this, dst, dpitch, src, spitch, width, height);
  return hipSuccess;
}

// The memory copy 3D support
hipError_t LZQueue::memCopy3D(void *dst, size_t dpitch, size_t dspitch,
			      const void *src, size_t spitch, size_t sspitch,
			      size_t width, size_t height, size_t depth) {
  CHECK_QUEUE_COMMAND_LIST();
  this->defaultCmdList->ExecuteMemCopyRegion(this, dst, dpitch, dspitch, src, spitch, sspitch,
					     width, height, depth);
  return hipSuccess;
}

hipError_t LZQueue::memCopy3DAsync(void *dst, size_t dpitch, size_t dspitch,
				   const void *src, size_t spitch, size_t sspitch,
				   size_t width, size_t height, size_t depth) {
  CHECK_QUEUE_COMMAND_LIST();
  this->defaultCmdList->ExecuteMemCopyRegionAsync(this, dst, dpitch, dspitch, src, spitch, sspitch,
						  width, height, depth);
  return hipSuccess;
}

hipError_t LZQueue::memCopyToTexture(LZTextureObject* texObj, void* src) {
  CHECK_QUEUE_COMMAND_LIST();
  this->defaultCmdList->ExecuteMemCopyToTexture(this, texObj, src);

  return hipSuccess;
}

// Make meory prefetch
hipError_t LZQueue::memPrefetch(const void* ptr, size_t size) {
  CHECK_QUEUE_COMMAND_LIST();
  this->GetDefaultCmdList()->ExecuteMemPrefetchAsync(this, ptr, size);
  return hipSuccess;
}

// Make the advise for the managed memory (i.e. unified shared memory)
hipError_t LZQueue::memAdvise(const void* ptr, size_t count, hipMemoryAdvise advise) {
  CHECK_QUEUE_COMMAND_LIST();
  this->GetDefaultCmdList()->ExecuteMemAdviseAsync(this, ptr, count, advise);
  return hipSuccess;
}

// Launch kernel support
hipError_t LZQueue::launch3(ClKernel *Kernel, dim3 grid, dim3 block) {
  HIP_PROCESS_ERROR_MSG("Not support LZQueue::launch3 yet!", hipErrorNotSupported);
}

// Launch kernel support
hipError_t LZQueue::launch(ClKernel *Kernel, ExecItem *Arguments) {
  CHECK_QUEUE_COMMAND_LIST();
  this->defaultCmdList->ExecuteKernelAsync(this, (LZKernel* )Kernel, (LZExecItem* )Arguments);
  return hipSuccess;
}

bool LZQueue::getNativeInfo(unsigned long* nativeInfo, int* size) {
  // Here we retrieve 4 handler inforations, Driver, Device, Context and Queue
  * size = 4;

  // Get queue handler
  nativeInfo[3] = (unsigned long)this->GetQueueHandle();

  // Get context handler
  LZContext* ctx = this->GetContext();
  nativeInfo[2] = (unsigned long)ctx->GetContextHandle();

  // Get device handler
  LZDevice* device = ctx->GetDevice();
  nativeInfo[1] = (unsigned long)device->GetDeviceHandle();

  // Get driver handler
  nativeInfo[0] = (unsigned long)device->GetDriverHandle();

  return true;
}

LZCommandList::LZCommandList(LZContext* lzContext_) {
  this->lzContext = lzContext_;

  // Initialize the shared memory buffer
  this->shared_buf = this->lzContext->allocate(32, 8, ClMemoryType::Shared);

  // Initialize the uint64_t part as 0
   * (uint64_t* )this->shared_buf = 0;
}

// Initialize stand Level-0 command list
bool LZStdCommandList::initializeCmdList() {
  // Create the command list
  ze_command_list_desc_t clDesc;
  clDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  clDesc.flags = ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY; // default hehaviour
  clDesc.commandQueueGroupOrdinal = 0;
  clDesc.pNext = nullptr;
  ze_result_t status = zeCommandListCreate(lzContext->GetContextHandle(),
					   lzContext->GetDevice()->GetDeviceHandle(),
					   &clDesc, &hCommandList);

  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListCreate FAILED with return code ", status);
  logDebug("LZ COMMAND LIST CREATION via calling zeCommandListCreate {} ", status);

  return true;
}

// Initialize immediate Level-0 command list
bool LZImmCommandList::initializeCmdList() {
  // Create command list via immidiately associated with a queue
  ze_command_queue_desc_t cqDesc;
  cqDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  cqDesc.pNext = nullptr;
  cqDesc.ordinal = 0;
  cqDesc.index = 0;
  cqDesc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY; // 0;
  cqDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  cqDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

  ze_result_t status = zeCommandListCreateImmediate(this->lzContext->GetContextHandle(),
						    this->lzContext->GetDevice()->GetDeviceHandle(),
						    &cqDesc, &hCommandList);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListCreate FAILED with return code ", status);
  logDebug("LZ COMMAND LIST CREATION via calling zeCommandListCreateImmediate {} ", status);

  // Initialize the internal event pool and finish event
  ze_event_pool_desc_t ep_desc = {};
  ep_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  ep_desc.count = 1;
  ep_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  ze_event_desc_t ev_desc = {};
  ev_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  ev_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  ev_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  ze_device_handle_t dev = lzContext->GetDevice()->GetDeviceHandle();
  status = zeEventPoolCreate(lzContext->GetContextHandle(), &ep_desc, 1, &dev, &(this->eventPool));
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventPoolCreate FAILED with return code ", status);
  status = zeEventCreate(this->eventPool, &ev_desc, &(this->finishEvent));
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventCreate FAILED with return code ", status);

  return true;
}

// Create HipLZ command list
LZCommandList* LZCommandList::CreateCmdList(LZContext* lzContext, bool immediate) {
  if (immediate)
    return new LZImmCommandList(lzContext);
  else
    return new LZStdCommandList(lzContext);
}

// Execute the Level-0 kernel

void LZCommandList::kernel(LZQueue* lzQueue, LZKernel* Kernel, LZExecItem* Arguments) {
  // Set group size
  ze_result_t status = zeKernelSetGroupSize(Kernel->GetKernelHandle(),
					    Arguments->BlockDim.x, Arguments->BlockDim.y,
					    Arguments->BlockDim.z);
  LZ_PROCESS_ERROR_MSG("could not set group size! ", status);

  logDebug("LZ KERNEL EXECUTION via calling zeKernelSetGroupSize {} ", status);

  // Set all kernel function arguments
  Arguments->setupAllArgs(Kernel);

  // Launch kernel via Level-0 command list
  uint32_t numGroupsX = Arguments->GridDim.x;
  uint32_t numGroupsY = Arguments->GridDim.y;
  uint32_t numGroupsz = Arguments->GridDim.z;
  ze_group_count_t hLaunchFuncArgs = { numGroupsX, numGroupsY, numGroupsz };

  status = zeCommandListAppendLaunchKernel(hCommandList, Kernel->GetKernelHandle(), &hLaunchFuncArgs,
                                           nullptr, 0, nullptr);
  LZ_PROCESS_ERROR_MSG("Hiplz zeCommandListAppendLaunchKernel FAILED with return code  ", status);

  logDebug("LZ KERNEL EXECUTION via calling zeCommandListAppendLaunchKernel {} ", status);
}


bool LZCommandList::ExecuteKernel(LZQueue* lzQueue, LZKernel* Kernel, LZExecItem* Arguments) {
  kernel(lzQueue, Kernel, Arguments);
  return Execute(lzQueue);
}

bool LZCommandList::ExecuteKernelAsync(LZQueue* lzQueue, LZKernel* Kernel, LZExecItem* Arguments) {
  kernel(lzQueue, Kernel, Arguments);
  return ExecuteAsync(lzQueue);
}

// Execute HipLZ memory copy command

void LZCommandList::memCopy(LZQueue* lzQueue, void *dst, const void *src, size_t sizeBytes) {
  ze_result_t status = zeCommandListAppendMemoryCopy(hCommandList, dst, src, sizeBytes,
                                                     nullptr, 0, nullptr);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendMemoryCopy FAILED with return code ", status);
}

bool LZCommandList::ExecuteMemCopy(LZQueue* lzQueue, void *dst, const void *src, size_t sizeBytes) {
  memCopy(lzQueue, dst, src, sizeBytes);
  return Execute(lzQueue);
}

bool LZCommandList::ExecuteMemCopyAsync(LZQueue* lzQueue, void *dst, const void *src, size_t sizeBytes) {
  memCopy(lzQueue, dst, src, sizeBytes);
  return ExecuteAsync(lzQueue);
}

// Execute memory HipLZ copy region
void LZCommandList::memCopyRegion(LZQueue* lzQueue, void *dst, size_t dpitch,
                                  const void *src, size_t spitch,
                                  size_t width, size_t height) {
  // Create region
  ze_copy_region_t dstRegion;
  dstRegion.originX = 0;
  dstRegion.originY = 0;
  dstRegion.originZ = 0;
  dstRegion.width = width;
  dstRegion.height = height;
  dstRegion.depth = 0;
  ze_copy_region_t srcRegion;
  srcRegion.originX = 0;
  srcRegion.originY = 0;
  srcRegion.originZ = 0;
  srcRegion.width = width;
  srcRegion.height = height;
  srcRegion.depth = 0;
  ze_result_t status = zeCommandListAppendMemoryCopyRegion(hCommandList, dst, &dstRegion, dpitch, 0,
							   src, &srcRegion, spitch, 0,
							   nullptr, 0, nullptr);

  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendMemoryCopyRegion FAILED with return code ", status);
}

bool LZCommandList::ExecuteMemCopyRegion(LZQueue* lzQueue, void *dst, size_t dpitch,
                                         const void *src, size_t spitch,
                                         size_t width, size_t height) {
  memCopyRegion(lzQueue, dst, dpitch, src, spitch, width, height);
  return Execute(lzQueue);
}

bool LZCommandList::ExecuteMemCopyRegionAsync(LZQueue* lzQueue, void *dst, size_t dpitch,
                                              const void *src, size_t spitch,
                                              size_t width, size_t height) {
  memCopyRegion(lzQueue, dst, dpitch, src, spitch, width, height);
  return ExecuteAsync(lzQueue);
}

bool LZCommandList::ExecuteMemCopyToTexture(LZQueue* lzQueue, LZTextureObject* texObj, void* src) {
  memCopyToTexture(lzQueue, texObj, src);
  return Execute(lzQueue);
}

void LZCommandList::memCopyRegion(LZQueue* lzQueue, void *dst, size_t dpitch, size_t dspitch,
                                  const void *src, size_t spitch, size_t sspitch,
                                  size_t width, size_t height, size_t depth) {
  ze_copy_region_t dstRegion;
  dstRegion.originX = 0;
  dstRegion.originY = 0;
  dstRegion.originZ = 0;
  dstRegion.width = width;
  dstRegion.height = height;
  dstRegion.depth = depth;
  ze_copy_region_t srcRegion;
  srcRegion.originX = 0;
  srcRegion.originY = 0;
  srcRegion.originZ = 0;
  srcRegion.width = width;
  srcRegion.height = height;
  srcRegion.depth = depth;
  ze_result_t status = zeCommandListAppendMemoryCopyRegion(hCommandList, dst, &dstRegion, dpitch,
                                                           dspitch, src, &srcRegion, spitch, sspitch,
                                                           nullptr, 0, nullptr);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendMemoryCopyRegion FAILED with return code ", status);
}

void LZCommandList::memCopyToTexture(LZQueue* lzQueue, LZTextureObject* texObj, void* src) {
  ze_image_handle_t imageHandle = (ze_image_handle_t)texObj->image;
  ze_result_t status =
    zeCommandListAppendImageCopyFromMemory(hCommandList, imageHandle, src, 0, 0, 0, 0);
  LZ_PROCESS_ERROR_MSG("HipLZ  FAILED with return code ", status);
}

bool LZCommandList::ExecuteMemCopyRegion(LZQueue* lzQueue, void *dst, size_t dpitch, size_t dspitch,
                                         const void *src, size_t spitch, size_t sspitch,
                                         size_t width, size_t height, size_t depth) {
  memCopyRegion(lzQueue, dst, dpitch, dspitch, src, spitch, sspitch, width, height, depth);
  return Execute(lzQueue);
}

bool LZCommandList::ExecuteMemCopyRegionAsync(LZQueue* lzQueue, void *dst, size_t dpitch, size_t dspitch,
                                              const void *src, size_t spitch, size_t sspitch,
                                              size_t width, size_t height, size_t depth) {
  memCopyRegion(lzQueue, dst, dpitch, dspitch, src, spitch, sspitch, width, height, depth);
  return ExecuteAsync(lzQueue);
}

// Execute HipLZ memory fill command

void LZCommandList::memFill(LZQueue* lzQueue, void *dst, size_t size, const void *pattern, size_t pattern_size) {
  ze_result_t status = zeCommandListAppendMemoryFill(hCommandList, dst, pattern, pattern_size, size, nullptr, 0, nullptr);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendMemoryFill FAILED with return code ", status);
}

bool LZCommandList::ExecuteMemFill(LZQueue* lzQueue, void *dst, size_t size, const void *pattern, size_t pattern_size) {
  memFill(lzQueue, dst, size, pattern, pattern_size);
  return Execute(lzQueue);
}

bool LZCommandList::ExecuteMemFillAsync(LZQueue* lzQueue, void *dst, size_t size, const void *pattern, size_t pattern_size) {
  memFill(lzQueue, dst, size, pattern, pattern_size);
  return ExecuteAsync(lzQueue);
}

// Execute HipLZ write global timestamp
bool LZCommandList::ExecuteWriteGlobalTimeStamp(LZQueue* lzQueue, uint64_t *timestamp) {
  ze_result_t status = zeCommandListAppendWriteGlobalTimestamp(hCommandList, (uint64_t*)(shared_buf), nullptr, 0, nullptr);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendWriteGlobalTimestamp FAILED with return code ", status);
  bool res = Execute(lzQueue);
  if (res)
    *timestamp = *(uint64_t*)(shared_buf);
  return res;
}

bool LZCommandList::ExecuteWriteGlobalTimeStampAsync(LZQueue* lzQueue, uint64_t *timestamp, LZEvent *event) {
  ze_result_t status = zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr);
  LZ_PROCESS_ERROR_MSG("HipLZ  zeCommandListAppendBarrier FAILED with return code ", status);
  status = zeCommandListAppendWriteGlobalTimestamp(hCommandList, (uint64_t *)(shared_buf), nullptr, 0, nullptr);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendWriteGlobalTimestamp FAILED with return code ", status);
  status = zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr);
  LZ_PROCESS_ERROR_MSG("HipLZ  zeCommandListAppendBarrier FAILED with return code ", status);
  status = zeCommandListAppendMemoryCopy(hCommandList, timestamp, shared_buf, sizeof(uint64_t), event->GetEventHandle(), 0, nullptr);
  LZ_PROCESS_ERROR_MSG("HipLZ  zeCommandListAppendMemoryCopy FAILED with return code ", status);
  return ExecuteAsync(lzQueue);
}

// Execute HipLZ memory prefetch

void LZCommandList::memPrefetch(LZQueue* lzQueue, const void* ptr, size_t size) {
  ze_result_t status = zeCommandListAppendMemoryPrefetch(hCommandList, ptr, size);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendMemoryPrefetch FAILED with return code ", status);
}

bool LZCommandList::ExecuteMemPrefetch(LZQueue* lzQueue, const void* ptr, size_t size) {
  memPrefetch(lzQueue, ptr, size);
  return Execute(lzQueue);
}

bool LZCommandList::ExecuteMemPrefetchAsync(LZQueue* lzQueue, const void* ptr, size_t size) {
  memPrefetch(lzQueue, ptr, size);
  return ExecuteAsync(lzQueue);
}

// Execute HipLZ memory advise

void LZCommandList::memAdvise(LZQueue* lzQueue, const void* ptr, size_t count, hipMemoryAdvise advice) {
  ze_memory_advice_t zeAdvise = ZE_MEMORY_ADVICE_FORCE_UINT32;
  switch (advice) {
  case hipMemAdviseSetReadMostly:
    zeAdvise = ZE_MEMORY_ADVICE_SET_READ_MOSTLY;
  case hipMemAdviseUnsetReadMostly:
    zeAdvise = ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY;
  case hipMemAdviseSetPreferredLocation:
    zeAdvise = ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION;
  case hipMemAdviseUnsetPreferredLocation:
    zeAdvise = ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION;
  case hipMemAdviseSetAccessedBy:
    zeAdvise = ZE_MEMORY_ADVICE_BIAS_CACHED;
  case hipMemAdviseUnsetAccessedBy:
    zeAdvise = ZE_MEMORY_ADVICE_BIAS_UNCACHED;
  default:
    zeAdvise = ZE_MEMORY_ADVICE_FORCE_UINT32;
  }

  ze_result_t status = zeCommandListAppendMemAdvise(hCommandList,
						    lzQueue->GetContext()->GetDevice()->GetDeviceHandle(),
						    (void* )ptr, count, zeAdvise);

  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendMemAdvise FAILED with return code ", status);
}

bool LZCommandList::ExecuteMemAdvise(LZQueue* lzQueue, const void* ptr, size_t count, hipMemoryAdvise advice) {
  memAdvise(lzQueue, ptr, count, advice);
  return Execute(lzQueue);
}

bool LZCommandList::ExecuteMemAdviseAsync(LZQueue* lzQueue, const void* ptr, size_t count, hipMemoryAdvise advice) {
  memAdvise(lzQueue, ptr, count, advice);
  return ExecuteAsync(lzQueue);
}

// Synchronize host with device kernel execution

bool LZCommandList::finish() {
  HIP_PROCESS_ERROR_MSG("HipLZ does not support LZCommandList::finish! ", hipErrorNotSupported);
  return true;
}

bool LZStdCommandList::finish() {
  // Do nothing here
  return true;
}

// Synchronize host with device kernel execution
bool LZImmCommandList::finish() {
  ze_result_t status = zeCommandListAppendBarrier(hCommandList, finishEvent, 0, nullptr);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendBarrier FAILED with return code ", status);
  status = zeEventHostSynchronize(finishEvent, UINT64_MAX);
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventHostSynchronize FAILED with return code ", status);
  status = zeEventHostReset(finishEvent);
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventHostReset FAILED with return code ", status);

  return true;
 }

bool LZCommandList::Execute(LZQueue* lzQueue) {
  HIP_PROCESS_ERROR_MSG("HipLZ does not support LZCommandList::Execute! ", hipErrorNotSupported);
  return true;
}

// Execute HipLZ command list in standard command list
bool LZStdCommandList::Execute(LZQueue* lzQueue) {
  // Finished appending commands (typically done on another thread)
  ze_result_t status = zeCommandListClose(hCommandList);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListClose FAILED with return code ", status);

  logDebug("LZ KERNEL EXECUTION via calling zeCommandListClose {} ", status);

  // Execute command list in command queue
  status = zeCommandQueueExecuteCommandLists(lzQueue->GetQueueHandle(), 1, &hCommandList, nullptr);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandQueueExecuteCommandLists FAILED with return code ", status);
  logDebug("LZ KERNEL EXECUTION via calling zeCommandQueueExecuteCommandLists {} ", status);

  status = zeCommandQueueSynchronize(lzQueue->GetQueueHandle(), UINT64_MAX);
  logDebug("LZ KERNEL EXECUTION via calling zeCommandQueueSynchronize {} ", status);

  // Reset (recycle) command list for new commands
  status = zeCommandListReset(hCommandList);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListReset FAILED with return code ", status);

  logDebug("LZ KERNEL EXECUTION via calling zeCommandListReset {} ", status);

  return true;
}


// Execute HipLZ command list in immediate command list
bool LZImmCommandList::Execute(LZQueue* lzQueue) {
  // Synchronize host with device kernel execution
  return finish();
}

bool LZCommandList::ExecuteAsync(LZQueue* lzQueue) {
  HIP_PROCESS_ERROR_MSG("HipLZ does not support LZCommandList::ExecuteAsync! ", hipErrorNotSupported);
  return true;
}

// Execute HipLZ command list asynchronously in standard command list
bool LZStdCommandList::ExecuteAsync(LZQueue* lzQueue) {
  HIP_PROCESS_ERROR_MSG("HipLZ does not support LZStdCommandList::ExecuteAsync! ", hipErrorNotSupported);
  return true;
}

// Execute HipLZ command list asynchronously in immediate command list
bool LZImmCommandList::ExecuteAsync(LZQueue* lzQueue) {
  return true;
}

int LZExecItem::setupAllArgs(ClKernel *k) {
  LZKernel *kernel = (LZKernel *)k;
  OCLFuncInfo *FuncInfo = kernel->getFuncInfo();
  size_t NumLocals = 0;
  int LastArgIdx = -1;

  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    if (FuncInfo->ArgTypeInfo[i].space == OCLSpace::Local) {
      ++ NumLocals;
    }
  }
  // there can only be one dynamic shared mem variable, per cuda spec
  assert(NumLocals <= 1);

  // Argument processing for the new HIP launch API.
  if (ArgsPointer) {

    for (size_t i = 0, argIdx = 0; i < FuncInfo->ArgTypeInfo.size(); ++ i, ++ argIdx) {
      OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];

      // std::cout << "KERNEL ARG SETUP - arg type:  " << (int)ai.type << " and size " << ai.size
      //           << " ArgPointer " << (unsigned long)(ArgsPointer[i]) << " and "
      //           << sizeof(intptr_t) << std::endl;

      if (ai.type == OCLType::Image) {
        // This is the case for Image type, but the actual pointer is for HipTextureObject
        LZTextureObject* texObj = (LZTextureObject* )(* ((unsigned long *)(ArgsPointer[1])));

        // Set image part
        logDebug("setImageArg {} size {}\n", argIdx, ai.size);
        ze_result_t status = zeKernelSetArgumentValue(kernel->GetKernelHandle(),
                                                      argIdx, ai.size, &(texObj->image));
        if (status != ZE_RESULT_SUCCESS) {
          logDebug("zeKernelSetArgumentValue failed with error {}\n", status);
          return CL_INVALID_VALUE;
        }
        logDebug("LZ SET IMAGE ARGUMENT VALUE via calling zeKernelSetArgumentValue {} ", status);

        // Set sampler part
        argIdx ++;

        logDebug("setImageArg {} size {}\n", argIdx, ai.size);
        status = zeKernelSetArgumentValue(kernel->GetKernelHandle(), argIdx, ai.size,
                                          &(texObj->sampler));
        if (status != ZE_RESULT_SUCCESS) {
          logDebug("zeKernelSetArgumentValue failed with error {}\n", status);
          return CL_INVALID_VALUE;
        }
        logDebug("LZ SET SAMPLER ARGUMENT VALUE via calling zeKernelSetArgumentValue {} ", status);
      } else {
        logDebug("setArg {} size {}\n", argIdx, ai.size);
        ze_result_t status = zeKernelSetArgumentValue(kernel->GetKernelHandle(),
                                                      argIdx, ai.size, ArgsPointer[i]);
        if (status != ZE_RESULT_SUCCESS) {
          logDebug("zeKernelSetArgumentValue failed with error {}\n", status);
          return CL_INVALID_VALUE;
        }
        logDebug("LZ SET ARGUMENT VALUE via calling zeKernelSetArgumentValue {} ", status);
      }
    }
  } else {
    // Argument processing for the old HIP launch API.
    if ((OffsetsSizes.size() + NumLocals) != FuncInfo->ArgTypeInfo.size()) {
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
        if ((std::get<0>(OffsetsSizes[i]) == 0) ||
            (std::get<1>(OffsetsSizes[i]) == 0) ||
            ((std::get<0>(OffsetsSizes[i - 1]) +
              std::get<1>(OffsetsSizes[i - 1])) >
             std::get<0>(OffsetsSizes[i]))) {
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
      logDebug("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n",
               i, std::get<0>(OffsetsSizes[i]), std::get<1>(OffsetsSizes[i]),
               (unsigned)ai.type, (unsigned)ai.space, ai.size);

      if (ai.type == OCLType::Pointer) {
        // TODO: sync with ExecItem's solution
        assert(ai.size == sizeof(void *));
        assert(std::get<1>(OffsetsSizes[i]) == ai.size);
        size_t size = std::get<1>(OffsetsSizes[i]);
        size_t offs = std::get<0>(OffsetsSizes[i]);
        const void *value = (void *)(start + offs);
        logDebug("setArg SVM {} to {}\n", i, p);
        ze_result_t status =
            zeKernelSetArgumentValue(kernel->GetKernelHandle(), i, size, value);

        if (status != ZE_RESULT_SUCCESS) {
          logDebug("zeKernelSetArgumentValue failed with error {}\n", err);
          return CL_INVALID_VALUE;
        }

        logDebug(
            "LZ SET ARGUMENT VALUE via calling zeKernelSetArgumentValue {} ",
            status);
      } else {
        size_t size = std::get<1>(OffsetsSizes[i]);
        size_t offs = std::get<0>(OffsetsSizes[i]);
        const void *value = (void *)(start + offs);
        logDebug("setArg {} size {} offs {}\n", i, size, offs);
        ze_result_t status =
            zeKernelSetArgumentValue(kernel->GetKernelHandle(), i, size, value);

        if (status != ZE_RESULT_SUCCESS) {
          logDebug("zeKernelSetArgumentValue failed with error {}\n", err);
          return CL_INVALID_VALUE;
        }

        logDebug(
            "LZ SET ARGUMENT VALUE via calling zeKernelSetArgumentValue {} ",
            status);
      }
    }
  }

  // Setup the kernel argument's value related to dynamically sized share memory
  if (NumLocals == 1) {
    ze_result_t status = zeKernelSetArgumentValue(kernel->GetKernelHandle(),
                                                  FuncInfo->ArgTypeInfo.size() - 1,
                                                  SharedMem, nullptr);
    logDebug("LZ set dynamically sized share memory related argument via calling zeKernelSetArgumentValue {} ", status);
  }

  return CL_SUCCESS;
}

LZProgram::LZProgram(LZContext* lzContext, uint8_t* funcIL, size_t ilSize) : ClProgram() {

  size_t numWords = ilSize / 4;
  int32_t * binarydata = new int32_t[numWords + 1];
  std::memcpy(binarydata, funcIL, ilSize);

  // Extract kernel function information
  bool res = parseSPIR(binarydata, numWords, FuncInfos);
  delete[] binarydata;
  if (!res)
    HIP_PROCESS_ERROR_MSG("Hiplz SPIR-V parsing failed", hipErrorInitializationError);

  // Create module with global address aware
  std::string compilerOptions = " -cl-std=CL2.0 -cl-take-global-address -cl-match-sincospi";
  ze_module_desc_t moduleDesc = {
    ZE_STRUCTURE_TYPE_MODULE_DESC,
    nullptr,
    ZE_MODULE_FORMAT_IL_SPIRV,
    ilSize,
    funcIL,
    compilerOptions.c_str(),
    nullptr
  };
  ze_result_t status = zeModuleCreate(lzContext->GetContextHandle(),
                                      lzContext->GetDevice()->GetDeviceHandle(),
                                      &moduleDesc, &this->hModule, nullptr);
  LZ_PROCESS_ERROR_MSG("Hiplz zeModuleCreate FAILED with return code  ", status);

  logDebug("LZ CREATE MODULE via calling zeModuleCreate {} ", status);
}

LZProgram::~LZProgram() {
  zeModuleDestroy(this->hModule);
}

// Create Level-0 kernel
void LZProgram::CreateKernel(std::string &funcName) {
  if (this->kernels.find(funcName) != this->kernels.end())
    return;

  // Register kernel
  if (FuncInfos.find(funcName) == FuncInfos.end()) {
    for (auto funcInfo : FuncInfos) {
      std::string kernelName = funcInfo.first;
      if (LZKernel::IsEquvalentKernelName(funcName, kernelName,
                                          "ClTextureObject", "hipTextureObject_s")) {
	if (this->kernels.find(kernelName) != this->kernels.end())
	  this->kernels[kernelName] = new LZKernel(this, kernelName, FuncInfos[kernelName]);
	this->kernels[funcName] = new LZKernel(this, kernelName, FuncInfos[kernelName]);

	return;
      }
    }

    HIP_PROCESS_ERROR_MSG("HipLZ could not find function information ", hipErrorInitializationError);
  }

  // Create kernel
  this->kernels[funcName] = new LZKernel(this, funcName, FuncInfos[funcName]);
}

// Get hte global pointer related information
bool LZProgram::getSymbolAddressSize(const char *name, hipDeviceptr_t *dptr, size_t* bytes) {
  size_t varSize = 0;
  ze_result_t status = zeModuleGetGlobalPointer(this->hModule, name, &varSize, dptr);
  if (status != ZE_RESULT_SUCCESS) {
    std::string varName = (const char *)name;
    std::cout << "No global variable found: " << varName << "  " << status << "   ";
    if (status == ZE_RESULT_ERROR_DEVICE_LOST)
      std::cout << "ZE_RESULT_ERROR_DEVICE_LOST";
    else if (status == ZE_RESULT_ERROR_UNINITIALIZED)
      std::cout << "ZE_RESULT_ERROR_UNINITIALIZED";
    else if (status == ZE_RESULT_ERROR_INVALID_NULL_HANDLE)
      std::cout << "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
    else if (status == ZE_RESULT_ERROR_INVALID_NULL_POINTER)
      std::cout << "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
    else if (status == ZE_RESULT_ERROR_INVALID_GLOBAL_NAME)
      std::cout << "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";

    std::cout << std::endl;

    return false;
  } else {
    // std::string varName = name;
    // std::cout << "getSymbolAddressSize  --  global variable: " << varName << " bytes: " << varSize << "  ptr: " << * dptr << std::endl;
  }

  if (varSize != 0)
    * bytes = varSize;

  return true;
}

// Create kernel via given name 
hipFunction_t LZProgram::getKernel(std::string &name) {
  if (this->kernels.find(name) == this->kernels.end()) {
    // Create kernel here, if the LZProgram::createProgram procedure did not create kernel
    this->CreateKernel(name);
    return this->kernels[name];
  } else
    return this->kernels[name];

  // HIP_PROCESS_ERROR_MSG("HipLZ could not find kernel function in LZProgram ", hipErrorInitializationError);
}

LZKernel::LZKernel(LZProgram* lzModule, std::string funcName, OCLFuncInfo* funcInfo) : ClKernel(funcName, funcInfo) {
  // Create kernel
  ze_kernel_desc_t kernelDesc = {
    ZE_STRUCTURE_TYPE_KERNEL_DESC,
    nullptr,
    0, // flags
    funcName.c_str()
  };
  ze_kernel_handle_t hKernel;
  ze_result_t status = zeKernelCreate(lzModule->GetModuleHandle(), &kernelDesc, &this->hKernel);
  LZ_PROCESS_ERROR_MSG("HipLZ zeKernelCreate FAILED with return code ", status);

  logDebug("LZ KERNEL CREATION via calling zeKernelCreate {} ", status);
}

LZKernel::~LZKernel() {
  zeKernelDestroy(this->hKernel);
}

// Compare if two kernel functions' names are equvalent, including their arguments' types 
bool LZKernel::IsEquvalentKernelName(std::string funcName, std::string targetFuncName,
				     std::string typeName, std::string targetTypeName) {
  int status = 0;
  const char* real_name = abi::__cxa_demangle(funcName.c_str(), 0, 0, &status);
  if (status)
    return false;
  std::string realName = real_name;

  const char* real_target_name = abi::__cxa_demangle(targetFuncName.c_str(), 0, 0, &status);
  if (status)
    return false;
  std::string realTargetName = real_target_name;
 
  std::size_t found = realName.find(typeName);
  if (found != std::string::npos) {
    // Replae type name with target type name
    realName.replace(found, typeName.length(), targetTypeName);

    return realName == realTargetName;
  }

  return false;
}

// Create HipLZ event
LZEvent::LZEvent(LZContext* c, unsigned flags) : ClEvent(flags)  {
  this->cont = c;
  this->hEvent = nullptr;
  this->hEventPool = nullptr;

  ze_event_pool_desc_t eventPoolDesc = {
    ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
    nullptr,
    ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // event in pool are visible to Host
    1 // count
  };
  ze_result_t status = zeEventPoolCreate(c->GetContextHandle(), &eventPoolDesc, 0, nullptr, &hEventPool);
  LZ_PROCESS_ERROR_MSG("HipLZ event pool creation fail! ", status);

  ze_event_desc_t eventDesc = {
    ZE_STRUCTURE_TYPE_EVENT_DESC,
    nullptr,
    0, // index
    ZE_EVENT_SCOPE_FLAG_HOST, // ensure memory/cache coherency required on signal
    ZE_EVENT_SCOPE_FLAG_HOST  // ensure memory coherency across device and Host after event completes
  };

  status = zeEventCreate(hEventPool, &eventDesc, &hEvent);
  LZ_PROCESS_ERROR_MSG("HipLZ event creation fail! ", status);
}

// Get the finish time of the event associated operation
uint64_t LZEvent::getFinishTime() {
  std::lock_guard<std::mutex> Lock(EventMutex);

  return getTimeStamp();
}

// Record event into stream
bool LZEvent::recordStream(hipStream_t S) {
  std::lock_guard<std::mutex> Lock(EventMutex);

  if (Status == EVENT_STATUS_RECORDED) {
    ze_result_t status = zeEventHostReset(this->hEvent);
    LZ_PROCESS_ERROR_MSG("HipLZ zeEventHostReset FAILED with return code ", status);
  }

  Stream = S;
  ((LZQueue* )Stream)->GetDefaultCmdList()->ExecuteWriteGlobalTimeStampAsync((LZQueue* )Stream, &timestamp, this);
  Status = EVENT_STATUS_RECORDING;

  return true;
}

bool LZEvent::updateFinishStatus() {
  std::lock_guard<std::mutex> Lock(EventMutex);
  if (Status != EVENT_STATUS_RECORDING)
    return false;

  ze_result_t status = zeEventQueryStatus(this->hEvent);
  LZ_PROCESS_ERROR_MSG("HipLZ zeEventQueryStatus FAILED with return code ", status);
  if (status == ZE_RESULT_SUCCESS)
    Status = EVENT_STATUS_RECORDED;

  return true;
}

bool LZEvent::wait() {
  std::lock_guard<std::mutex> Lock(EventMutex);
  if (Status != EVENT_STATUS_RECORDING)
    return false;

  ze_result_t status = zeEventHostSynchronize(this->hEvent, UINT64_MAX);
  LZ_PROCESS_ERROR_MSG("HipLZ event synchronization error! ", status);

  Status = EVENT_STATUS_RECORDED;
  return true;
}

LZImage::LZImage(LZContext* lzContext, hipResourceDesc* resDesc, hipTextureDesc* texDesc) {
  this->lzContext = lzContext;

  // TODO: parse the resource and texture descriptor
  ze_image_format_t format = {
    ZE_IMAGE_FORMAT_LAYOUT_32,
    ZE_IMAGE_FORMAT_TYPE_FLOAT,
    ZE_IMAGE_FORMAT_SWIZZLE_R,
    ZE_IMAGE_FORMAT_SWIZZLE_0,
    ZE_IMAGE_FORMAT_SWIZZLE_0,
    ZE_IMAGE_FORMAT_SWIZZLE_1
  };

  ze_image_desc_t imageDesc = {
    ZE_STRUCTURE_TYPE_IMAGE_DESC,
    nullptr,
    0, // read-only
    ZE_IMAGE_TYPE_2D,
    format,
    128, 128, 0, 0, 0
  };

  ze_result_t status = zeImageCreate(lzContext->GetContextHandle(),
				      lzContext->GetDevice()->GetDeviceHandle(),
				      &imageDesc, &this->hImage);
  LZ_PROCESS_ERROR_MSG("HipLZ zeImageCreate FAILED with return code ", status);
}

// Update data to image
bool LZImage::upload(hipStream_t stream, void* srcptr) {
  ze_result_t status = zeCommandListAppendImageCopyFromMemory(((LZQueue* )stream)->GetDefaultCmdList()->GetCommandListHandle(),
							      hImage, srcptr, nullptr, nullptr, 0,
							      nullptr);
  LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListAppendImageCopyFromMemory with return code ", status);

  return true;
}

// The factory function for creating the LZ texture object 
LZTextureObject* LZTextureObject::CreateTextureObject(LZContext* lzCtx,
                                                      const hipResourceDesc* pResDesc,
                                                      const hipTextureDesc* pTexDesc,
                                                      const struct hipResourceViewDesc* pResViewDesc) {
  // Create the LZ Texture class here
  LZTextureObject* texObj = new	LZTextureObject();
  
  ze_image_handle_t   imageHandle;
  ze_sampler_handle_t samplerHandle;
  if (CreateImage(lzCtx, pResDesc, pTexDesc, pResViewDesc, &imageHandle)
      && CreateSampler(lzCtx, pResDesc, pTexDesc, pResViewDesc, &samplerHandle)) {
    texObj->image   = (intptr_t)imageHandle;
    texObj->sampler = (intptr_t)samplerHandle;

    // Check if need to copy data in
    if (pResDesc->res.array.array != nullptr) {
      hipArray* hipArr = pResDesc->res.array.array;
      lzCtx->memCopyToTexture(texObj, (unsigned char *)hipArr->data);
    }
  } else
    return nullptr;
  
  return texObj;
}

// Destroy HIP texture object
bool LZTextureObject::DestroyTextureObject(LZTextureObject* texObj) {
  if (texObj == nullptr)
    return false;

  ze_image_handle_t   imageHandle =  (ze_image_handle_t)texObj->image;
  ze_sampler_handle_t samplerHandle = (ze_sampler_handle_t)texObj->sampler;
  if (DestroyImage(imageHandle) && DestroySampler(samplerHandle)) {
    delete texObj;

    return true;
  } else
    return false;
    
}

// The factory function for create the LZ image object
bool LZTextureObject::CreateImage(LZContext* lzCtx,
                                  const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const struct hipResourceViewDesc* pResViewDesc,
                                  ze_image_handle_t* handle) {
  if (pResDesc->resType != hipResourceTypeArray) {
    LZ_PROCESS_ERROR_MSG("HipLZ only support hipArray as image storage ", ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
  }

  hipArray* hipArr = pResDesc->res.array.array;
  hipChannelFormatDesc channelDesc = hipArr->desc;

  ze_image_format_layout_t format_layout = ZE_IMAGE_FORMAT_LAYOUT_32;
  if (channelDesc.x == 8) {
    format_layout = ZE_IMAGE_FORMAT_LAYOUT_8;
  } else if (channelDesc.x == 16) {
    format_layout = ZE_IMAGE_FORMAT_LAYOUT_16;
  } else if (channelDesc.x == 32) {
    format_layout = ZE_IMAGE_FORMAT_LAYOUT_32;
  } else {
    LZ_PROCESS_ERROR_MSG("hipChannelFormatDesc value is out of the scope ", ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
  }

  ze_image_format_type_t format_type = ZE_IMAGE_FORMAT_TYPE_FLOAT;
  if (channelDesc.f == hipChannelFormatKindSigned) {
    format_type = ZE_IMAGE_FORMAT_TYPE_SINT;
  } else if (channelDesc.f == hipChannelFormatKindUnsigned) {
    format_type = ZE_IMAGE_FORMAT_TYPE_UINT;
  } else if (channelDesc.f == hipChannelFormatKindFloat) {
    format_type = ZE_IMAGE_FORMAT_TYPE_FLOAT;
  } else if (channelDesc.f == hipChannelFormatKindNone) {
    format_type = ZE_IMAGE_FORMAT_TYPE_FORCE_UINT32;
  } else {
    LZ_PROCESS_ERROR_MSG("hipChannelFormatKind value is out of scope ", ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
  }

  ze_image_format_t format = {
    format_layout,
    format_type,
    ZE_IMAGE_FORMAT_SWIZZLE_R,
    ZE_IMAGE_FORMAT_SWIZZLE_0,
    ZE_IMAGE_FORMAT_SWIZZLE_0,
    ZE_IMAGE_FORMAT_SWIZZLE_1
  };

  ze_image_type_t image_type = ZE_IMAGE_TYPE_2D;

  ze_image_desc_t imageDesc = {
    ZE_STRUCTURE_TYPE_IMAGE_DESC,
    nullptr,
    0, // read-only  
    image_type,
    format,
    // 128, 128, 0, 0, 0
    hipArr->width, hipArr->height, 0, 0, 0
  };

  // Create LZ image handle
  ze_result_t status = zeImageCreate(lzCtx->GetContextHandle(),
				     lzCtx->GetDevice()->GetDeviceHandle(),
				     &imageDesc, handle);
  LZ_PROCESS_ERROR_MSG("HipLZ zeImageCreate FAILED with return code ", status);

  return true;
}

// Destroy the LZ image object
bool LZTextureObject::DestroyImage(ze_image_handle_t handle) {
  // Destroy LZ image handle   
  ze_result_t status = zeImageDestroy(handle);
  LZ_PROCESS_ERROR_MSG("HipLZ zeImageDestroy FAILED with return code ", status);

  return true;
}

// The factory function for create the LZ sampler object
bool LZTextureObject::CreateSampler(LZContext* lzCtx,
                                    const hipResourceDesc* pResDesc,
                                    const hipTextureDesc* pTexDesc,
                                    const struct hipResourceViewDesc* pResViewDesc,
                                    ze_sampler_handle_t* handle) {
  // Identify the address mode 
  ze_sampler_address_mode_t addressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
  if (pTexDesc->addressMode[0] == hipAddressModeWrap)
    addressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
  else if (pTexDesc->addressMode[0] == hipAddressModeClamp)
    addressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP;
  else if (pTexDesc->addressMode[0] == hipAddressModeMirror)
    addressMode = ZE_SAMPLER_ADDRESS_MODE_MIRROR;
  else if (pTexDesc->addressMode[0] == hipAddressModeBorder)
    addressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;

  // Identify the filter mode
  ze_sampler_filter_mode_t filterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
  if (pTexDesc->filterMode == hipFilterModePoint)
    filterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;  
  else if (pTexDesc->filterMode == hipFilterModeLinear)
    filterMode = ZE_SAMPLER_FILTER_MODE_LINEAR;

  // Identify the normalization
  ze_bool_t isNormalized = 0;
  if (pTexDesc->normalizedCoords == 0)
    isNormalized = 0;
  else
    isNormalized = 1;

  ze_sampler_desc_t samplerDesc = {
    ZE_STRUCTURE_TYPE_SAMPLER_DESC,
    nullptr,
    addressMode,
    filterMode,
    isNormalized
  };

  // Create LZ samler handle
  ze_result_t status = zeSamplerCreate(lzCtx->GetContextHandle(),
				       lzCtx->GetDevice()->GetDeviceHandle(),
				       &samplerDesc, handle);
  LZ_PROCESS_ERROR_MSG("HipLZ zeSamplerCreate FAILED with return code ", status);

  return true;
}

// Destroy the LZ sampler object
bool LZTextureObject::DestroySampler(ze_sampler_handle_t handle) {
  // Destroy LZ samler
  ze_result_t status = zeSamplerDestroy(handle);
  LZ_PROCESS_ERROR_MSG("HipLZ zeSamplerDestroy FAILED with return code ", status);

  return true;
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

  // Register fat binary modules
  for (std::string* module : LZDriver::FatBinModules) {
    for (size_t driverId = 0; driverId < NumLZDrivers; ++ driverId) {
      LZDriver::HipLZDriverById(driverId).registerModule(module);
    }
  }

  // Register functions
  for (auto fi : LZDriver::RegFunctions) {
    std::string * module      = std::get<0>(fi);
    const void * hostFunction = std::get<1>(fi);
    const char * deviceName   = std::get<2>(fi);
    for (size_t driverId = 0; driverId < NumLZDrivers; ++ driverId) {
      if (LZDriver::HipLZDriverById(driverId).registerFunction(module, hostFunction, deviceName)) {
	logDebug("__hipRegisterFunction: HipLZ kernel {} found\n", deviceName);
      } else {
	logCritical("__hipRegisterFunction can NOT find HipLZ kernel: {} \n", deviceName);
	std::abort();
      }
    }
  }

  // Register globale variables
  for (auto vi : LZDriver::GlobalVars) {
    std::string * module    = std::get<0>(vi);
    char * hostVar          = std::get<1>(vi);
    const char * deviceName = std::get<2>(vi);
    int size                = std::get<3>(vi);
    std::string devName = deviceName;
    for (size_t driverId = 0; driverId < NumLZDrivers; ++ driverId) {
      if (LZDriver::HipLZDriverById(driverId).registerVar(module, hostVar, deviceName, size)) {
	logDebug("__hipRegisterVar: variable {} found\n", deviceName);
      } else {
	logError("__hipRegisterVar could not find: {}\n", deviceName);
      }
    }
  }
}

void InitializeHipLZ() {
  // This is to consider the case that InitializeHipLZFromOutside was invoked
  // TODO: consider for lock protection
  if (HipLZDrivers.size() != 0)
    return;

  static std::once_flag hipLZInitialized;
  std::call_once(hipLZInitialized, InitializeHipLZCallOnce);
}

void InitializeHipLZFromOutside(ze_driver_handle_t hDriver,
                                ze_device_handle_t hDevice,
				ze_context_handle_t hContext,
                                ze_command_queue_handle_t hQueue) {
  // Initialize HipLZ device drivers and relevant devices
  LZDriver::InitDriver(HipLZDrivers, ZE_DEVICE_TYPE_GPU, hDriver, hDevice, hContext, hQueue);

  // Register fat binary modules
  for (std::string* module : LZDriver::FatBinModules) {
    for (size_t driverId = 0; driverId < NumLZDrivers; ++ driverId) {
      LZDriver::HipLZDriverById(driverId).registerModule(module);
    }
  }

  // Register functions
  for (auto fi : LZDriver::RegFunctions) {
    std::string * module      = std::get<0>(fi);
    const void * hostFunction = std::get<1>(fi);
    const char * deviceName   = std::get<2>(fi);
    for (size_t driverId = 0; driverId < NumLZDrivers; ++ driverId) {
      if (LZDriver::HipLZDriverById(driverId).registerFunction(module, hostFunction, deviceName)) {
        logDebug("__hipRegisterFunction: HipLZ kernel {} found\n", deviceName);
      } else {
        logCritical("__hipRegisterFunction can NOT find HipLZ kernel: {} \n", deviceName);
        std::abort();
      }
    }
  }

  // Register globale variables
  for (auto vi : LZDriver::GlobalVars) {
    std::string * module    = std::get<0>(vi);
    char * hostVar          = std::get<1>(vi);
    const char * deviceName = std::get<2>(vi);
    int size                = std::get<3>(vi);
    std::string devName = deviceName;
    for (size_t driverId = 0; driverId < NumLZDrivers; ++ driverId) {
      if (LZDriver::HipLZDriverById(driverId).registerVar(module, hostVar, deviceName, size)) {
        logDebug("__hipRegisterVar: variable {} found\n", deviceName);
      } else {
        logError("__hipRegisterVar could not find: {}\n", deviceName);
      }
    }
  }
}

LZDevice &HipLZDeviceById(int deviceId) {
  return *HipLZDevices.at(deviceId);
}

/***********************************************************************/


#ifdef __GNUC__
#pragma GCC visibility pop
#endif
