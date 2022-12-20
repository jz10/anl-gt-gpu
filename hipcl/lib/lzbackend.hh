#include <list>
#include <map>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <vector>
#include <iostream>

#include <pthread.h>

#include "backend.hh"

#include "ze_api.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

hipError_t lzConvertResult(ze_result_t status);
const char * lzResultToString(ze_result_t status);

#define LZ_LOG_ERROR(msg, status) logError("{} ({}) in {}:{}:{}\n", msg, lzResultToString(status), __FILE__, __LINE__, __func__)

#define LZ_PROCESS_ERROR_MSG(msg, status) do {\
  if (status != ZE_RESULT_SUCCESS && status != ZE_RESULT_NOT_READY) { \
    LZ_LOG_ERROR(msg, status); \
    throw status; \
  } \
} while(0)

#define LZ_PROCESS_ERROR(status) LZ_PROCESS_ERROR_MSG("Level Zero Error", status)

#define LZ_RETURN_ERROR_MSG(msg, status) do {\
  if (status != ZE_RESULT_SUCCESS && status != ZE_RESULT_NOT_READY) { \
    LZ_LOG_ERROR(msg, status); \
    return lzConvertResult(status); \
  } \
} while(0)


/*#define LZ_RETURN_ERROR(status) LZ_RETURN_ERROR_MSG("Level Zero Error", status)

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
*/

#define LZ_TRY try {
#define LZ_CATCH } catch (ze_result_t _status) { \
  RETURN(lzConvertResult(_status)); \
} catch (hipError_t _status) { \
  RETURN(_status); \
}
#define LZ_CATCH_NO_SET } catch (ze_result_t _status) { \
  return lzConvertResult(_status); \
} catch (hipError_t _status) { \
  return _status; \
}

class InvalidLevel0Initialization : public std::out_of_range {
  using std::out_of_range::out_of_range;
};

struct hipStreamCallbackData {
  hipStream_t Stream;
  ze_event_pool_handle_t eventPool;
  ze_event_handle_t waitEvent;
  ze_event_handle_t signalEvent;
  ze_event_handle_t waitEvent2;
  hipError_t Status;
  void *UserData;
  hipStreamCallback_t Callback;
};

struct hipContextSyncData {
  ze_event_pool_handle_t eventPool;
  ze_event_handle_t waitEvent;
  std::vector<ze_event_handle_t> signaledEvents;
};

class LZExecItem;

class LZKernel;

class LZExecItem : public ExecItem {
public:
  LZExecItem(dim3 grid, dim3 block, size_t shared, hipStream_t stream)
    : ArgsBuf(0), ExecItem(grid, block, shared, stream) {}

  virtual ~LZExecItem() {
    if (ArgsBuf)
      delete ArgsBuf;
  }

  // Setup all arguments for HipLZ kernel funciton invocation
  virtual int setupAllArgs(ClKernel *kernel);

  // Get data buffer
  void** ExtractArgsPointer(); 

protected:
  void** ArgsBuf;
};

class LZContext;

class LZDriver;

// HipLZ device object that
class LZDevice {
protected:
  enum PeerAccessState{ Accessible, UnAccessible, Accessible_Disabled, Uninitialized };

  // Synchronization mutex
  std::mutex DeviceMutex;
  // The integer ID of current device
  hipDevice_t Index;//deviceId;
  // The current device's properties
  hipDeviceProp_t Properties;
  // The names of modules
  std::vector<std::string *> Modules;
  // The map between host function pointer and module name
  std::map<const void *, std::string *> HostPtrToModuleMap;
  // The map between host function pointer nd name
  std::map<const void *, std::string> HostPtrToNameMap;
  // The Hip attribute map
  std::map<hipDeviceAttribute_t, int> Attributes;
  // The default context associated with this device
  LZContext* PrimaryContext;
  // The size of total used memory
  size_t TotalUsedMem;

  // The device handler
  ze_device_handle_t hDevice;

  // The driver object
  LZDriver* driver;

  // The device memory properties
  ze_device_memory_properties_t deviceMemoryProps;

  // The device compute properties
  ze_device_compute_properties_t deviceComputeProps;

  // The device cache properties
  ze_device_cache_properties_t deviceCacheProps;

  // The device module properties
  ze_device_module_properties_t deviceModuleProps;

  // The handle of device properties
  ze_device_properties_t deviceProps;

  // The command queue ordinal
  uint32_t cmdQueueGraphOrdinal;

  // The device peer access states
  std::map<int, PeerAccessState> peerAccessTable;

public:
  LZDevice(hipDevice_t id,  ze_device_handle_t hDevice, LZDriver* driver);
  LZDevice(hipDevice_t id,  ze_device_handle_t hDevice, LZDriver* driver, ze_context_handle_t hContext,
	   ze_command_queue_handle_t hQueue);

  // Get device properties
  ze_device_properties_t* GetDeviceProps() { return &(this->deviceProps); };

  // Get device handle
  ze_device_handle_t GetDeviceHandle() { return this->hDevice; };

  // Get current device driver handle
  ze_driver_handle_t GetDriverHandle();

  // Get current device's integer ID
  hipDevice_t getHipDeviceT() { return Index; }

  // Check if the device can access another device
  static hipError_t CanAccessPeer(LZDevice& device, LZDevice& peerDevice, int* canAccessPeer);

  // Check if the curren device can be accessed by another device
  hipError_t CanBeAccessed(LZDevice& srcDevice, int* canAccessPeer);

  // Enable/Disable the peer access from given devince
  hipError_t SetAccess(LZDevice& srcDevice, int flags, bool canAccessPeer);

  // Check if the current device has same PCI bus ID as the one given by input
  bool HasPCIBusId(int pciDomainID, int pciBusID, int pciDeviceID);

  // Register HipLZ module which is presented as IL
  void registerModule(std::string* module);

  // Regsiter HipLZ module, kernel function name with host function which is a wrapper
  bool registerFunction(std::string *module, const void *HostFunction, const char *FunctionName);

  // Register global variable
  bool registerVar(std::string *module, const void *HostVar, const char *VarName);
  bool registerVar(std::string *module, const void *HostVar, const char *VarName, size_t size);

  // Get host function pointer's corresponding name
  std::string GetHostFunctionName(const void* HostFunction);

  // Get primary context
  LZContext* getPrimaryCtx() { return PrimaryContext; };

  // Get the size of global memory
  size_t getGlobalMemSize() const { return this->deviceMemoryProps.totalSize; }

  // Get the size of used global memory
  size_t getUsedGlobalMem() const { return TotalUsedMem; }

  // Reserver memory
  bool reserveMem(size_t bytes) { return true; };

  // Release memory
  bool releaseMem(size_t bytes) { return true; };

  // Reset current device
  void reset();

  // Copy device properties to given property data structure
  void copyProperties(hipDeviceProp_t *prop);

  // Get Hip attribute from attribute enum ID
  int getAttr(int *pi, hipDeviceAttribute_t attr);

  // Get the max allocation size in MB
  size_t GetMaxAllocSize() {
    return this->deviceProps.maxMemAllocSize;
  }

  // Get HipLZ device name
  const char *getName() const { return Properties.name; };

  // Get command group ordinal
  uint32_t GetCmdQueueGroupOrdinal() { return this->cmdQueueGraphOrdinal; };

protected:
  // Retrieve device properties related data
  void retrieveDeviceProperties();

  // Setup HipLZ device properties
  void setupProperties(int index);

  bool retrieveCmdQueueGroupOrdinal(uint32_t& ordinal);
};

class LZProgram;

class LZKernel : public ClKernel {
protected:
  // HipLZ kernel handle
  ze_kernel_handle_t hKernel;

public:
  LZKernel(LZProgram* lzModule, std::string funcName, OCLFuncInfo* funcInfo);
  ~LZKernel();

  ze_kernel_handle_t GetKernelHandle() { return this->hKernel; }

  // Compare if two kernel functions' names are equvalent, including their arguments' types 
  static bool IsEquvalentKernelName(std::string funcName, std::string targetFuncName,
				    std::string typeName, std::string targeTypeName);
};

class LZProgram : public ClProgram {
protected:
  // HipLZ module handle
  ze_module_handle_t hModule;

public:
  LZProgram(LZContext* lzContext, uint8_t* funcIL, size_t ilSize);
  ~LZProgram();

  // Get kernel via given name
  virtual hipFunction_t getKernel(std::string &name);
  
  // Get HipLZ module handle
  ze_module_handle_t GetModuleHandle() { return this->hModule; }

  // Create HipLZ kernel via function name
  virtual void CreateKernel(std::string &funcName);

  // Check if support symbol address
  virtual bool symbolSupported() {
    return true;
  }

  // Get the pointer address and size information via gievn symbol's name
  virtual bool getSymbolAddressSize(const char *name, hipDeviceptr_t *dptr, size_t* bytes);
};

class LZEvent : public ClEvent {
protected:
  // The associated HipLZ context
  LZContext* cont;

  // The handler of HipLZ event_pool and event
  ze_event_handle_t hEvent;
  ze_event_pool_handle_t hEventPool;

  // The timestamp value
  uint64_t timestamp;

public:
  LZEvent(LZContext* c, unsigned flags);

  virtual ~LZEvent() {
    if (hEvent)
      zeEventDestroy(hEvent);
    if (hEventPool)
      zeEventPoolDestroy(hEventPool);
  }

  virtual uint64_t getFinishTime();

  // Record the event to stream
  virtual bool recordStream(hipStream_t S);

  // Update event's finish status
  virtual bool updateFinishStatus();

  // Wait on event get finished
  virtual bool wait();

  // Get current event handler
  ze_event_handle_t GetEventHandle() { return this->hEvent; };

  // Record the time stamp
  void recordTimeStamp(uint64_t value) { this->timestamp = value; };

  // Get the time stamp
  uint64_t getTimeStamp() { return this->timestamp; };
};

class LZTextureObject;

class LZQueue;

class LZCommandList;

class LZGraph;

class LZContext : public ClContext {
protected:
  // Reference to HipLZ device
  LZDevice* lzDevice;

  // Map between IL binary to HipLZ module
  std::map<uint8_t* , LZProgram* > IL2Module;

  // HipLZ context handle
  ze_context_handle_t hContext;

  // The map between global variable name and its relevant HipLZ module, device poitner and size information
  std::map<std::string, std::tuple<LZProgram *, hipDeviceptr_t, size_t>> GlobalVarsMap;

  // Monitor thread to release events for stream synchronization
  std::mutex syncDatasMutex;
  pthread_t monitorThreadId;
  bool stopMonitor;

  LZQueue* captureQueue;

  // List of events to release
  std::list<hipContextSyncData> syncDatas;

public:
  LZContext(ClDevice* D, unsigned f) : ClContext(D, f), lzDevice(0), captureQueue(0) {
    monitorThreadId = 0;
    stopMonitor = false;
    if (CreateMonitor())
      logError("LZ CONTEXT sync event monitor could not be created");
  }
  LZContext(LZDevice* dev);
  LZContext(LZDevice* dev, ze_context_handle_t hContext, ze_command_queue_handle_t hQueue);
  ~LZContext();

  // Create SPIR-V module
  bool CreateModule(uint8_t* moduleIL, size_t ilSize, std::string funcName);

  // Get Level-0 handle for context
  ze_context_handle_t GetContextHandle() { return this->hContext; }

  // Get Level-0 device object
  LZDevice* GetDevice() { return this->lzDevice; };

  // Launch HipLZ kernel (old HIP launch API).
  virtual hipError_t launchHostFunc(const void* HostFunction);

  // Launch HipLZ kernel (new HIP launch API).
  virtual hipError_t launchHostFunc(const void *function_address, dim3 numBlocks,
                                    dim3 dimBlocks, void **args, size_t sharedMemBytes,
                                    hipStream_t stream);
  virtual hipError_t launchWithKernelParams(dim3 grid, dim3 block, size_t shared,
                                            hipStream_t stream, void **kernelParams,
                                            hipFunction_t kernel);
  virtual hipError_t launchWithExtraParams(dim3 grid, dim3 block,
                                           size_t shared, hipStream_t stream,
                                           void **extraParams,
                                           hipFunction_t kernel);

  // Memory allocation
  virtual void* allocate(size_t size);
  virtual void *allocate(size_t size, ClMemoryType memTy);
  virtual void* allocate(size_t size, size_t alignment, ClMemoryType memTy);

  // Memory free
  virtual bool free(void *p);

  // Get pointer info
  bool findPointerInfo(hipDeviceptr_t dptr, hipDeviceptr_t *pbase, size_t *psize);
  bool getPointerSize(void *ptr, size_t *size);

  // Memory copy
  virtual hipError_t memCopy(void *dst, const void *src, size_t sizeBytes, hipStream_t stream);
  virtual hipError_t memCopyAsync(void *dst, const void *src, size_t sizeBytes, hipStream_t stream);

  // Memory fill
  virtual hipError_t memFill(void *dst, size_t size, const void *pattern, size_t pattern_size, hipStream_t stream);
  virtual hipError_t memFillAsync(void *dst, size_t size, const void *pattern, size_t pattern_size, hipStream_t stream);

  // Memory copy 2D
  virtual hipError_t memCopy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
			       size_t width, size_t height, hipStream_t stream);
  virtual hipError_t memCopy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
			    size_t width, size_t height, hipStream_t stream);

  // Memory copy 3D
  virtual hipError_t memCopy3D(void *dst, size_t dpitch, size_t dspitch,
		       const void *src, size_t spitch, size_t sspitch,
		       size_t width, size_t height, size_t depth, hipStream_t stream);
  virtual hipError_t memCopy3DAsync(void *dst, size_t dpitch, size_t dspitch,
			    const void *src, size_t spitch, size_t sspitch,
			    size_t width, size_t height, size_t depth, hipStream_t stream);

  // Memory copy to texture object, i.e. image
  hipError_t memCopyToTexture(LZTextureObject* texObj, void* src, hipStream_t stream);

  hipError_t memCopyToTexture(LZTextureObject* texObj, void* src) {
    return memCopyToTexture(texObj, src, getDefaultQueue());
  }
  
  // Make meory prefetch
  virtual hipError_t memPrefetch(const void* ptr, size_t size, hipStream_t stream = 0);

  // Make the advise for the managed memory (i.e. unified shared memory)
  virtual hipError_t memAdvise(const void* ptr, size_t count, hipMemoryAdvise advice, hipStream_t stream = 0);

  // Create HipLZ event
  virtual hipEvent_t createEvent(unsigned flags);
  virtual hipError_t recordEvent(hipStream_t stream, hipEvent_t event);

  // Create stream/queue
  virtual bool createQueue(hipStream_t *stream, unsigned int Flags, int priority);
  virtual bool releaseQueue(hipStream_t stream);

  // Get the elapse between two events
  virtual hipError_t eventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop);

  // Reset current context
  virtual void reset();

  // Register global variable
  bool registerVar(std::string *module, const void *HostVar, const char *VarName);
  bool registerVar(std::string *module, const void *HostVar, const char *VarName, size_t size);

  // Get the address and size for the given symbol's name
  virtual bool getSymbolAddressSize(const char *name, hipDeviceptr_t *dptr, size_t *bytes);

  // Create Level-0 image object
  LZImage* createImage(hipResourceDesc* resDesc, hipTextureDesc* texDesc);

  // Create HIP texture object
  virtual hipTextureObject_t createTextureObject(const hipResourceDesc* pResDesc,
                                                 const hipTextureDesc* pTexDesc,
                                                 const struct hipResourceViewDesc* pResViewDesc);

  // Destroy HIP texture object
  virtual bool destroyTextureObject(hipTextureObject_t textureObject);
  
  // Start to capture knernel invocation on the given stream
  hipError_t StartCaptureMode(ClQueue* stream) {
    if (this->captureQueue != nullptr) {
      return hipErrorInvalidResourceHandle;
    } else if (stream == nullptr) {
      return hipErrorInvalidResourceHandle;
    } else {
      this->captureQueue = (LZQueue* )stream;
    }

    return hipSuccess;
  }

  // End the capture mode
  hipError_t EndCaptureMode(ClQueue* stream, LZGraph* graph);

  virtual void synchronizeQueues(hipStream_t queue);

  bool CreateSyncEventPool(uint32_t count, ze_event_pool_handle_t &pool);

  bool GetSyncEvent(hipContextSyncData *syncEvent);

  bool CreateMonitor();

  void WaitEventMonitor();

  bool StopMonitor() { return stopMonitor; }

  // Create LZProgam from binary
  virtual ClProgram *createProgram(std::string &binary);
  
protected:
   // Get HipLZ kernel via function name
  hipFunction_t GetKernelByFunctionName(std::string funcName);

  virtual ExecItem* createExecItem(dim3 grid, dim3 block, size_t shared, hipStream_t stream) {
    return new LZExecItem(grid, block, shared, stream);
  }
};

// HipLZ driver object that manages device objects and the current context
class LZDriver {
protected:
  // The synchronization mutex
  std::mutex DriverMutex;

  // The driver handler
  ze_driver_handle_t hDriver;

  // Thhe device type maintained in this driver
  ze_device_type_t deviceType;

  // The device objects
  std::vector<LZDevice* > devices;

  // The primary device ID
  int primaryDeviceId;

public:
  // The global storage for module binary
  static std::vector<std::string *> FatBinModules;

  // The global storage for kernel functions, includig module, hostFunction and deviceName
  static std::vector<std::tuple<std::string *, const void *, const char* >> RegFunctions;

  // The global storage for global variables include module, hostVar, deviceName and size
  static std::vector<std::tuple<std::string *, char *, const char *, int>> GlobalVars;

public:
  LZDriver(ze_driver_handle_t hDriver_, const ze_device_type_t deviceType_) : primaryDeviceId(0) {
    this->hDriver = hDriver_;
    this->deviceType = deviceType_;

    // Collect HipLZ devices
    FindHipLZDevices();
  };

  LZDriver(ze_driver_handle_t hDriver_, const ze_device_type_t deviceType_,
	   ze_device_handle_t hDevice,
	   ze_context_handle_t hContext,
	   ze_command_queue_handle_t hQueue) : primaryDeviceId(0) {
    this->hDriver = hDriver_;
    this->deviceType = deviceType_;

    // Setup HipLZ devie
    FindHipLZDevices(hDevice, hContext, hQueue);
  }

  // Get and initialize the drivers
  static bool InitDrivers(std::vector<LZDriver* >& drivers, const ze_device_type_t deviceType);

  // Initialize the driver via pre-initialized resource
  static bool InitDriver(std::vector<LZDriver* >& drivers,
			 const ze_device_type_t deviceType,
			 ze_driver_handle_t driverHandle,
			 ze_device_handle_t deviceHandle,
			 ze_context_handle_t ctxPtr,
			 ze_command_queue_handle_t queueHandle);

  // Get HipLZ driver via integer ID
  static LZDriver& HipLZDriverById(int id);

  // Get the primary HipLZ driver
  static LZDriver& GetPrimaryDriver() {
    // Here we use driver 0 as the primary one
    return HipLZDriverById(0);
  };

  // Get the number of HipLZ devices
  int GetNumOfDevices() { return this->devices.size(); };

  // Get the driver handler
  ze_driver_handle_t GetDriverHandle() { return this->hDriver; };

  // Set the primary device
  void setPrimaryDevice(int deviceId) {
    primaryDeviceId = deviceId;
  }

  // Get the primary device
  LZDevice& getPrimaryDevice() {
    return * devices.at(primaryDeviceId);
  }

  // Register the given module to all devices
  void registerModule(std::string *module) {
    for (int deviceId = 0; deviceId < devices.size(); deviceId ++) {
      devices[deviceId]->registerModule(module);
    }
  }

  // Register the given kernel function to all devices
  bool registerFunction(std::string *module, const void *HostFunction, const char *FunctionName) {
    for (int deviceId = 0; deviceId < devices.size(); deviceId ++) {
      if (!devices[deviceId]->registerFunction(module, HostFunction, FunctionName))
	return false;
    }

    return true;
  }

  // Register global variable
  bool registerVar(std::string *module, const void *HostVar, const char *VarName) {
    for (int deviceId = 0; deviceId < devices.size(); deviceId ++) {
      if (!devices[deviceId]->registerVar(module, HostVar, VarName))
        return false;
    }

    return true;
  }

  bool registerVar(std::string *module, const void *HostVar, const char *VarName, size_t size) {
    for (int deviceId = 0; deviceId < devices.size(); deviceId ++) {
      if (!devices[deviceId]->registerVar(module, HostVar, VarName, size))
        return false;
    }

    return true;
  }

  // Get the HipLZ device driver by ID
  LZDevice& GetDeviceById(int id);

protected:
  // Collect HipLZ device that belongs to this driver
  bool FindHipLZDevices(ze_device_handle_t hDevice = nullptr,
			ze_context_handle_t hContext = nullptr,
			ze_command_queue_handle_t hQueue = nullptr);
};

class LZCommandList {
protected:
  // Current associated HipLZ context
  LZContext* lzContext;

  ze_event_pool_handle_t eventPool;
  ze_event_handle_t finishEvent;
  // HipLZ command list handler
  ze_command_list_handle_t hCommandList;

  // The shared memory buffer
  void* shared_buf;

public:
  LZCommandList(LZContext* lzContext);

  // Create HipLZ command list
  static LZCommandList* CreateCmdList(LZContext* lzContext, bool immediate = true);

  // Get command list handler
  ze_command_list_handle_t GetCommandListHandle() { return this->hCommandList; }

  // Execute Level-0 kernel
  bool ExecuteKernel(LZQueue* lzQueue, LZKernel* Kernel, LZExecItem* Arguments);
  bool ExecuteKernelAsync(LZQueue* lzQueue, LZKernel* Kernel, LZExecItem* Arguments);

  // Execute HipLZ memory copy command
  bool ExecuteMemCopy(LZQueue* lzQueue, void *dst, const void *src, size_t sizeBytes);
  bool ExecuteMemCopyAsync(LZQueue* lzQueue, void *dst, const void *src, size_t sizeBytes);

  // Execute memory HipLZ copy region
  bool ExecuteMemCopyRegion(LZQueue* lzQueue, void *dst, size_t dpitch, const void *src, size_t spitch,
			    size_t width, size_t height);
  bool ExecuteMemCopyRegionAsync(LZQueue* lzQueue, void *dst, size_t dpitch, const void *src,
				 size_t spitch,  size_t width, size_t height);

  bool ExecuteMemCopyRegion(LZQueue* lzQueue, void *dst, size_t dpitch, size_t dspitch,
			    const void *src, size_t spitch, size_t sspitch,
			    size_t width, size_t height, size_t depth);
  bool ExecuteMemCopyRegionAsync(LZQueue* lzQueue, void *dst, size_t dpitch, size_t dspitch,
				 const void *src, size_t spitch, size_t sspitch,
				 size_t width, size_t height, size_t depth);

  // Execute HipLZ memory fill command
  bool ExecuteMemFill(LZQueue* lzQueue, void *dst, size_t size, const void *pattern, size_t pattern_size);
  bool ExecuteMemFillAsync(LZQueue* lzQueue, void *dst, size_t size, const void *pattern, size_t pattern_size);

  // Execute HipLZ write global timestamp
  bool ExecuteWriteGlobalTimeStamp(LZQueue* lzQueue, uint64_t *timestamp);
  bool ExecuteWriteGlobalTimeStampAsync(LZQueue* lzQueue, uint64_t *timestamp, LZEvent *event);

  // Execute HipLZ memory copy to texture object, i.e. image
  bool ExecuteMemCopyToTexture(LZQueue* lzQueue, LZTextureObject* texObj, void* src);
  
  // Execute the memory prefetch
  bool ExecuteMemPrefetch(LZQueue* lzQueue, const void* ptr, size_t size);
  bool ExecuteMemPrefetchAsync(LZQueue* lzQueue, const void* ptr, size_t size);

  // Make the advise for the managed memory (i.e. unified shared memory)
  bool ExecuteMemAdvise(LZQueue* lzQueue, const void* ptr, size_t count, hipMemoryAdvise advice);
  bool ExecuteMemAdviseAsync(LZQueue* lzQueue, const void* ptr, size_t count, hipMemoryAdvise advice);
		
  // Execute HipLZ command list
  virtual bool Execute(LZQueue* lzQueue);

  // Execute HipLZ command list asynchronously
  virtual bool ExecuteAsync(LZQueue* lzQueue);

  // Synchronize host with device kernel execution
  virtual bool finish();

private:
  void kernel(LZQueue* lzQueue, LZKernel* Kernel, LZExecItem* Arguments);
  void memCopy(LZQueue* lzQueue, void *dst, const void *src, size_t sizeBytes);
  void memCopyRegion(LZQueue* lzQueue, void *dst, size_t dpitch, const void *src, size_t spitch,
                     size_t width, size_t height);
  void memCopyRegion(LZQueue* lzQueue, void *dst, size_t dpitch, size_t dspitch,
                     const void *src, size_t spitch, size_t sspitch,
                     size_t width, size_t height, size_t depth);
  void memFill(LZQueue* lzQueue, void *dst, size_t size, const void *pattern, size_t pattern_size);
  void memCopyToTexture(LZQueue* lzQueue, LZTextureObject* texObj, void* src);
  void memPrefetch(LZQueue* lzQueue, const void* ptr, size_t size);
  void memAdvise(LZQueue* lzQueue, const void* ptr, size_t count, hipMemoryAdvise advice);
};

// The standard level-0 command list
class LZStdCommandList : public LZCommandList {
public:
  LZStdCommandList(LZContext* lzContext) : LZCommandList(lzContext) {
    initializeCmdList();
  };

  // Execute HipLZ command list
  virtual bool Execute(LZQueue* lzQueue);

  // Execute HipLZ command list asynchronously
  virtual bool ExecuteAsync(LZQueue* lzQueue);

  // Synchronize host with device kernel execution
  virtual bool finish();

protected:
  // Initialize standard level-0 command list
  bool initializeCmdList();
};

// The immedate level-0 command list
class LZImmCommandList : public LZCommandList {
protected:
  // The per-command list event pool that assists synchronization for immediate command list
  ze_event_pool_handle_t eventPool;

  // The finish event that denotes the finish of current command list items
  ze_event_handle_t finishEvent;

public:
  LZImmCommandList(LZContext* lzContext) : LZCommandList(lzContext) {
    initializeCmdList();
  };

  // Execute HipLZ command list
  virtual bool Execute(LZQueue* lzQueue);

  // Execute HipLZ command list asynchronously
  virtual bool ExecuteAsync(LZQueue* lzQueue);

  // Synchronize host with device kernel execution
  virtual bool finish();

protected:
  // Initialize standard level-0 command list
  bool initializeCmdList();
};

class LZGraphNode;

class LZQueue : public ClQueue {
protected:
  // Level-0 context reference
  LZContext* lzContext;

  // Level-0 handler
  ze_command_queue_handle_t hQueue;

  // Default command list
  LZCommandList* defaultCmdList;

  // The current HipLZ event, currently, we only maintain one event for each queue
  // LZEvent* currentEvent;

  // The list of local events
  std::list<LZEvent* > localEvents;

  // The lock of event list
  std::mutex EventsMutex;

  // The list of callbacks
  std::list<hipStreamCallbackData> callbacks;

  // The lock of callback list
  std::mutex CallbacksMutex;

  // The thread ID for monitor thread
  pthread_t monitorThreadId;

  // The buffer for captured kernel function call
  std::vector<LZGraphNode* > nodeBuf;

public:
  LZQueue(LZContext* lzContext, unsigned int f, int p);
  LZQueue(LZContext* lzContext, LZCommandList* lzCmdList, unsigned int f, int p);
  LZQueue(LZContext* lzContext_, ze_command_queue_handle_t hQueue_, LZCommandList* lzCmdList, unsigned int f, int p);

  ~LZQueue() {
    // Detach from LZContext object
    this->lzContext = nullptr;
    // Do thread join to wait for thread termination
    WaitEventMonitor();
  };

  // Get Level-0 queue handler
  ze_command_queue_handle_t GetQueueHandle() { return this->hQueue; }

  // Get the Level-0 context object

  // Get OpenCL command queue
  virtual cl::CommandQueue &getQueue();
  // Get queue flags
  virtual unsigned int getFlags() const { return Flags; }
  // Get queue priority
  virtual int getPriority() const { return Priority; }

  // Queue synchronous support
  virtual bool finish();
  // Enqueue barrier for event
  virtual bool enqueueBarrierForEvent(hipEvent_t event);
  // Enqueue barrier and signal event
  virtual bool enqueueZeBarrier(ze_event_handle_t event = nullptr, uint32_t waitCount = 0, ze_event_handle_t *waitList = nullptr);
  // Add call back
  virtual bool addCallback(hipStreamCallback_t callback, void *userData);
  // Record event
  virtual bool recordEvent(hipEvent_t e);

  // Memory copy support
  virtual hipError_t memCopy(void *dst, const void *src, size_t size);
  virtual hipError_t memCopyAsync(void *dst, const void *src, size_t sizeBytes);
  // Memory fill support
  virtual hipError_t memFill(void *dst, size_t size, const void *pattern, size_t pattern_size);
  virtual hipError_t memFillAsync(void *dst, size_t size, const void *pattern, size_t pattern_size);
  // The memory copy 2D support
  virtual hipError_t memCopy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
			       size_t width, size_t height);
  virtual hipError_t memCopy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
			    size_t width, size_t height);
  // The memory copy 3D support
  virtual hipError_t memCopy3D(void *dst, size_t dpitch, size_t dspitch,
			       const void *src, size_t spitch, size_t sspitch,
			       size_t width, size_t height, size_t depth);
  virtual hipError_t memCopy3DAsync(void *dst, size_t dpitch, size_t dspitch,
			    const void *src, size_t spitch, size_t sspitch,
			    size_t width, size_t height, size_t depth);

  // Memory copy to texture object, i.e. image
  hipError_t memCopyToTexture(LZTextureObject* texObj, void* src);
    
  // Make meory prefetch
  virtual hipError_t memPrefetch(const void* ptr, size_t size);

  // Make the advise for the managed memory (i.e. unified shared memory)
  virtual hipError_t memAdvise(const void* ptr, size_t count, hipMemoryAdvise advice);

  // Launch kernel support
  virtual hipError_t launch3(ClKernel *Kernel, dim3 grid, dim3 block);
  // Launch kernel support
  virtual hipError_t launch(ClKernel *Kernel, ExecItem *Arguments);

  // Get HipLZ context object
  LZContext* GetContext() {
    return this->lzContext;
  };

  // Get callback from lock protected callback list
  bool GetCallback(hipStreamCallbackData* data);

  // Get the default command list
  LZCommandList* GetDefaultCmdList() { return this->defaultCmdList; };

  // Get the native information
  virtual bool getNativeInfo(unsigned long* nativeInfo, int* size);

  // Capture the kernel invocation
  bool CaptureKernelCall(const void *function_address, dim3 numBlocks,
			 dim3 dimBlocks, void **args, size_t sharedMemBytes);

  // End capture
  bool EndCapture(LZGraph* graph);

protected:
  // Initialize Level-0 queue
  void initializeQueue(LZContext* lzContext, bool needDefaultCmdList = false);

  // Create the callback monitor on-demand
  bool CheckAndCreateMonitor();

  // Synchronize on the event monitor thread
  void WaitEventMonitor();

  // Enforce HIP correct stream synchronization
  void synchronizeQueues() { lzContext->synchronizeQueues(this); }
};

class LZImage {
protected:
  // Image handle
  ze_image_handle_t hImage;

  // The reference to HipLZ context
  LZContext* lzContext;

public:
  LZImage(LZContext* lzContext, hipResourceDesc* resDesc, hipTextureDesc* texDesc);

  // Get the image handle
  ze_image_handle_t GetImageHandle() { return this->hImage; };

  // Update data to image
  bool upload(hipStream_t stream, void* srcptr);
};

// The struct that accomodate the L0/Hip texture object's content
class LZTextureObject {
public:
  intptr_t  image;
  intptr_t  sampler;

  LZTextureObject() {};

  // The factory function for creating the LZ texture object
  static LZTextureObject* CreateTextureObject(LZContext* lzCtx,
                                              const hipResourceDesc* pResDesc,
                                              const hipTextureDesc* pTexDesc,
                                              const struct hipResourceViewDesc* pResViewDesc);

  // Destroy the HIP texture object
  static bool DestroyTextureObject(LZTextureObject* texObj);
  
protected:
  // The factory function for create the LZ image object
  static bool CreateImage(LZContext* lzCtx,
                          const hipResourceDesc* pResDesc,
                          const hipTextureDesc* pTexDesc,
                          const struct hipResourceViewDesc* pResViewDesc,
                          ze_image_handle_t* handle);

  // Destroy the LZ image object
  static bool DestroyImage(ze_image_handle_t handle);
  
  // The factory function for create the LZ sampler object
  static bool CreateSampler(LZContext* lzCtx,
                            const hipResourceDesc* pResDesc,
                            const hipTextureDesc* pTexDesc,
                            const struct hipResourceViewDesc* pResViewDesc,
                            ze_sampler_handle_t* handle);

  // Destroy the LZ sampler object
  static bool DestroySampler(ze_sampler_handle_t handle);
};

// HIP graph support

class LZGraphExec;

class LZGraphNode;

class LZGraph {
public:
  LZGraph() : root(0), tail(0) {};

  // Release all graph nodes
  ~LZGraph() {
    destroy();
  };

  // Add graph node
  bool addGraphNode(LZGraphNode* graphNode, LZGraphNode** dependNodes, int numDeps);

  // Add graph node as tail
  bool addTailNode(LZGraphNode* graphNode);

  // Instantiate the graph for execution
  bool instantiate(LZGraphExec* graphExec);

  // Destroy graph
  void destroy();

protected:
  // The map beteween graph node and its successors 
  std::map<LZGraphNode* , std::vector<LZGraphNode* > > nodeSuccs;

  // The root graph node
  LZGraphNode* root;

  // The tail graph node
  LZGraphNode* tail;
};

class LZGraphNode {
protected:
  // Current graph node ID
  int ID;

  friend LZGraph;

public:
  LZGraphNode() : ID(0) {};
  virtual ~LZGraphNode() {};

  // Execute the graph node 
  virtual bool execute(ClQueue* queue) = 0;

  // Instantiate the graph node
  virtual bool instantiate() = 0;
};

class LZGraphNodeKernel : public LZGraphNode {
public:
  LZGraphNodeKernel(hipKernelNodeParams* params) {
    // Copy parameters
    this->params = * params;
  }

  virtual ~LZGraphNodeKernel() {};

  // Execute the kernel function
  virtual bool execute(ClQueue* queue);

  // Instantiate the graph node 
  virtual bool instantiate();

protected:
  // Kernel function parameters
  hipKernelNodeParams params;
};

class LZGraphNodeMemcpy : public LZGraphNode {
public:
  LZGraphNodeMemcpy(void* dst, const void* src, size_t count, hipMemcpyKind kind) : dst(dst), 
										    src(src), 
										    count(count), 
										    kind(kind) {
  };

  virtual ~LZGraphNodeMemcpy() {};

  // Execute the memory copy
  virtual bool execute(ClQueue* queue);

  // Instantiate the graph node 
  virtual bool instantiate();

protected:
  void* dst;
  const void* src;
  size_t count; 
  hipMemcpyKind kind;
};

class LZGraphExec {
protected:
  // Add graph nodes for execution
  void addInstantiatedGraphNode(LZGraphNode* node) {
    nodes.push_back(node);
  }

  // Instantiate gaph node
  void instantiate() {
    for (int i = 0; i < nodes.size(); i ++) {
      nodes[i]->instantiate();
    }
  }

  friend LZGraph;

public:
  LZGraphExec() {};

  // Iteratively execute instantiated graph nodes
  bool execute(ClQueue* queue) {
    for (int i = 0; i < nodes.size(); i ++) {
      if (!nodes[i]->execute(queue))
	// The graph execution is interrupted if there's a failure happened
	return false;
    }

    return true;
  }

protected:
  // Graph nodes for execution 
  std::vector<LZGraphNode* > nodes;
};

LZDevice &HipLZDeviceById(int deviceId);

extern size_t NumLZDevices;

extern size_t NumLZDrivers;

void InitializeHipLZ();

void InitializeHipLZFromOutside(ze_driver_handle_t hDriver,
				ze_device_handle_t hDevice,
				ze_context_handle_t hContext,
				ze_command_queue_handle_t hQueue);

/********************************************************************/

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
