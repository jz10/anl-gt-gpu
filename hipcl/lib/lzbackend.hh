#include <list>
#include <map>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <vector>

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

enum class LZMemoryType : unsigned { Host = 0, Device = 1, Shared = 2};

class LZExecItem;

class LZKernel;

class LZExecItem : public ExecItem {
public:
  dim3 GridDim;
  dim3 BlockDim;

  LZExecItem(dim3 grid, dim3 block, size_t shared, hipStream_t stream) : ExecItem(grid, block, shared, stream) {
    GridDim = grid;
    BlockDim = block;
  }

  // Setup all arguments for HipLZ kernel funciton invocation
  int setupAllArgs(LZKernel *kernel);

  virtual bool launch(LZKernel *Kernel);

  // If this execution item support HipLZ
  virtual bool SupportLZ() { return true; };
};

class LZContext;

class LZDriver;

// HipLZ device object that 
class LZDevice {
protected:
  enum PeerAccessState{ Accessible, UnAccessible, Accessible_Disabled, Uninitialized };
  
  // Synchronization mutex
  std::mutex DeviceMutex;

  // The default context associated with this device
  LZContext* defaultContext;

  // The device handler
  ze_device_handle_t hDevice;

  // The driver object
  LZDriver* driver;

  // The names of modules
  std::vector<std::string *> Modules;
  
  // The map between host function pointer and module name 
  std::map<const void *, std::string *> HostPtrToModuleMap;

  // The map between host function pointer nd name
  std::map<const void *, std::string> HostPtrToNameMap;

  // The device memory properties 
  ze_device_memory_properties_t deviceMemoryProps;
  
  // The device compute properties
  ze_device_compute_properties_t deviceComputeProps;

  // The device cache properties
  ze_device_cache_properties_t deviceCacheProps;

  // The device module properties
  ze_device_module_properties_t deviceModuleProps;
  
  // The size of total used memory
  size_t TotalUsedMem;

  // The handle of device properties
  ze_device_properties_t deviceProps;

  // The integer ID of current device
  hipDevice_t deviceId;

  // The current device's properties
  hipDeviceProp_t Properties;
  
  // The Hip attribute map
  std::map<hipDeviceAttribute_t, int> Attributes;

  // The command queue ordinal 
  uint32_t cmdQueueGraphOrdinal;

  // The device peer access states
  std::map<int, PeerAccessState> peerAccessTable;
  
public:
  LZDevice(hipDevice_t id,  ze_device_handle_t hDevice, LZDriver* driver);
  
  // Get device properties
  ze_device_properties_t* GetDeviceProps() { return &(this->deviceProps); };

  // Get device handle
  ze_device_handle_t GetDeviceHandle() { return this->hDevice; };

  // Get current device driver handle
  ze_driver_handle_t GetDriverHandle();

  // Get current device's integer ID
  hipDevice_t getHipDeviceT() {
    return this->deviceId;
  }

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
  LZContext* getPrimaryCtx() { return this->defaultContext; };

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
  // Setup HipLZ device properties 
  void setupProperties(int index);

  bool retrieveCmdQueueGroupOrdinal(uint32_t& ordinal);
};

class LZModule;

class LZKernel : public ClKernel {
protected:
  // HipLZ kernel handle
  ze_kernel_handle_t hKernel;
  // The function info
  OCLFuncInfo *FuncInfo;

public: 
  LZKernel(LZModule* lzModule, std::string funcName, OCLFuncInfo* FuncInfo_);
  ~LZKernel();

  ze_kernel_handle_t GetKernelHandle() { return this->hKernel; }

  OCLFuncInfo *getFuncInfo() const { return FuncInfo; }

  // If this kernel support HipLZ
  virtual bool SupportLZ() { return true; };
};

class LZModule {
protected:
  // HipLZ module handle
  ze_module_handle_t hModule;
  // The name --> HipLZ kernel map
  std::map<std::string, LZKernel* > kernels;

public:
  LZModule(LZContext* lzContext, uint8_t* funcIL, size_t ilSize);
  ~LZModule();
  
  // Get HipLZ module handle  
  ze_module_handle_t GetModuleHandle() { return this->hModule; }

  // Create HipLZ kernel via function name
  void CreateKernel(std::string funcName, OpenCLFunctionInfoMap& FuncInfos);

  // Get HipLZ kernel via funciton name
  LZKernel* GetKernel(std::string funcName);

  // Check if support symbol address
  bool symbolSupported() {
    return true;
  }
  
  // Get the pointer address and size information via gievn symbol's name
  bool getSymbolAddressSize(const char *name, hipDeviceptr_t *dptr, size_t* bytes);
};

class LZEventPool;

class LZEvent : public ClEvent {
protected:
  // The mutual exclusion support
  std::mutex EventMutex;
  // cl::Event *Event;
  // Associated stream
  hipStream_t Stream;
  // Status
  event_status_e Status;
  // Flags
  unsigned Flags;
  // cl::Context Context;

  // The associated HipLZ context
  LZContext* cont;
  
  // The handler of HipLZ event
  ze_event_handle_t hEvent;

  // The timestamp value
  uint64_t timestamp;

public:
  LZEvent(LZContext* c, unsigned flags, LZEventPool* eventPool);

  LZEvent(cl::Context &c, unsigned flags) {
    // TODO:
  };
    
  virtual ~LZEvent() {
    if (Event)
      delete Event;
  }

  // The memory space for global timestamp
  alignas(8) char timestamp_buf[32];

  virtual uint64_t getFinishTime();
  
  // Get the event object? this is only for OpenCL 
  virtual cl::Event getEvent();

  // Check if the event is from same cl::Context? this is only for OpenCL
  virtual bool isFromContext(cl::Context &Other);

  // Check if the event is from same stream
  virtual bool isFromStream(hipStream_t &Other) { return (Stream == Other); }

  // Check if the event has been finished
  virtual bool isFinished() const { return (Status == EVENT_STATUS_RECORDED); }

  // Check if the event is during recording or has been recorded
  virtual bool isRecordingOrRecorded() const { return (Status >= EVENT_STATUS_RECORDING); }

  // Record the event to stream
  virtual bool recordStream(hipStream_t S, cl_event E);

  // Update event's finish status
  virtual bool updateFinishStatus();

  // Wait on event get finished
  virtual bool wait();

  // Get current event handler
  ze_event_handle_t GetEventHandler() { return this->hEvent; };  

  // Record the time stamp
  void recordTimeStamp(uint64_t value) { this->timestamp = value; };

  // Get the time stamp
  uint64_t getTimeStamp() { return this->timestamp; };
};

class LZEventPool {
protected:
  // The thread-safe event pool management
  std::mutex PoolMutex;
  // The associated HipLZ context
  LZContext* lzContext;
  // The handler of event pool
  ze_event_pool_handle_t hEventPool;

public:
  LZEventPool(LZContext* c);

  // Create new event
  LZEvent* createEvent(unsigned flags);

  // Get the handler of event pool
  ze_event_pool_handle_t GetEventPoolHandler() { return this->hEventPool; };
};

class LZCommandList;

class LZContext : public ClContext {
protected:
  // Lock for thread safeness
  std::mutex mtx;
  // Reference to HipLZ device
  LZDevice* lzDevice;
  
  // Map between IL binary to HipLZ module
  std::map<uint8_t* , LZModule* > IL2Module;
  
  // Reference to HipLZ command list
  LZCommandList* lzCommandList;
  // Reference to HipLZ queue
  LZQueue* lzQueue;
  // The default event ppol
  LZEventPool* defaultEventPool;

  // HipLZ context handle
  ze_context_handle_t hContext;
  
  // OpenCL function information map, this is used for presenting SPIR-V kernel funcitons' arguments
  OpenCLFunctionInfoMap FuncInfos;

  // The map between global variable name and its relevant HipLZ module, device poitner and size information
  std::map<std::string, std::tuple<LZModule *, hipDeviceptr_t, size_t>> GlobalVarsMap;
  
public:
  LZContext(ClDevice* D, unsigned f) : ClContext(D, f), lzDevice(0), lzCommandList(0), 
				       lzQueue(0), defaultEventPool(0) {}
  LZContext(LZDevice* dev);  

  // Create SPIR-V module
  bool CreateModule(uint8_t* moduleIL, size_t ilSize, std::string funcName);

  // Get Level-0 handle for context
  ze_context_handle_t GetContextHandle() { return this->hContext; }

  // Get Level-0 device object
  LZDevice* GetDevice() { return this->lzDevice; };

  // Get Level-0 queue object
  LZQueue*  GetQueue() { return this->lzQueue; };
  
  // Configure the call for kernel
  hipError_t configureCall(dim3 grid, dim3 block, size_t shared, hipStream_t q);
  
  // Set argument
  hipError_t setArg(const void *arg, size_t size, size_t offset);

  // Launch HipLZ kernel
  bool launchHostFunc(const void* HostFunction);

  // Memory allocation
  void *allocate(size_t size);

  // Memory free
  bool free(void *p);

  // Get pointer info
  bool findPointerInfo(hipDeviceptr_t dptr, hipDeviceptr_t *pbase, size_t *psize);
  bool getPointerSize(void *ptr, size_t *size);

  // Memory copy
  hipError_t memCopy(void *dst, const void *src, size_t sizeBytes, hipStream_t stream);
  hipError_t memCopy(void *dst, const void *src, size_t sizeBytes);

  // Memory copy 2D
  virtual hipError_t memCopy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
			       size_t width, size_t height, hipStream_t stream);

  // Memory copy 3D
  virtual hipError_t memCopy3D(void *dst, size_t dpitch, size_t dspitch,
		       const void *src, size_t spitch, size_t sspitch,
		       size_t width, size_t height, size_t depth, hipStream_t stream);
  
  // Asynchronous memory copy
  virtual hipError_t memCopyAsync(void *dst, const void *src, size_t sizeBytes, hipStream_t stream);

  // Asynchronous memory copy 2D
  virtual hipError_t memCopy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
			    size_t width, size_t height, hipStream_t stream);

  // Asynchronous memory copy 3D
  virtual hipError_t memCopy3DAsync(void *dst, size_t dpitch, size_t dspitch,
			    const void *src, size_t spitch, size_t sspitch,  
			    size_t width, size_t height, size_t depth, hipStream_t stream);
  
  // Memory fill
  hipError_t memFill(void *dst, size_t size, const void *pattern, size_t pattern_size, hipStream_t stream);
  hipError_t memFill(void *dst, size_t size, const void *pattern, size_t pattern_size);

  // Asynchronous memory fill
  hipError_t memFillAsync(void *dst, size_t size, const void *pattern, size_t pattern_size, hipStream_t stream);

  // Cteate HipLZ event
  LZEvent* createEvent(unsigned flags);
  
  // Create stream/queue
  virtual bool createQueue(hipStream_t *stream, unsigned int Flags, int priority);

  // Get the elapse between two events
  virtual hipError_t eventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop);

  // Synchronize all streams
  virtual bool finishAll();

  // Reset current context
  virtual void reset();
  
  // Allocate memory via Level-0 runtime
  void* allocate(size_t size, size_t alignment, LZMemoryType memTy);

  // Register global variable
  bool registerVar(std::string *module, const void *HostVar, const char *VarName);
  bool registerVar(std::string *module, const void *HostVar, const char *VarName, size_t size);
  
  // Get the address and size for the given symbol's name
  virtual bool getSymbolAddressSize(const char *name, hipDeviceptr_t *dptr, size_t *bytes);
  
  // Create Level-0 image object
  LZImage* createImage(hipResourceDesc* resDesc, hipTextureDesc* texDesc);

protected:
   // Get HipLZ kernel via function name
  LZKernel* GetKernelByFunctionName(std::string funcName);
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
  int primaryDevieId;
  
public:
  LZDriver(ze_driver_handle_t hDriver_, const ze_device_type_t deviceType_) : primaryDevieId(0) { 
    this->hDriver = hDriver_;
    this->deviceType = deviceType_; 
  
    // Collect HipLZ devices
    FindHipLZDevices();
  };
  
  // Get and initialize the drivers
  static bool InitDrivers(std::vector<LZDriver* >& drivers, const ze_device_type_t deviceType);

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
    primaryDevieId = deviceId;
  }
  
  // Get the primary device
  LZDevice& getPrimaryDevice() {
    return * devices.at(primaryDevieId);
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
  bool FindHipLZDevices();
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
  
  // Execute HipLZ memory copy command 
  bool ExecuteMemCopy(LZQueue* lzQueue, void *dst, const void *src, size_t sizeBytes);

  // Execute memory HipLZ copy regiion
  bool ExecuteMemCopyRegion(LZQueue* lzQueue, void *dst, size_t dpitch, const void *src, size_t spitch,
			    size_t width, size_t height);

  bool ExecuteMemCopyRegion(LZQueue* lzQueue, void *dst, size_t dpitch, size_t dspitch,
			    const void *src, size_t spitch, size_t sspitch,
			    size_t width, size_t height, size_t depth);
  
  // Execute HipLZ memory copy command asynchronously
  bool ExecuteMemCopyAsync(LZQueue* lzQueue, void *dst, const void *src, size_t sizeBytes);

  // Execute memory HipLZ copy asynchronously
  bool ExecuteMemCopyRegionAsync(LZQueue* lzQueue, void *dst, size_t dpitch, const void *src,
				 size_t spitch,  size_t width, size_t height);

  bool ExecuteMemCopyRegionAsync(LZQueue* lzQueue, void *dst, size_t dpitch, size_t dspitch,
				 const void *src, size_t spitch, size_t sspitch,
				 size_t width, size_t height, size_t depth);
  
  // Execute HipLZ memory fill command
  bool ExecuteMemFill(LZQueue* lzQueue, void *dst, size_t size, const void *pattern, size_t pattern_size);

  // Execute HipLZ memory fill command asynchronously
  bool ExecuteMemFillAsync(LZQueue* lzQueue, void *dst, size_t size, const void *pattern, size_t pattern_size);

  // Execute HipLZ write global timestamp  
  uint64_t ExecuteWriteGlobalTimeStamp(LZQueue* lzQueue);

  // Execute HipLZ command list 
  virtual bool Execute(LZQueue* lzQueue);

  // Execute HipLZ command list asynchronously
  virtual bool ExecuteAsync(LZQueue* lzQueue);

  // Synchronize host with device kernel execution
  virtual bool finish();

protected:
  // Get the potential signal event 
  LZEvent* GetSignalEvent(LZQueue* lzQueue);
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
  
public:
  LZQueue(cl::CommandQueue q, unsigned int f, int p) :  ClQueue(q, f, p) {
    lzContext = nullptr;
    defaultCmdList = nullptr;
    monitorThreadId = 0;
  };
  LZQueue(LZContext* lzContext, bool needDefaultCmdList = false);
  LZQueue(LZContext* lzContext, LZCommandList* lzCmdList); 

  ~LZQueue() {
    // Detach from LZContext object
    this->lzContext = nullptr; 
    // Do thread join to wait for thread termination
    WaitEventMonitor();
  };

  // Get Level-0 queue handler
  ze_command_queue_handle_t GetQueueHandle() { return this->hQueue; }

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
  // Add call back
  virtual bool addCallback(hipStreamCallback_t callback, void *userData);
  // Record event
  virtual bool recordEvent(hipEvent_t e);

  // Memory copy support
  virtual hipError_t memCopy(void *dst, const void *src, size_t size);
  // The memory copy 2D support
  virtual hipError_t memCopy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
			       size_t width, size_t height);
  // The memory copy 3D support
  virtual hipError_t memCopy3D(void *dst, size_t dpitch, size_t dspitch,
			       const void *src, size_t spitch, size_t sspitch,
			       size_t width, size_t height, size_t depth);
  // Memory fill support
  virtual hipError_t memFill(void *dst, size_t size, const void *pattern, size_t pattern_size);
  // Launch kernel support
  virtual hipError_t launch3(ClKernel *Kernel, dim3 grid, dim3 block);
  // Launch kernel support
  virtual hipError_t launch(ClKernel *Kernel, ExecItem *Arguments);

  // If this queue support HipLZ
  virtual bool SupportLZ() { return true; };

  // The asynchronously memory copy support
  virtual bool memCopyAsync(void *dst, const void *src, size_t sizeBytes);

  // The memory copy 2D support
  virtual hipError_t memCopy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
			    size_t width, size_t height);
  // The memory copy 3D support
  virtual hipError_t memCopy3DAsync(void *dst, size_t dpitch, size_t dspitch,
			    const void *src, size_t spitch, size_t sspitch,
			    size_t width, size_t height, size_t depth);
  // The asynchronously memory fill support
  virtual bool memFillAsync(void *dst, size_t size, const void *pattern, size_t pattern_size);
  
  // The set the current event
  bool SetEvent(LZEvent* event);

  // Get and clear current event
  LZEvent* GetAndClearEvent();

  // Create and monitor event                                                                 
  LZEvent* CreateAndMonitorEvent(LZEvent* event);
  
  // Get HipLZ context object
  LZContext* GetContext() {
    return this->lzContext;
  };
  
  // Get an event from event list
  LZEvent* GetPendingEvent();
  
  // Get callback from lock protected callback list   
  bool GetCallback(hipStreamCallbackData* data);

  // Get the default command list
  LZCommandList* GetDefaultCmdList() { return this->defaultCmdList; };
  
protected:
  // Initialize Level-0 queue
  void initializeQueue(LZContext* lzContext, bool needDefaultCmdList = false);

  // Create the callback monitor on-demand 
  bool CheckAndCreateMonitor();
  
  // Synchronize on the event monitor thread  
  void WaitEventMonitor();
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
  ze_image_handle_t GtImageHandle() { return this->hImage; };

  // Update data to image
  bool upload(hipStream_t stream, void* srcptr);
};


LZDevice &HipLZDeviceById(int deviceId);

extern size_t NumLZDevices;

extern size_t NumLZDrivers;

void InitializeHipLZ();

/********************************************************************/

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
