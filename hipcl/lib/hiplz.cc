#include <algorithm>
#include <cassert>
#include <fstream>
#include <stack>

#include "ze_api.h"
#include "lzbackend.hh"

#include <iostream>

static thread_local hipError_t tls_LastError = hipSuccess;

static thread_local ClContext *tls_defaultCtx = nullptr;

static thread_local std::stack<ClContext *> tls_ctxStack;

static thread_local bool tls_getPrimaryCtx = true;

static hipFunctionMap g_functions INIT_PRIORITY(120);

static ClContext *getTlsDefaultCtx() {
  if ((tls_defaultCtx == nullptr) && (NumDevices > 0))
    tls_defaultCtx = CLDeviceById(0).getPrimaryCtx();
  return tls_defaultCtx;
}

// Here we still need to consider the context stack for thread local contexts   
static thread_local LZContext *tls_defaultLzCtx = nullptr;

static LZContext *getTlsDefaultLzCtx() {
  if (tls_defaultLzCtx == nullptr)
    tls_defaultLzCtx = LZDriver::HipLZDriverById(0).getPrimaryDevice().getPrimaryCtx();

  return tls_defaultLzCtx;
}

#define RETURN(x)                                                              \
  do {                                                                         \
    hipError_t err = (x);                                                      \
    tls_LastError = err;                                                       \
    return err;                                                                \
  } while (0)

#define ERROR_IF(cond, err)                                                    \
  if (cond)                                                                    \
    do {                                                                       \
      logError("Error {} at {}:{} code {}", err, __FILE__, __LINE__, #cond);   \
      tls_LastError = err;                                                     \
      return err;                                                              \
  } while (0)

#define ERROR_CHECK_DEVNUM(device)                                             \
  ERROR_IF(((device < 0) ||                                                    \
           ((size_t)device >= LZDriver::GetPrimaryDriver().GetNumOfDevices())),\
           hipErrorInvalidDevice)

#define HIPLZ_INIT()                                                           \
  do {	                                                                       \
    LZ_TRY                                                                     \
    InitializeOpenCL();                                                        \
    InitializeHipLZ();                                                         \
    LZ_CATCH                                                                   \
  } while (0)

/***********************************************************************/

hipError_t hipGetDevice(int *deviceId) {
  HIPLZ_INIT();
  
  ERROR_IF((deviceId == nullptr), hipErrorInvalidValue);

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  *deviceId = cont->GetDevice()->getHipDeviceT();
  RETURN(hipSuccess);
}

hipError_t hipGetDeviceCount(int *count) {
  HIPLZ_INIT();
  ERROR_IF((count == nullptr), hipErrorInvalidValue);
  *count = LZDriver::GetPrimaryDriver().GetNumOfDevices();
  
  RETURN(hipSuccess);
}

hipError_t hipSetDevice(int deviceId) {
  HIPLZ_INIT();

  ERROR_CHECK_DEVNUM(deviceId);

  LZ_TRY
    tls_defaultLzCtx = LZDriver::GetPrimaryDriver().GetDeviceById(deviceId).getPrimaryCtx();
  LZDriver::GetPrimaryDriver().setPrimaryDevice(deviceId);
    
    RETURN(hipSuccess);
  LZ_CATCH
}

hipError_t hipDeviceSynchronize(void) {
  HIPLZ_INIT();

  LZ_TRY

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  // Synchronize among HipLZ queues
  cont->finishAll();
  RETURN(hipSuccess);

  LZ_CATCH
}

hipError_t hipDeviceReset(void) {
  HIPLZ_INIT();

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  
  LZDevice* dev = cont->GetDevice();
  
  dev->reset();
  RETURN(hipSuccess);
}

hipError_t hipDeviceGet(hipDevice_t *device, int ordinal) {
  HIPLZ_INIT();

  ERROR_IF((device == nullptr), hipErrorInvalidDevice);
  ERROR_CHECK_DEVNUM(ordinal);

  *device = ordinal;
  RETURN(hipSuccess);
}

hipError_t hipDeviceComputeCapability(int *major, int *minor,
                                      hipDevice_t deviceId) {
  HIPLZ_INIT();

  ERROR_CHECK_DEVNUM(deviceId);

  hipDeviceProp_t props;
  LZDriver::GetPrimaryDriver().GetDeviceById(deviceId).copyProperties(&props);
  
  if (major)
    *major = props.major;
  if (minor)
    *minor = props.minor;

  RETURN(hipSuccess);
}

hipError_t hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr,
                                 int deviceId) {
  HIPLZ_INIT();

  ERROR_CHECK_DEVNUM(deviceId);

  if (LZDriver::GetPrimaryDriver().GetDeviceById(deviceId).getAttr(pi, attr)) 
    RETURN(hipErrorInvalidValue);
  else
    RETURN(hipSuccess);
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId) {
  HIPLZ_INIT();

  // TODO: make a real properties retrieving function
  ERROR_CHECK_DEVNUM(deviceId);
  LZDriver::GetPrimaryDriver().GetDeviceById(deviceId).copyProperties(prop);

  RETURN(hipSuccess);
}

hipError_t hipDeviceGetLimit(size_t *pValue, enum hipLimit_t limit) {
  HIPLZ_INIT();

  ERROR_IF((pValue == nullptr), hipErrorInvalidValue);
  switch (limit) {
  case hipLimitMallocHeapSize:
    *pValue = 0;
    break;
  default:
    RETURN(hipErrorUnsupportedLimit);
  }
  RETURN(hipSuccess);
}

hipError_t hipDeviceGetName(char *name, int len, hipDevice_t deviceId) {
  HIPLZ_INIT();

  ERROR_CHECK_DEVNUM(deviceId);

  size_t namelen = strlen(LZDriver::GetPrimaryDriver().GetDeviceById(deviceId).getName());
  namelen = (namelen < (size_t)len ? namelen : len - 1);
  memcpy(name, LZDriver::GetPrimaryDriver().GetDeviceById(deviceId).getName(), namelen);
  name[namelen] = 0;
  RETURN(hipSuccess);
}

hipError_t hipDeviceTotalMem(size_t *bytes, hipDevice_t deviceId) {
  HIPLZ_INIT();

  ERROR_CHECK_DEVNUM(deviceId);

  if (bytes)
    *bytes = LZDriver::GetPrimaryDriver().GetDeviceById(deviceId).getGlobalMemSize();
  RETURN(hipSuccess);
}

hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
  HIPLZ_INIT();

  RETURN(hipSuccess);
}

hipError_t hipDeviceGetCacheConfig(hipFuncCache_t *cacheConfig) {
  HIPLZ_INIT();

  if (cacheConfig)
    *cacheConfig = hipFuncCachePreferNone;
  RETURN(hipSuccess);
}

hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig *pConfig) {
  HIPLZ_INIT();

  if (pConfig)
    *pConfig = hipSharedMemBankSizeFourByte;
  RETURN(hipSuccess);
}

hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig pConfig) {
  HIPLZ_INIT();

  RETURN(hipSuccess);
}

hipError_t hipFuncSetCacheConfig(const void *func, hipFuncCache_t config) {
  HIPLZ_INIT();

  RETURN(hipSuccess);
}

hipError_t hipDeviceGetPCIBusId(char *pciBusId, int len, int deviceId) {
  HIPLZ_INIT();

  LZDevice& device = LZDriver::GetPrimaryDriver().GetDeviceById(deviceId);
  
  hipDeviceProp_t prop;
  device.copyProperties(&prop);
  snprintf(pciBusId, len, "%04x:%04x:%04x",
           prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
  RETURN(hipSuccess);
}

hipError_t hipDeviceGetByPCIBusId(int * deviceId, const char * pciBusId) {
  HIPLZ_INIT();

  int pciDomainID, pciBusID, pciDeviceID;
  int err = sscanf(pciBusId, "%4x:%4x:%4x", &pciDomainID, &pciBusID, &pciDeviceID);
  if (err == EOF || err < 3)
    RETURN(hipErrorInvalidValue);
  for (size_t i = 0; i < LZDriver::GetPrimaryDriver().GetNumOfDevices(); i ++) {
    LZDevice& device = LZDriver::GetPrimaryDriver().GetDeviceById(i);
    if (device.HasPCIBusId(pciDomainID, pciBusID, pciDeviceID)) {
      * deviceId = i;
      RETURN(hipSuccess);
    }
  }
  
  RETURN(hipErrorInvalidDevice);
}

hipError_t hipSetDeviceFlags(unsigned flags) {
  HIPLZ_INIT();

  // TODO
  RETURN(hipSuccess);
}

hipError_t hipDeviceCanAccessPeer(int *canAccessPeer, int deviceId,
                                  int peerDeviceId) {
  HIPLZ_INIT();

  ERROR_CHECK_DEVNUM(deviceId);
  ERROR_CHECK_DEVNUM(peerDeviceId);
  if (deviceId == peerDeviceId) {
    *canAccessPeer = 0;
    RETURN(hipSuccess);
  }

  LZ_TRY
     
  LZDevice& device = LZDriver::GetPrimaryDriver().GetDeviceById(deviceId);
  LZDevice& peerDevice = LZDriver::GetPrimaryDriver().GetDeviceById(peerDeviceId);
  LZDevice::CanAccessPeer(device, peerDevice, canAccessPeer);

  LZ_CATCH
    
  RETURN(hipSuccess);
}

hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
  HIPLZ_INIT();

  // TODO
  int deviceId = getTlsDefaultLzCtx()->GetDevice()->getHipDeviceT();

  LZ_TRY
  LZDevice& device = LZDriver::GetPrimaryDriver().GetDeviceById(deviceId);
  LZDevice& peerDevice = LZDriver::GetPrimaryDriver().GetDeviceById(peerDeviceId);
  peerDevice.SetAccess(device, flags, true);
  
  LZ_CATCH

  RETURN(hipSuccess);
  // RETURN(hipErrorInvalidDevice);
}

hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
  HIPLZ_INIT();

  // TODO
  int deviceId = getTlsDefaultLzCtx()->GetDevice()->getHipDeviceT();

  LZ_TRY
  LZDevice& device = LZDriver::GetPrimaryDriver().GetDeviceById(deviceId);
  LZDevice& peerDevice = LZDriver::GetPrimaryDriver().GetDeviceById(peerDeviceId);
  peerDevice.SetAccess(device, 0, false);
  
  LZ_CATCH
    
  RETURN(hipSuccess);
  // RETURN(hipErrorPeerAccessNotEnabled);
}

hipError_t hipChooseDevice(int *device, const hipDeviceProp_t *prop) {
  HIPLZ_INIT();

  hipDeviceProp_t tempProp;
  ERROR_IF(((device == nullptr) || (prop == nullptr)), hipErrorInvalidValue);

  int inPropCount = 0;
  int matchedPropCount = 0;

  *device = 0;

  LZ_TRY
    
  for (size_t i = 0; i < LZDriver::GetPrimaryDriver().GetNumOfDevices(); i++) {
    // CLDeviceById(i).copyProperties(&tempProp);
    LZDriver::GetPrimaryDriver().GetDeviceById(i).copyProperties(&tempProp);
    
    if (prop->major != 0) {
      inPropCount++;
      if (tempProp.major >= prop->major) {
        matchedPropCount++;
      }
      if (prop->minor != 0) {
        inPropCount++;
        if (tempProp.minor >= prop->minor) {
          matchedPropCount++;
        }
      }
    }
    if (prop->totalGlobalMem != 0) {
      inPropCount++;
      if (tempProp.totalGlobalMem >= prop->totalGlobalMem) {
        matchedPropCount++;
      }
    }
    if (prop->sharedMemPerBlock != 0) {
      inPropCount++;
      if (tempProp.sharedMemPerBlock >= prop->sharedMemPerBlock) {
        matchedPropCount++;
      }
    }
    if (prop->maxThreadsPerBlock != 0) {
      inPropCount++;
      if (tempProp.maxThreadsPerBlock >= prop->maxThreadsPerBlock) {
        matchedPropCount++;
      }
    }
    if (prop->totalConstMem != 0) {
      inPropCount++;
      if (tempProp.totalConstMem >= prop->totalConstMem) {
        matchedPropCount++;
      }
    }
    if (prop->multiProcessorCount != 0) {
      inPropCount++;
      if (tempProp.multiProcessorCount >= prop->multiProcessorCount) {
        matchedPropCount++;
      }
    }
    if (prop->maxThreadsPerMultiProcessor != 0) {
      inPropCount++;
      if (tempProp.maxThreadsPerMultiProcessor >=
          prop->maxThreadsPerMultiProcessor) {
        matchedPropCount++;
      }
    }
    if (prop->memoryClockRate != 0) {
      inPropCount++;
      if (tempProp.memoryClockRate >= prop->memoryClockRate) {
        matchedPropCount++;
      }
    }
    if (inPropCount == matchedPropCount) {
      *device = i;
      RETURN(hipSuccess);
    }
  }

  LZ_CATCH
    
  RETURN(hipErrorInvalidValue);
}

hipError_t hipDriverGetVersion(int *driverVersion) {
  HIPLZ_INIT();

  if (driverVersion) {
    *driverVersion = 4;
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipRuntimeGetVersion(int *runtimeVersion) {
  HIPLZ_INIT();

  if (runtimeVersion) {
    *runtimeVersion = 1;
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);
}

/********************************************************************/

hipError_t hipGetLastError(void) {
  HIPLZ_INIT();

  hipError_t temp = tls_LastError;
  tls_LastError = hipSuccess;
  return temp;
}

hipError_t hipPeekAtLastError(void) {
  HIPLZ_INIT();

  return tls_LastError;
}

const char *hipGetErrorName(hipError_t hip_error) {
  switch (hip_error) {
  case hipSuccess:
    return "hipSuccess";
  case hipErrorOutOfMemory:
    return "hipErrorOutOfMemory";
  case hipErrorNotInitialized:
    return "hipErrorNotInitialized";
  case hipErrorDeinitialized:
    return "hipErrorDeinitialized";
  case hipErrorProfilerDisabled:
    return "hipErrorProfilerDisabled";
  case hipErrorProfilerNotInitialized:
    return "hipErrorProfilerNotInitialized";
  case hipErrorProfilerAlreadyStarted:
    return "hipErrorProfilerAlreadyStarted";
  case hipErrorProfilerAlreadyStopped:
    return "hipErrorProfilerAlreadyStopped";
  case hipErrorInvalidImage:
    return "hipErrorInvalidImage";
  case hipErrorInvalidContext:
    return "hipErrorInvalidContext";
  case hipErrorContextAlreadyCurrent:
    return "hipErrorContextAlreadyCurrent";
  case hipErrorMapFailed:
    return "hipErrorMapFailed";
  case hipErrorUnmapFailed:
    return "hipErrorUnmapFailed";
  case hipErrorArrayIsMapped:
    return "hipErrorArrayIsMapped";
  case hipErrorAlreadyMapped:
    return "hipErrorAlreadyMapped";
  case hipErrorNoBinaryForGpu:
    return "hipErrorNoBinaryForGpu";
  case hipErrorAlreadyAcquired:
    return "hipErrorAlreadyAcquired";
  case hipErrorNotMapped:
    return "hipErrorNotMapped";
  case hipErrorNotMappedAsArray:
    return "hipErrorNotMappedAsArray";
  case hipErrorNotMappedAsPointer:
    return "hipErrorNotMappedAsPointer";
  case hipErrorECCNotCorrectable:
    return "hipErrorECCNotCorrectable";
  case hipErrorUnsupportedLimit:
    return "hipErrorUnsupportedLimit";
  case hipErrorContextAlreadyInUse:
    return "hipErrorContextAlreadyInUse";
  case hipErrorPeerAccessUnsupported:
    return "hipErrorPeerAccessUnsupported";
  case hipErrorInvalidKernelFile:
    return "hipErrorInvalidKernelFile";
  case hipErrorInvalidGraphicsContext:
    return "hipErrorInvalidGraphicsContext";
  case hipErrorInvalidSource:
    return "hipErrorInvalidSource";
  case hipErrorFileNotFound:
    return "hipErrorFileNotFound";
  case hipErrorSharedObjectSymbolNotFound:
    return "hipErrorSharedObjectSymbolNotFound";
  case hipErrorSharedObjectInitFailed:
    return "hipErrorSharedObjectInitFailed";
  case hipErrorOperatingSystem:
    return "hipErrorOperatingSystem";
  case hipErrorSetOnActiveProcess:
    return "hipErrorSetOnActiveProcess";
  case hipErrorInvalidHandle:
    return "hipErrorInvalidHandle";
  case hipErrorNotFound:
    return "hipErrorNotFound";
  case hipErrorIllegalAddress:
    return "hipErrorIllegalAddress";
  case hipErrorInvalidSymbol:
    return "hipErrorInvalidSymbol";
    
  case hipErrorMissingConfiguration:
    return "hipErrorMissingConfiguration";
  case hipErrorMemoryAllocation:
    return "hipErrorMemoryAllocation";
  case hipErrorInitializationError:
    return "hipErrorInitializationError";
  case hipErrorLaunchFailure:
    return "hipErrorLaunchFailure";
  case hipErrorPriorLaunchFailure:
    return "hipErrorPriorLaunchFailure";
  case hipErrorLaunchTimeOut:
    return "hipErrorLaunchTimeOut";
  case hipErrorLaunchOutOfResources:
    return "hipErrorLaunchOutOfResources";
  case hipErrorInvalidDeviceFunction:
    return "hipErrorInvalidDeviceFunction";
  case hipErrorInvalidConfiguration:
    return "hipErrorInvalidConfiguration";
  case hipErrorInvalidDevice:
    return "hipErrorInvalidDevice";
  case hipErrorInvalidValue:
    return "hipErrorInvalidValue";
  case hipErrorInvalidDevicePointer:
    return "hipErrorInvalidDevicePointer";
  case hipErrorInvalidMemcpyDirection:
    return "hipErrorInvalidMemcpyDirection";
  case hipErrorUnknown:
    return "hipErrorUnknown";
  case hipErrorInvalidResourceHandle:
    return "hipErrorInvalidResourceHandle";
  case hipErrorNotReady:
    return "hipErrorNotReady";
  case hipErrorNoDevice:
    return "hipErrorNoDevice";
  case hipErrorPeerAccessAlreadyEnabled:
    return "hipErrorPeerAccessAlreadyEnabled";
  case hipErrorNotSupported:
    return "hipErrorNotSupported";
  case hipErrorPeerAccessNotEnabled:
    return "hipErrorPeerAccessNotEnabled";
  case hipErrorRuntimeMemory:
    return "hipErrorRuntimeMemory";
  case hipErrorRuntimeOther:
    return "hipErrorRuntimeOther";
  case hipErrorHostMemoryAlreadyRegistered:
    return "hipErrorHostMemoryAlreadyRegistered";
  case hipErrorHostMemoryNotRegistered:
    return "hipErrorHostMemoryNotRegistered";
  case hipErrorTbd:
    return "hipErrorTbd";
  default:
    return "hipErrorUnknown";
  }
}

const char *hipGetErrorString(hipError_t hipError) {
  return hipGetErrorName(hipError);
}

/********************************************************************/

hipError_t hipStreamCreate(hipStream_t *stream) {
  return hipStreamCreateWithFlags(stream, 0);
}

hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags) {
  return hipStreamCreateWithPriority(stream, flags, 0);
}

hipError_t hipStreamCreateWithPriority(hipStream_t *stream, unsigned int flags,
                                       int priority) {
  HIPLZ_INIT();

  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);

  LZ_TRY

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  if (cont->createQueue(stream, flags, priority))
     RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);

  LZ_CATCH

  // TODO priority & flags require an OpenCL extensions
  // ClContext *cont = getTlsDefaultCtx();
  // ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  // if (cont->createQueue(stream, flags, priority))
  //   RETURN(hipSuccess);
  // else
  //   RETURN(hipErrorInvalidValue);
}

hipError_t hipDeviceGetStreamPriorityRange(int *leastPriority,
                                           int *greatestPriority) {
  if (leastPriority)
    *leastPriority = 1;
  if (greatestPriority)
    *greatestPriority = 0;
  RETURN(hipSuccess);
}

hipError_t hipStreamDestroy(hipStream_t stream) {
  HIPLZ_INIT();

  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if (cont->releaseQueue(stream))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipStreamQuery(hipStream_t stream) {
  HIPLZ_INIT();

  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  // TODO requires OpenCL extension
  return hipSuccess;
}

hipError_t hiplzStreamNativeInfo(hipStream_t stream, unsigned long* nativeInfo, int* size) {
  ERROR_IF((stream == nullptr), hipErrorInvalidValue);

  LZQueue* lzQueue = (LZQueue* )stream;
  // TODO: consider the saft type cast
  lzQueue->getNativeInfo(nativeInfo, size);

  return hipSuccess;
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  HIPLZ_INIT();

  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  stream->finish();
  RETURN(hipSuccess);
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                              unsigned int flags) {
  HIPLZ_INIT();

  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  if (stream->enqueueBarrierForEvent(event))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags) {
  HIPLZ_INIT();

  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  ERROR_IF((flags == nullptr), hipErrorInvalidValue);

  *flags = stream->getFlags();
  RETURN(hipSuccess);
}

hipError_t hipStreamGetPriority(hipStream_t stream, int *priority) {
  HIPLZ_INIT();

  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  ERROR_IF((priority == nullptr), hipErrorInvalidValue);

  *priority = stream->getPriority();
  RETURN(hipSuccess);
}

hipError_t hipStreamAddCallback(hipStream_t stream,
                                hipStreamCallback_t callback, void *userData,
                                unsigned int flags) {
  HIPLZ_INIT();

  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  ERROR_IF((callback == nullptr), hipErrorInvalidValue);

  if (stream->addCallback(callback, userData))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

/********************************************************************/

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags, hipDevice_t device) {
  HIPLZ_INIT();

  ERROR_IF((ctx == nullptr), hipErrorInvalidValue);
  ERROR_CHECK_DEVNUM(device);

  ClContext *cont = CLDeviceById(device).newContext(flags);
  // ClContext *cont = new ClContext(device);
  ERROR_IF((cont == nullptr), hipErrorOutOfMemory);

  // device->addContext(cont)
  *ctx = cont;
  tls_defaultCtx = cont;
  tls_getPrimaryCtx = false;
  tls_ctxStack.push(cont);
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxDestroy(hipCtx_t ctx) {
  HIPLZ_INIT();

  ClContext *primaryCtx = ctx->getDevice()->getPrimaryCtx();
  ERROR_IF((primaryCtx == ctx), hipErrorInvalidValue);

  ClContext *currentCtx = getTlsDefaultCtx();
  if (currentCtx == ctx) {
    // need to destroy the ctx associated with calling thread
    tls_ctxStack.pop();
  }

  ctx->getDevice()->removeContext(ctx);

  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxPopCurrent(hipCtx_t *ctx) {
  HIPLZ_INIT();

  ERROR_IF((ctx == nullptr), hipErrorInvalidValue);
  ClContext *currentCtx = getTlsDefaultCtx();
  ClDevice *device = currentCtx->getDevice();
  *ctx = currentCtx;

  if (!tls_ctxStack.empty()) {
    tls_ctxStack.pop();
  }

  if (!tls_ctxStack.empty()) {
    currentCtx = tls_ctxStack.top();
  } else {
    currentCtx = device->getPrimaryCtx();
  }

  tls_defaultCtx = currentCtx;
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  HIPLZ_INIT();

  ERROR_IF((ctx == nullptr), hipErrorInvalidContext);

  tls_defaultCtx = ctx;
  tls_ctxStack.push(ctx);
  tls_getPrimaryCtx = false;
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  HIPLZ_INIT();

  if (ctx == nullptr) {
    tls_ctxStack.pop();
  } else {
    tls_defaultCtx = ctx;
    tls_ctxStack.push(ctx);
    tls_getPrimaryCtx = false;
  }
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetCurrent(hipCtx_t *ctx) {
  HIPLZ_INIT();

  if ((tls_getPrimaryCtx) || tls_ctxStack.empty()) {
    *ctx = getTlsDefaultCtx();
  } else {
    *ctx = tls_ctxStack.top();
  }
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetDevice(hipDevice_t *device) {
  HIPLZ_INIT();

  ClContext *ctx = getTlsDefaultCtx();

  ERROR_IF(((ctx == nullptr) || (device == nullptr)), hipErrorInvalidContext);

  ClDevice *dev = ctx->getDevice();
  *device = dev->getHipDeviceT();
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int *apiVersion) {
  HIPLZ_INIT();

  if (apiVersion) {
    *apiVersion = 4;
  }
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetCacheConfig(hipFuncCache_t *cacheConfig) {
  HIPLZ_INIT();

  if (cacheConfig)
    *cacheConfig = hipFuncCachePreferNone;

  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
  HIPLZ_INIT();

  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
  HIPLZ_INIT();

  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig *pConfig) {
  HIPLZ_INIT();

  if (pConfig)
    *pConfig = hipSharedMemBankSizeFourByte;
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSynchronize(void) {
  HIPLZ_INIT();

  ClContext *ctx = getTlsDefaultCtx();
  ctx->finishAll();
  return hipSuccess;
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetFlags(unsigned int *flags) {
  HIPLZ_INIT();

  ClContext *ctx = getTlsDefaultCtx();
  ERROR_IF((flags == nullptr), hipErrorInvalidValue);

  *flags = ctx->getFlags();
  RETURN(hipSuccess);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
  HIPLZ_INIT();

  RETURN(hipErrorInvalidValue);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
  HIPLZ_INIT();

  RETURN(hipErrorInvalidValue);
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t *pbase, size_t *psize,
                                 hipDeviceptr_t dptr) {
  HIPLZ_INIT();

  LZ_TRY
  LZContext *ctx = getTlsDefaultLzCtx();
  ERROR_IF((ctx == nullptr), hipErrorInvalidContext);

  if (ctx->findPointerInfo(dptr, pbase, psize))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
  LZ_CATCH
}

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t deviceId,
                                       unsigned int *flags, int *active) {
  HIPLZ_INIT();

  ERROR_CHECK_DEVNUM(deviceId);

  ERROR_IF((flags == nullptr || active == nullptr), hipErrorInvalidValue);

  ClContext *currentCtx = getTlsDefaultCtx();
  ClContext *primaryCtx = CLDeviceById(deviceId).getPrimaryCtx();

  *active = (primaryCtx == currentCtx) ? 1 : 0;
  *flags = primaryCtx->getFlags();

  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t deviceId) {
  HIPLZ_INIT();

  ERROR_CHECK_DEVNUM(deviceId);
  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t *pctx, hipDevice_t deviceId) {
  HIPLZ_INIT();

  ERROR_CHECK_DEVNUM(deviceId);
  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxReset(hipDevice_t deviceId) {
  HIPLZ_INIT();

  ERROR_CHECK_DEVNUM(deviceId);

  CLDeviceById(deviceId).getPrimaryCtx()->reset();

  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t deviceId,
                                       unsigned int flags) {
  HIPLZ_INIT();

  ERROR_CHECK_DEVNUM(deviceId);

  RETURN(hipErrorContextAlreadyInUse);
}

/********************************************************************/

hipError_t hipEventCreate(hipEvent_t *event) {
  return hipEventCreateWithFlags(event, 0);
}

hipError_t hipEventCreateWithFlags(hipEvent_t *event, unsigned flags) {
  HIPLZ_INIT();

  LZ_TRY

  LZContext* cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  hipEvent_t EventPtr = cont->createEvent(flags);
  if (EventPtr) {
    *event = EventPtr;
    RETURN(hipSuccess);
  } else {
    RETURN(hipErrorOutOfMemory);
  }

  LZ_CATCH
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  HIPLZ_INIT();

  LZ_TRY
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  LZContext* cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  RETURN(cont->recordEvent(stream, event));

  LZ_CATCH
}

hipError_t hipEventDestroy(hipEvent_t event) {
  HIPLZ_INIT();

  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  delete event;
  RETURN(hipSuccess);
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  HIPLZ_INIT();

  LZ_TRY
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  if (event->wait())
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
  LZ_CATCH
}

hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  HIPLZ_INIT();

  LZ_TRY
  ERROR_IF((start == nullptr), hipErrorInvalidValue);
  ERROR_IF((stop == nullptr), hipErrorInvalidValue);

  LZContext* cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  RETURN(cont->eventElapsedTime(ms, start, stop));
  LZ_CATCH
}

hipError_t hipEventQuery(hipEvent_t event) {
  HIPLZ_INIT();

  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  if (event->isFinished())
    RETURN(hipSuccess);
  else
    RETURN(hipErrorNotReady);
}

/********************************************************************/

hipError_t hipMalloc(void **ptr, size_t size) {
  HIPLZ_INIT();

  LZ_TRY
  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  if (size == 0) {
    *ptr = nullptr;
    RETURN(hipSuccess);
  }

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  void *retval = cont->allocate(size);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  LZ_CATCH
  RETURN(hipSuccess);
}

hipError_t hipMallocManaged(void ** ptr, size_t size) {
  HIPLZ_INIT();

  LZ_TRY
  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  if (size == 0) {
    *ptr = nullptr;
    RETURN(hipSuccess);
  }

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  void *retval = cont->allocate(size, LZMemoryType::Shared);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  LZ_CATCH
  RETURN(hipSuccess);
}

hipError_t hipMemPrefetchAsync(const void* ptr, size_t count, int dstDevId, hipStream_t stream) {
  HIPLZ_INIT();

  LZ_TRY
  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  // Get the relevant context
  LZDevice& dev = LZDriver::HipLZDriverById(0).GetDeviceById(dstDevId);
  
  LZContext *cont = dev.getPrimaryCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  bool retval = cont->memPrefetch(ptr, count, stream);
  ERROR_IF(retval, hipErrorInvalidDevice);
  
  LZ_CATCH
  RETURN(hipSuccess);
}

hipError_t hipMemAdvise(const void* ptr, size_t count, hipMemoryAdvise advice, int dstDevId) {
  HIPLZ_INIT();

  LZ_TRY
  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  if (ptr == 0 || count == 0) {
    RETURN(hipSuccess);
  }

  // Get the relevant context
  LZDevice& dev = LZDriver::HipLZDriverById(0).GetDeviceById(dstDevId);

  // Get the relevant context
  LZContext *cont = dev.getPrimaryCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  // Make the advise
  bool retval = cont->memAdvise(ptr, count, advice);
  ERROR_IF(retval, hipErrorInvalidDevice);
  
  LZ_CATCH
  RETURN(hipSuccess);
}

DEPRECATED("use hipHostMalloc instead")
hipError_t hipMallocHost(void **ptr, size_t size) {
  return hipMalloc(ptr, size);
}

DEPRECATED("use hipHostMalloc instead")
hipError_t hipHostAlloc(void **ptr, size_t size, unsigned int flags) {
  return hipMalloc(ptr, size);
}

hipError_t hipFree(void *ptr) {
  LZ_TRY
  ERROR_IF((ptr == nullptr), hipSuccess);

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if (cont->free(ptr))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidDevicePointer);
  LZ_CATCH
}

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
  HIPLZ_INIT();

  LZ_TRY
  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  *ptr = cont->allocate(size, 0x1000, LZMemoryType::Shared);
  LZ_CATCH
  RETURN(hipSuccess);
}

hipError_t hipHostFree(void *ptr) {
  return hipFree(ptr);
}

DEPRECATED("use hipHostFree instead")
hipError_t hipFreeHost(void *ptr) { return hipHostFree(ptr); }

hipError_t hipHostGetDevicePointer(void **devPtr, void *hstPtr,
                                   unsigned int flags) {
  HIPLZ_INIT();

  ERROR_IF(((hstPtr == nullptr) || (devPtr == nullptr)), hipErrorInvalidValue);

  *devPtr = hstPtr;
  RETURN(hipSuccess);
}

hipError_t hipHostGetFlags(unsigned int *flagsPtr, void *hostPtr) {
  HIPLZ_INIT();

  // TODO dummy implementation
  *flagsPtr = 0;
  RETURN(hipSuccess);
}

hipError_t hipHostRegister(void *hostPtr, size_t sizeBytes,
                           unsigned int flags) {
  HIPLZ_INIT();

  RETURN(hipSuccess);
}

hipError_t hipHostUnregister(void *hostPtr) {
  HIPLZ_INIT();

  RETURN(hipSuccess);
}

static hipError_t hipMallocPitch3D(void **ptr, size_t *pitch, size_t width,
                                   size_t height, size_t depth) {
  HIPLZ_INIT();

  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  *pitch = ((((int)width - 1) / SVM_ALIGNMENT) + 1) * SVM_ALIGNMENT;
  const size_t sizeBytes = (*pitch) * height * ((depth == 0) ? 1 : depth);

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  void *retval = cont->allocate(sizeBytes);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  RETURN(hipSuccess);
}

hipError_t hipMallocPitch(void **ptr, size_t *pitch, size_t width,
                          size_t height) {
  HIPLZ_INIT();

  LZ_TRY
  return hipMallocPitch3D(ptr, pitch, width, height, 0);
  LZ_CATCH
}

hipError_t hipMallocArray(hipArray **array, const hipChannelFormatDesc *desc,
                          size_t width, size_t height, unsigned int flags) {
  HIPLZ_INIT();

  LZ_TRY
  ERROR_IF((width == 0), hipErrorInvalidValue);

  auto cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidContext);

  *array = new hipArray;
  ERROR_IF((*array == nullptr), hipErrorOutOfMemory);

  array[0]->type = flags;
  array[0]->width = width;
  array[0]->height = height;
  array[0]->depth = 1;
  array[0]->desc = *desc;
  array[0]->isDrv = false;
  array[0]->textureType = hipTextureType2D;
  void **ptr = &array[0]->data;

  size_t size = width;
  if (height > 0) {
    size = size * height;
  }
  const size_t allocSize = size * ((desc->x + desc->y + desc->z + desc->w) / 8);

  void *retval = cont->allocate(allocSize);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  LZ_CATCH
  RETURN(hipSuccess);
}

hipError_t hipArrayCreate(hipArray **array,
                          const HIP_ARRAY_DESCRIPTOR *pAllocateArray) {
  HIPLZ_INIT();

  LZ_TRY
  ERROR_IF((pAllocateArray->width == 0), hipErrorInvalidValue);

  auto cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidContext);

  *array = new hipArray;
  ERROR_IF((*array == nullptr), hipErrorOutOfMemory);

  array[0]->drvDesc = *pAllocateArray;
  array[0]->width = pAllocateArray->width;
  array[0]->height = pAllocateArray->height;
  array[0]->isDrv = true;
  array[0]->textureType = hipTextureType2D;
  void **ptr = &array[0]->data;

  size_t size = pAllocateArray->width;
  if (pAllocateArray->height > 0) {
    size = size * pAllocateArray->height;
  }
  size_t allocSize = 0;
  switch (pAllocateArray->format) {
  case HIP_AD_FORMAT_UNSIGNED_INT8:
    allocSize = size * sizeof(uint8_t);
    break;
  case HIP_AD_FORMAT_UNSIGNED_INT16:
    allocSize = size * sizeof(uint16_t);
    break;
  case HIP_AD_FORMAT_UNSIGNED_INT32:
    allocSize = size * sizeof(uint32_t);
    break;
  case HIP_AD_FORMAT_SIGNED_INT8:
    allocSize = size * sizeof(int8_t);
    break;
  case HIP_AD_FORMAT_SIGNED_INT16:
    allocSize = size * sizeof(int16_t);
    break;
  case HIP_AD_FORMAT_SIGNED_INT32:
    allocSize = size * sizeof(int32_t);
    break;
  case HIP_AD_FORMAT_HALF:
    allocSize = size * sizeof(int16_t);
    break;
  case HIP_AD_FORMAT_FLOAT:
    allocSize = size * sizeof(float);
    break;
  default:
    allocSize = size;
    break;
  }

  void *retval = cont->allocate(allocSize);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  LZ_CATCH
  RETURN(hipSuccess);
}

hipError_t hipFreeArray(hipArray *array) {
  HIPLZ_INIT();

  ERROR_IF((array == nullptr), hipErrorInvalidValue);

  assert(array->data != nullptr);

  hipError_t e = hipFree(array->data);

  delete array;

  return e;
}

hipError_t hipMalloc3D(hipPitchedPtr *pitchedDevPtr, hipExtent extent) {
  HIPLZ_INIT();

  LZ_TRY
  ERROR_IF((extent.width == 0 || extent.height == 0), hipErrorInvalidValue);
  ERROR_IF((pitchedDevPtr == nullptr), hipErrorInvalidValue);

  size_t pitch;

  hipError_t hip_status = hipMallocPitch3D(
      &pitchedDevPtr->ptr, &pitch, extent.width, extent.height, extent.depth);

  if (hip_status == hipSuccess) {
    pitchedDevPtr->pitch = pitch;
    pitchedDevPtr->xsize = extent.width;
    pitchedDevPtr->ysize = extent.height;
  }
  RETURN(hip_status);
  LZ_CATCH
}

hipError_t hipMemGetInfo(size_t *free, size_t *total) {
  HIPLZ_INIT();


  ERROR_IF((total == nullptr || free == nullptr), hipErrorInvalidValue);

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  auto device = cont->GetDevice();
  *total = device->getGlobalMemSize();
  assert (device->getGlobalMemSize() > device->getUsedGlobalMem() );
  *free =  device->getGlobalMemSize() - device->getUsedGlobalMem();

  RETURN(hipSuccess);
}

hipError_t hipMemPtrGetInfo(void *ptr, size_t *size) {
  HIPLZ_INIT();

  LZ_TRY
  ERROR_IF((ptr == nullptr || size == nullptr), hipErrorInvalidValue);

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if (cont->getPointerSize(ptr, size))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
  LZ_CATCH
}

/********************************************************************/

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream) {
  HIPLZ_INIT();

  LZ_TRY
  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  /*
  if ((kind == hipMemcpyDeviceToDevice) || (kind == hipMemcpyDeviceToHost)) {
    if (!cont->hasPointer(src))
      RETURN(hipErrorInvalidDevicePointer);
  }

  if ((kind == hipMemcpyDeviceToDevice) || (kind == hipMemcpyHostToDevice)) {
    if (!cont->hasPointer(dst))
      RETURN(hipErrorInvalidDevicePointer);
  }*/

  if (kind == hipMemcpyHostToHost) {
    memcpy(dst, src, sizeBytes);
    RETURN(hipSuccess);
  } else {
    RETURN(cont->memCopyAsync(dst, src, sizeBytes, stream));
  }
  LZ_CATCH
}

hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes,
                     hipMemcpyKind kind) {
  HIPLZ_INIT();

  LZ_TRY
  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if (kind == hipMemcpyHostToHost) {
    memcpy(dst, src, sizeBytes);
    RETURN(hipSuccess);
  } else {
    RETURN(cont->memCopy(dst, src, sizeBytes, nullptr));
  }
  LZ_CATCH
  // ze_result_t status = zeCommandQueueSynchronize(cont->hQueue, UINT64_MAX);
  // if (status != ZE_RESULT_SUCCESS) {
  // 	  throw InvalidLevel0Initialization("HipLZ zeCommandQueueSynchronize FAILED with return code " + std::to_string(status));
  // }
  // RETURN(hipSuccess);
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src,
                              size_t sizeBytes, hipStream_t stream) {
  return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, stream);
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src,
                         size_t sizeBytes) {
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToDevice);
}

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void *src, size_t sizeBytes,
                              hipStream_t stream) {
  return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyHostToDevice, stream);
}

hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void *src, size_t sizeBytes) {
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyHostToDevice);
}

hipError_t hipMemcpyDtoHAsync(void *dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream) {
  return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToHost, stream);
}

hipError_t hipMemcpyDtoH(void *dst, hipDeviceptr_t src, size_t sizeBytes) {
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToHost);
}

/********************************************************************/

hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count,
                             hipStream_t stream) {
  HIPLZ_INIT();

  LZ_TRY
  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  RETURN(cont->memFillAsync(dst, 4 * count, &value, 4, stream));
  LZ_CATCH
}

hipError_t hipMemsetD32(hipDeviceptr_t dst, int value, size_t count) {
  HIPLZ_INIT();

  LZ_TRY
  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  RETURN(cont->memFill(dst, 4 * count, &value, 4));
  LZ_CATCH
}

hipError_t hipMemset2DAsync(void *dst, size_t pitch, int value, size_t width,
                            size_t height, hipStream_t stream) {

  size_t sizeBytes = pitch * height;
  return hipMemsetAsync(dst, value, sizeBytes, stream);
}

hipError_t hipMemset2D(void *dst, size_t pitch, int value, size_t width,
                       size_t height) {

  size_t sizeBytes = pitch * height;
  return hipMemset(dst, value, sizeBytes);
}

hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value,
                            hipExtent extent, hipStream_t stream) {

  size_t sizeBytes = pitchedDevPtr.pitch * extent.height * extent.depth;
  return hipMemsetAsync(pitchedDevPtr.ptr, value, sizeBytes, stream);
}

hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value,
                       hipExtent extent) {

  size_t sizeBytes = pitchedDevPtr.pitch * extent.height * extent.depth;
  return hipMemset(pitchedDevPtr.ptr, value, sizeBytes);
}

hipError_t hipMemsetAsync(void *dst, int value, size_t sizeBytes,
                          hipStream_t stream) {
  HIPLZ_INIT();

  LZ_TRY
  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  char c_value = value;
  RETURN(cont->memFillAsync(dst, sizeBytes, &c_value, 1, stream));
  LZ_CATCH
}

hipError_t hipMemset(void *dst, int value, size_t sizeBytes) {
  HIPLZ_INIT();

  LZ_TRY
  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  char c_value = value;
  RETURN(cont->memFill(dst, sizeBytes, &c_value, 1));
  LZ_CATCH
}

hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value,
                       size_t sizeBytes) {
  return hipMemset(dest, value, sizeBytes);
}

/********************************************************************/

hipError_t hipMemcpyParam2D(const hip_Memcpy2D *pCopy) {
  ERROR_IF((pCopy == nullptr), hipErrorInvalidValue);

  return hipMemcpy2D(pCopy->dstArray->data, pCopy->widthInBytes, pCopy->srcHost,
                     pCopy->srcPitch, pCopy->widthInBytes, pCopy->height,
                     hipMemcpyDefault);
}

hipError_t hipMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                            size_t spitch, size_t width, size_t height,
                            hipMemcpyKind kind, hipStream_t stream) {
  HIPLZ_INIT();

  if (spitch == 0)
    spitch = width;
  if (dpitch == 0)
    dpitch = width;

  if (spitch == 0 || dpitch == 0)
    RETURN(hipErrorInvalidValue);

  for (size_t i = 0; i < height; ++i) {
    if (hipMemcpyAsync(dst, src, width, kind, stream) != hipSuccess)
      RETURN(hipErrorLaunchFailure);
    src = (char *)src + spitch;
    dst = (char *)dst + dpitch;
  }
  RETURN(hipSuccess);
}

hipError_t hipMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                       size_t width, size_t height, hipMemcpyKind kind) {
  HIPLZ_INIT();

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  hipError_t e = hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind,
                                  cont->getDefaultQueue());
  if (e != hipSuccess)
    return e;

  cont->getDefaultQueue()->finish();
  RETURN(hipSuccess);
}

hipError_t hipMemcpy2DToArray(hipArray *dst, size_t wOffset, size_t hOffset,
                              const void *src, size_t spitch, size_t width,
                              size_t height, hipMemcpyKind kind) {
  HIPLZ_INIT();

  size_t byteSize;
  if (dst) {
    switch (dst[0].desc.f) {
    case hipChannelFormatKindSigned:
      byteSize = sizeof(int);
      break;
    case hipChannelFormatKindUnsigned:
      byteSize = sizeof(unsigned int);
      break;
    case hipChannelFormatKindFloat:
      byteSize = sizeof(float);
      break;
    case hipChannelFormatKindNone:
      byteSize = sizeof(size_t);
      break;
    }
  } else {
    RETURN(hipErrorUnknown);
  }

  if ((wOffset + width > (dst->width * byteSize)) || width > spitch) {
    RETURN(hipErrorInvalidValue);
  }

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  size_t src_w = spitch;
  size_t dst_w = (dst->width) * byteSize;

  for (size_t i = 0; i < height; ++i) {
    void *dst_p = ((unsigned char *)dst->data + i * dst_w);
    void *src_p = ((unsigned char *)src + i * src_w);
    if (hipMemcpyAsync(dst_p, src_p, width, kind, cont->getDefaultQueue()) != hipSuccess)
      RETURN(hipErrorLaunchFailure);
  }

  cont->getDefaultQueue()->finish();
  RETURN(hipSuccess);
}

hipError_t hipMemcpyToArray(hipArray *dst, size_t wOffset, size_t hOffset,
                            const void *src, size_t count, hipMemcpyKind kind) {

  void *dst_p = (unsigned char *)dst->data + wOffset;
  return hipMemcpy(dst_p, src, count, kind);
}

hipError_t hipMemcpyFromArray(void *dst, hipArray_const_t srcArray,
                              size_t wOffset, size_t hOffset, size_t count,
                              hipMemcpyKind kind) {

  void *src_p = (unsigned char *)srcArray->data + wOffset;
  return hipMemcpy(dst, src_p, count, kind);
}

hipError_t hipMemcpyAtoH(void *dst, hipArray *srcArray, size_t srcOffset,
                         size_t count) {
  return hipMemcpy((char *)dst, (char *)srcArray->data + srcOffset, count,
                   hipMemcpyDeviceToHost);
}

hipError_t hipMemcpyHtoA(hipArray *dstArray, size_t dstOffset,
                         const void *srcHost, size_t count) {
  return hipMemcpy((char *)dstArray->data + dstOffset, srcHost, count,
                   hipMemcpyHostToDevice);
}

hipError_t hipMemcpy3D(const struct hipMemcpy3DParms *p) {
  HIPLZ_INIT();

  ERROR_IF((p == nullptr), hipErrorInvalidValue);

  size_t byteSize;
  size_t depth;
  size_t height;
  size_t widthInBytes;
  size_t srcPitch;
  size_t dstPitch;
  void *srcPtr;
  void *dstPtr;
  size_t ySize;

  if (p->dstArray != nullptr) {
    if (p->dstArray->isDrv == false) {
      switch (p->dstArray->desc.f) {
      case hipChannelFormatKindSigned:
        byteSize = sizeof(int);
        break;
      case hipChannelFormatKindUnsigned:
        byteSize = sizeof(unsigned int);
        break;
      case hipChannelFormatKindFloat:
        byteSize = sizeof(float);
        break;
      case hipChannelFormatKindNone:
        byteSize = sizeof(size_t);
        break;
      }
      depth = p->extent.depth;
      height = p->extent.height;
      widthInBytes = p->extent.width * byteSize;
      srcPitch = p->srcPtr.pitch;
      srcPtr = p->srcPtr.ptr;
      ySize = p->srcPtr.ysize;
      dstPitch = p->dstArray->width * byteSize;
      dstPtr = p->dstArray->data;
    } else {
      depth = p->Depth;
      height = p->Height;
      widthInBytes = p->WidthInBytes;
      dstPitch = p->dstArray->width * 4;
      srcPitch = p->srcPitch;
      srcPtr = (void *)p->srcHost;
      ySize = p->srcHeight;
      dstPtr = p->dstArray->data;
    }
  } else {
    // Non array destination
    depth = p->extent.depth;
    height = p->extent.height;
    widthInBytes = p->extent.width;
    srcPitch = p->srcPtr.pitch;
    srcPtr = p->srcPtr.ptr;
    dstPtr = p->dstPtr.ptr;
    ySize = p->srcPtr.ysize;
    dstPitch = p->dstPtr.pitch;
  }

  if ((widthInBytes == dstPitch) && (widthInBytes == srcPitch)) {
    return hipMemcpy((void *)dstPtr, (void *)srcPtr,
                     widthInBytes * height * depth, p->kind);
  } else {

    LZContext *cont = getTlsDefaultLzCtx();
    ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

    for (size_t i = 0; i < depth; i++) {
      for (size_t j = 0; j < height; j++) {

        unsigned char *src =
            (unsigned char *)srcPtr + i * ySize * srcPitch + j * srcPitch;
        unsigned char *dst =
            (unsigned char *)dstPtr + i * height * dstPitch + j * dstPitch;
        if (hipMemcpyAsync(dst, src, widthInBytes, p->kind, cont->getDefaultQueue()) != hipSuccess)
	  RETURN(hipErrorLaunchFailure);
      }
    }

    cont->getDefaultQueue()->finish();
    RETURN(hipSuccess);
  }
}

/********************************************************************/

hipError_t hipInit(unsigned int flags) {
  HIPLZ_INIT();

  RETURN(hipSuccess);
}

hipError_t hipInitFromOutside(void* driverPtr, void* devicePtr, void* ctxPtr, void* queuePtr) {
  LZ_TRY
  InitializeHipLZFromOutside((ze_driver_handle_t)driverPtr, (ze_device_handle_t)devicePtr,
                             (ze_context_handle_t)ctxPtr, (ze_command_queue_handle_t)queuePtr);
  LZ_CATCH
  RETURN(hipSuccess);
}


/********************************************************************/

hipError_t hipFuncGetAttributes(hipFuncAttributes *attr, const void *func) {
  logError("hipFuncGetAttributes not implemented \n");
  RETURN(hipErrorInvalidValue);
}

hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes,
                              hipModule_t hmod, const char *name) {
  HIPLZ_INIT();

  ERROR_IF((!dptr || !bytes || !name || !hmod), hipErrorInvalidValue);
  ERROR_IF((!hmod->symbolSupported()), hipErrorNotSupported);
  ERROR_IF((!hmod->getSymbolAddressSize(name, dptr, bytes)), hipErrorInvalidSymbol);

  RETURN(hipSuccess);
}

hipError_t hipGetSymbolAddress(void **devPtr, const void *symbol) {
  HIPLZ_INIT();

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  size_t bytes;
  ERROR_IF((!cont->getSymbolAddressSize((const char* )symbol, (hipDeviceptr_t *)devPtr, &bytes)),
	   hipErrorInvalidSymbol);

  RETURN(hipSuccess);
}

hipError_t hipGetSymbolSize(size_t *size, const void *symbol) {
  HIPLZ_INIT();

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  hipDeviceptr_t devPtr;
  ERROR_IF((!cont->getSymbolAddressSize((const char* )symbol, &devPtr, size)), hipErrorInvalidSymbol);

  RETURN(hipSuccess);
}

hipError_t hipMemcpyToSymbol(const void *symbol, const void *src, size_t sizeBytes, size_t offset,
                             hipMemcpyKind kind) {
  HIPLZ_INIT();

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  hipError_t e = hipMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, kind, cont->getDefaultQueue());
  if (e != hipSuccess)
    RETURN(e);

  cont->getDefaultQueue()->finish();
  
  RETURN(hipSuccess);
}

hipError_t hipMemcpyToSymbolAsync(const void *symbol, const void *src, size_t sizeBytes, size_t offset,
                                  hipMemcpyKind kind, hipStream_t stream) {
  HIPLZ_INIT();

  void *symPtr = NULL;
  size_t symSize = 0;
  ClContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  ERROR_IF((!cont->getSymbolAddressSize((const char*)symbol, &symPtr, &symSize)), hipErrorInvalidSymbol);
  RETURN(hipMemcpyAsync((void *)((intptr_t)symPtr + offset), src, sizeBytes, kind, stream));
}

hipError_t hipMemcpyFromSymbol(void *dst, const void *symbol, size_t sizeBytes, size_t offset,
			       hipMemcpyKind kind) {
  HIPLZ_INIT();

  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  hipError_t e = hipMemcpyFromSymbolAsync(dst, symbol, sizeBytes, offset, kind, cont->getDefaultQueue());
  if (e != hipSuccess)
    RETURN(e);

  cont->getDefaultQueue()->finish();
  RETURN(hipSuccess);
}

hipError_t hipMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind, hipStream_t stream) {
  HIPLZ_INIT();

  void *symPtr;
  size_t symSize;
  LZContext *cont = getTlsDefaultLzCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  ERROR_IF((!cont->getSymbolAddressSize((const char*)symbol, &symPtr, &symSize)), hipErrorInvalidSymbol);
  RETURN(hipMemcpyAsync(dst, (void *)((intptr_t)symPtr + offset), sizeBytes, kind, stream));
}

hipError_t hipModuleLoadData(hipModule_t *module, const void *image) {
  logError("hipModuleLoadData not implemented\n");
  return hipErrorNoBinaryForGpu;
}

hipError_t hipModuleLoadDataEx(hipModule_t *module, const void *image,
                               unsigned int numOptions, hipJitOption *options,
                               void **optionValues) {
  return hipModuleLoadData(module, image);
}

// For old kernel HIP launch API.
hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                            hipStream_t stream) {
  HIPLZ_INIT();

  LZ_TRY

  LZContext* lzCtx = getTlsDefaultLzCtx();
  ERROR_IF((lzCtx == nullptr), hipErrorInvalidDevice);
  RETURN(lzCtx->configureCall(gridDim, blockDim, sharedMem, stream));

  LZ_CATCH
}

// For old kernel HIP launch API.
hipError_t hipSetupArgument(const void *arg, size_t size, size_t offset) {
  HIPLZ_INIT();

  LZ_TRY 

  // Try for HipLZ kernel at first
  LZContext* lzCtx = getTlsDefaultLzCtx();
  ERROR_IF((lzCtx == nullptr), hipErrorInvalidDevice);  
  RETURN(lzCtx->setArg(arg, size, offset));

  LZ_CATCH
}

// For old kernel HIP launch API.
hipError_t hipLaunchByPtr(const void *hostFunction) {
  HIPLZ_INIT();

  LZ_TRY

  // Try for HipLZ kernel at first
  LZContext* lzCtx = getTlsDefaultLzCtx();
  ERROR_IF((lzCtx == nullptr), hipErrorInvalidDevice);
  if (lzCtx->launchHostFunc(hostFunction)) 
    RETURN(hipSuccess);
  else
    RETURN(hipErrorLaunchFailure);

  LZ_CATCH
}

// For new kernel HIP launch API.
hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                      size_t sharedMem, hipStream_t stream) {
  HIPLZ_INIT();
  LZ_TRY
  LZContext* lzCtx = getTlsDefaultLzCtx();
  ERROR_IF((lzCtx == nullptr), hipErrorInvalidDevice);
  RETURN(lzCtx->configureCall(gridDim, blockDim, sharedMem, stream));
  LZ_CATCH
}

// For new kernel HIP launch API.
hipError_t __hipPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                     size_t *sharedMem, hipStream_t *stream) {
  HIPLZ_INIT();
  LZ_TRY
  LZContext* lzCtx = getTlsDefaultLzCtx();
  ERROR_IF((lzCtx == nullptr), hipErrorInvalidDevice);
  RETURN(lzCtx->popCallConfiguration(gridDim, blockDim, sharedMem, stream));
  LZ_CATCH
}

// For new kernel HIP launch API.
hipError_t hipLaunchKernel(const void *hostFunction, dim3 gridDim,
                           dim3 blockDim, void **args, size_t sharedMem,
                           hipStream_t stream) {
  HIPLZ_INIT();
  LZ_TRY
  LZContext* lzCtx = getTlsDefaultLzCtx();
  ERROR_IF((lzCtx == nullptr), hipErrorInvalidDevice);
  if (lzCtx->launchHostFunc(hostFunction, gridDim, blockDim, args,
                            sharedMem, stream))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorLaunchFailure);
  LZ_CATCH
}


/********************************************************************/
hipError_t hipCreateTextureObject(hipTextureObject_t* texObj, hipResourceDesc* resDesc,
				  hipTextureDesc* texDesc, void* opt) {
  HIPLZ_INIT();

  LZ_TRY
  LZContext* lzCtx = getTlsDefaultLzCtx();
  ERROR_IF((lzCtx == nullptr), hipErrorInvalidDevice);
  hipTextureObject_t retObj = lzCtx->createImage(resDesc, texDesc);
  if (retObj != nullptr) {
    * texObj = retObj;
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorLaunchFailure);
  LZ_CATCH
}

/********************************************************************/
hipError_t hipModuleLoad(hipModule_t *module, const char *fname) {
  HIPLZ_INIT();

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  std::ifstream file(fname, std::ios::in | std::ios::binary | std::ios::ate);
  ERROR_IF((file.fail()), hipErrorFileNotFound);

  size_t size = file.tellg();
  char *memblock = new char[size];
  file.seekg(0, std::ios::beg);
  file.read(memblock, size);
  file.close();
  std::string content(memblock, size);
  delete[] memblock;

  RETURN(hipErrorNotSupported);
  /* TODO: fix this by implement hipModule_t related operations via Level-0
   *module = cont->createProgram(content);
  if (*module == nullptr)
    RETURN(hipErrorInvalidValue);
  else
    RETURN(hipSuccess);
    */
}

hipError_t hipModuleUnload(hipModule_t module) {
  HIPLZ_INIT();

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  RETURN(hipErrorNotSupported);
  // TODO: fix this by implement hipModule_t related operations via Level-0
  // RETURN(cont->destroyProgram(module));
}

hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module,
                                const char *kname) {
  HIPLZ_INIT();

  ClProgram *p = (ClProgram *)module;
  ClKernel *k = p->getKernel(kname);

  ERROR_IF((k == nullptr), hipErrorInvalidDeviceFunction);

  *function = k;
  RETURN(hipSuccess);
}

hipError_t hipModuleLaunchKernel(hipFunction_t k, unsigned int gridDimX,
                                 unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ,
                                 unsigned int sharedMemBytes,
                                 hipStream_t stream, void **kernelParams,
                                 void **extra) {
  HIPLZ_INIT();

  logDebug("hipModuleLaunchKernel\n");

  if (sharedMemBytes > 0) {
    logError("Dynamic shared memory isn't supported ATM\n");
    RETURN(hipErrorLaunchFailure);
  }

  ClContext *cont = getTlsDefaultCtx();
  ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  if (kernelParams == nullptr && extra == nullptr) {
    logError("either kernelParams or extra is required!\n");
    RETURN(hipErrorLaunchFailure);
  }

  dim3 grid(gridDimX, gridDimY, gridDimZ);
  dim3 block(blockDimX, blockDimY, blockDimZ);

  if (kernelParams)
    RETURN(cont->launchWithKernelParams(grid, block, sharedMemBytes, stream,
                                        kernelParams, k));
  else
    RETURN(cont->launchWithExtraParams(grid, block, sharedMemBytes, stream,
                                       extra, k));
}

/*******************************************************************************/

#include "hip/hip_fatbin.h"

#define SPIR_TRIPLE "hip-spir64-unknown-unknown"

static unsigned binaries_loaded = 0;

extern "C" void **__hipRegisterFatBinary(const void *data) {
  InitializeOpenCL();
  // Here we do not initialize HipLZ but put fat binary into a global temproary storage
  // xxx InitializeHipLZ();
  
  const __CudaFatBinaryWrapper *fbwrapper =
      reinterpret_cast<const __CudaFatBinaryWrapper *>(data);
  if (fbwrapper->magic != __hipFatMAGIC2 || fbwrapper->version != 1) {
    logCritical("The given object is not hipFatBinary !\n");
    std::abort();
  }

  const __ClangOffloadBundleHeader *header = fbwrapper->binary;
  std::string magic(reinterpret_cast<const char *>(header),
                    sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC)) {
    logCritical("The bundled binaries are not Clang bundled "
                "(CLANG_OFFLOAD_BUNDLER_MAGIC is missing)\n");
    std::abort();
  }

  std::string *module = new std::string;
  if (!module) {
    logCritical("Failed to allocate memory\n");
    std::abort();
  }

  const __ClangOffloadBundleDesc *desc = &header->desc[0];
  bool found = false;

  for (uint64_t i = 0; i < header->numBundles;
       ++i, desc = reinterpret_cast<const __ClangOffloadBundleDesc *>(
                reinterpret_cast<uintptr_t>(&desc->triple[0]) +
                desc->tripleSize)) {

    std::string triple{&desc->triple[0], sizeof(SPIR_TRIPLE) - 1};
    logDebug("Triple of bundle {} is: {}\n", i, triple);

    if (triple.compare(SPIR_TRIPLE) == 0) {
      found = true;
      break;
    } else {
      logDebug("not a SPIR triple, ignoring\n");
      continue;
    }
  }

  if (!found) {
    logDebug("Didn't find any suitable compiled binary!\n");
    std::abort();
  }

  const char *string_data = reinterpret_cast<const char *>(
      reinterpret_cast<uintptr_t>(header) + (uintptr_t)desc->offset);
  size_t string_size = desc->size;
  module->assign(string_data, string_size);

  logDebug("Register module: {} \n", (void *)module);

  for (size_t deviceId = 0; deviceId < NumDevices; ++deviceId) {
    CLDeviceById(deviceId).registerModule(module);
  }

  // Put HipLZ module into global fat binary storage
  // for (size_t driverId = 0; driverId < NumLZDrivers; ++ driverId) {
  //   LZDriver::HipLZDriverById(driverId).registerModule(module);
  // }
  LZDriver::FatBinModules.push_back(module);

  ++ binaries_loaded;
  logDebug("__hipRegisterFatBinary {}\n", binaries_loaded);

  return (void **)module;
}

extern "C" void __hipUnregisterFatBinary(void *data) {
  std::string *module = reinterpret_cast<std::string *>(data);

  logDebug("Unregister module: {} \n", (void *)module);
  for (size_t deviceId = 0; deviceId < NumDevices; ++deviceId) {
    CLDeviceById(deviceId).unregisterModule(module);
  }

  --binaries_loaded;
  logDebug("__hipUnRegisterFatBinary {}\n", binaries_loaded);

  if (binaries_loaded == 0) {
    UnInitializeOpenCL();
  }

  delete module;
}


extern "C" void __hipRegisterFunction(void **data, const void *hostFunction,
                                      char *deviceFunction,
                                      const char *deviceName,
                                      unsigned int threadLimit, void *tid,
                                      void *bid, dim3 *blockDim, dim3 *gridDim,
                                      int *wSize) {
  InitializeOpenCL();
  
  std::string devFunc = deviceFunction;
  // Initialize HipLZ here (this may not be the 1st place, but the intiialization process is protected via single-execution)
  // Here we do not initialize HipLZ, but store the function informqtion into a temproary storage
  // xxx InitializeHipLZ();

  // std::cout << "module data: " << (unsigned long)data << std::endl;
  std::string *module = reinterpret_cast<std::string *>(data);
  logDebug("RegisterFunction on module {}\n", (void *)module);

#if 0
  for (size_t deviceId = 0; deviceId < NumDevices; ++deviceId) {

    if (CLDeviceById(deviceId).registerFunction(module, hostFunction,
                                                deviceName)) {
      logDebug("__hipRegisterFunction: kernel {} found\n", deviceName);
    } else {
      logCritical("__hipRegisterFunction can NOT find kernel: {} \n",
                  deviceName);
      std::abort();
    }
  }
#endif

#if 0
  for (size_t driverId = 0; driverId < NumLZDrivers; ++ driverId) {
    if (LZDriver::HipLZDriverById(driverId).registerFunction(module, hostFunction, deviceName)) {
      logDebug("__hipRegisterFunction: HipLZ kernel {} found\n", deviceName);
    } else {
      logCritical("__hipRegisterFunction can NOT find HipLZ kernel: {} \n", deviceName);
      std::abort();
    }
  }
#endif
  
  // Put the function information into a temproary storage
  LZDriver::RegFunctions.push_back(std::make_tuple(module, hostFunction, deviceName));
}

extern "C" void __hipRegisterVar(void** data, // std::vector<hipModule_t> *modules,
                                 char *hostVar, char *deviceVar,
                                 const char *deviceName, int ext, int size,
                                 int constant, int global) {
  // logError("__hipRegisterVar not implemented yet\n");
  InitializeOpenCL();

  // Initialize HipLZ here (this may not be the 1st place, but the intiialization process is protected via single-execution
  // Here we do not initialize HipLZ, but store the global variable informqtion into a temproary storage
  // xxx InitializeHipLZ();

  std::string *module = reinterpret_cast<std::string *>(data);
  logDebug("RegisterVar on module {}\n", (void *)module);

  /* xxx
  for (size_t driverId = 0; driverId < NumLZDrivers; ++ driverId) {
    if (LZDriver::HipLZDriverById(driverId).registerVar(module, hostVar, deviceName, size)) {
      logDebug("__hipRegisterVar: variable {} found\n", deviceName);
    } else {
      logError("__hipRegisterVar could not find: {}\n", deviceName);
    }
    }*/
  
  // Put the global variable information into a temproary storage
  // LZDriver::GlobalVars.push_back(std::make_tuple(module, hostVar, deviceName, size));
}

