#include <iostream>
#include <random>
#include <functional>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cfloat>
  
// hip header file
#include "hip/hip_runtime.h"

#include "sycl_hiplz_interop.h"
 
#define MM_SHARED

#define WIDTH 1024
// the required shared memory is (2 * 4 * THREADS_PER_BLOCK * THREADS_PER_BLOCK) bytes
#define THREADS_PER_BLOCK 16

#define ERR_CHECK_2 \
  do { \
  err = hipGetLastError(); \
    if (err != hipSuccess) { \
      std::cerr << "HIP API error\n"; \
      return -1; \
    } \
  } while (0)


#define ERR_CHECK \
  do { \
    if (err != hipSuccess) { \
      std::cerr << "HIP API error\n"; \
      return -1; \
    } \
  } while (0)


__global__ void gpuMatrixMul(const float * __restrict A,
			     const float * __restrict B,
			     float * __restrict C,
			     uint M, uint N, uint K) {
  // Thread identifiers
  const uint globalRow = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; // Row ID of C (0..M)
  const uint globalCol = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y; // Col ID of C (0..N)

  // Compute a single element (loop over K)
  float acc = 0.0f;
  for (uint k = 0; k < K; k++) {
    acc += A[k * M + globalRow] * B[globalCol * K + k];
  }

  // Store the result
  C[globalCol * M + globalRow] = acc;
}

/*****************************************************************************/

int hiplzInit(void* driverPtr, void* deviePtr, void* contextPtr, void* queuePtr) {
  hipError_t err = hipInitFromOutside(driverPtr, deviePtr, contextPtr, queuePtr);
  ERR_CHECK;
  
  return 0;
}

int hipMatrixMultiplicationTest(const float* A, const float* B, float* C, int M, int N) {

  hipError_t err;

  hipDeviceProp_t devProp;
  err = hipGetDeviceProperties(&devProp, 0);
  ERR_CHECK;
  
  std::cout << "RunTestMM 3 " << std::endl;

  // Lauching kernel from host
  hipLaunchKernelGGL(gpuMatrixMul,
		     dim3(WIDTH / THREADS_PER_BLOCK, WIDTH / THREADS_PER_BLOCK),
		     dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK),
		     0, 0,
		     A, B, C, M, N, M);
  ERR_CHECK_2;

  std::cout << "RunTestMM 4 " << std::endl;
  // Memory transfer from device to host
  // err = hipMemcpy(MultiplyMatrix, gpuMultiplyMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);
  ERR_CHECK;
  
  err = hipDeviceSynchronize();
  ERR_CHECK;
  
  return 0;
}
