#include "hip/hip_runtime.h"

// STL classes
#include <exception>
#include <iostream>

#include <vector>

#include "hiplz_sycl_interop.h"

using namespace std;

const int WIDTH = 10;

// CPU implementation of matrix transpose
void matrixMultiplyCPUReference(const float * __restrict A,
                                const float * __restrict B,
                                float * __restrict C) {
  for (uint i = 0; i < WIDTH; i++) {
    for (uint j = 0; j < WIDTH; j++) {
      float acc = 0.0f;
      for (uint k = 0; k < WIDTH; k++) {
	acc += B[i*WIDTH + k] * A[k*WIDTH + j];
      }
      C[i*WIDTH + j] = acc;
    }
  }
}

int main() { 
  vector<float> A(100, 1);
  vector<float> B(100, 2);
  vector<float> C(100, 0);
  int m, n, k;
  m = n = k = WIDTH;
  int ldA, ldB, ldC;
  ldA = ldB = ldC = WIDTH;
  double alpha = 1.0;
  double beta  = 1.1;
  
  // Create HipLZ stream  
  hipStream_t stream = nullptr;
  hipStreamCreate(&stream);

  unsigned long nativeHandlers[4];
  int numItems = 0;
  hipStreamNativeInfo(stream, nativeHandlers, &numItems);
  
  // Invoke oneMKL GEEM
  oneMKLGemmTest(nativeHandlers, A.data(), B.data(), C.data(), m, m, k, ldA, ldB, ldC, alpha, beta);
  
  return 0;
}
