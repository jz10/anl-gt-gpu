// Check maxGridSize properties are positive.
#include "hip/hip_runtime.h"
#include <iostream>

int main() {
  hipDeviceProp_t deviceProps;
  auto error = hipGetDeviceProperties(&deviceProps, 0);
  if (error != hipSuccess) {
    std::cout << "HIP ERROR: '" << hipGetErrorString(error) << "'\n";
    return 1;
  }
  for (int i = 0; i < 3; i++) {
    auto size = deviceProps.maxGridSize[i];
    if (size < 0) {
      std::cout << "FAIL: maxGridSize[" << i << "] is negative (" << size
                << ")!\n";
      return 2;
    }
  }
  std::cout << "PASSED\n";
  return 0;
}
