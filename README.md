# anl-gt-gpu

This repository contains the modified HIPCL(https://github.com/cpc/hipcl) that retargets from a vendor's OpenCL runtime to Intel Level-Zero runtime. To build this version, a new CMake parameter is introduced that indicates where the Level-Zero runtime was installed. Here are the steps:
```
cd ${YOUR_HIPCL_SRC}
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${YOUR_HIPCL_INSTALL_FOLDER} 
      -DCMAKE_CXX_COMPILER=${YOUR_LLVM_HIPCL_CLANG_INSTALL}/bin/clang++ 
      -DCMAKE_C_COMPILER=${YOUR_LLVM_HIPCL_CLANG_INSTALL}/bin/clang 
      -DOpenCL_INCLUDE_DIR=${OPENCL_INSTALL_INCLUDE} 
      -DOpenCL_LIBRARY=${OPENCL_INSTALL_LIB} 
      -DLevel_0_INCLUDE_DIR=${LEVEL_0_INSTALL_HEADERS}/level_zero
      -DLevel_0_LIBRARIES=${LEVEL_0_INSTALL}/libze_loader.so .. 
```
