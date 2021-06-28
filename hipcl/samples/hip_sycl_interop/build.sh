# Pre-requirement
# export PATH=${ONEAPI_COMPILER_PREFIX}/bin/:${PATH}
# export LD_LIBRARY_PATH=${ONEAPI_COMPILER_PREFIX}/lib:${ONEAPI_COMPILER_PREFIX}/compiler/lib:${LD_LIBRARY_PATH}
# The ONEAPI_COMPILER_PREFIX is the intallation of DPC++, e.g. ~/intel/oneapi/compiler/latest/linux
# Set the HIPLZ_INSTALL_PREFIX as HipLZ installation since the dynamic shared library that encapsulates HIP matrix muplication was pre-built and installed at ${HIPLZ_INSTALL_PREFIX}/lib
clang++ onemkl_gemm_wrapper.cpp -DMKL_ILP64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -o libonemkl_gemm_wrapper.so -fsycl -lze_loader -shared -fPIC
