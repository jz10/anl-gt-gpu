# Pre-requirement
# Setup Intel runtime:
module use /soft/modulefiles/
module load intel_compute_runtime cmake

# Setup the DPC++ environment:
# export PATH=${ONEAPI_COMPILER_PREFIX}/bin/:${PATH}
# export LD_LIBRARY_PATH=${ONEAPI_COMPILER_PREFIX}/lib:${ONEAPI_COMPILER_PREFIX}/compiler/lib:${LD_LIBRARY_PATH}
# The ONEAPI_COMPILER_PREFIX is the intallation of DPC++, e.g. ~/intel/oneapi/compiler/latest/linux
#
# Or using DPC++ on JLSE:
SHARE_JY=/home/ac.jyoung/gpfs_share
module use ${SHARE_JY}/compilers/modulefiles/oneapi/2021.3.0/
module load mkl compiler

# Set the HIPLZ_INSTALL_PREFIX as HipLZ installation since the dynamic shared library that encapsulates HIP matrix muplication was pre-built and installed at ${HIPLZ_INSTALL_PREFIX}/lib
HOME_JZ=/home/ac.jzhao1
export HIPLZ_INSTALL_PREFIX=${HOME_JZ}/hipclworkspace/hipcl/install/
clang++ sycl_hiplz_interop.cpp -o sycl_hiplz_interop.exe -fsycl  -lze_loader -L${HIPLZ_INSTALL_PREFIX}/lib -lSyCL2HipLZMM -lhiplz

# Run the built test
export LD_LIBRARY_PATH=${HIPLZ_INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}
./sycl_hiplz_interop.exe
