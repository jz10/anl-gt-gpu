# LLVM requires CMake >= 3.4.3
cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
cmake_policy(VERSION 3.4)

set(HIPLZ_VERSION_MAJOR 0)
set(HIPLZ_VERSION_MINOR 9)
set(HIPLZ_VERSION_PATCH 0)
set(HIPLZ_VERSION_FULL "${HIPLZ_VERSION_MAJOR}.${HIPLZ_VERSION_MINOR}.${HIPLZ_VERSION_PATCH}")

set(LIB_BUILD_VERSION "${HIPLZ_VERSION_FULL}")
set(LIB_API_VERSION "${HIPLZ_VERSION_MAJOR}")

project(HIP_CL
        VERSION "${HIPLZ_VERSION_FULL}"
        DESCRIPTION "Hip implementation over OpenCL"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_UBSAN "Enable UBSanitizer" OFF)
option(ENABLE_LSAN "Enable LeakSanitizer" OFF)
set(SANITIZER_LIBS)

if(ENABLE_ASAN)
  if("${CMAKE_C_COMPILER_VERSION}" VERSION_LESS "6.0.0")
    list(APPEND SANITIZER_OPTIONS "-fsanitize=address")
  else()
    list(APPEND SANITIZER_OPTIONS "-fsanitize=address" "-fsanitize-recover=address")
  endif()
  list(APPEND SANITIZER_LIBS "asan")
endif()

if(ENABLE_LSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=leak")
  list(APPEND SANITIZER_LIBS "lsan")
endif()

if(ENABLE_TSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=thread")
  list(APPEND SANITIZER_LIBS "tsan")
endif()

if(ENABLE_UBSAN)
  list(APPEND SANITIZER_OPTIONS "-fsanitize=undefined")
  list(APPEND SANITIZER_LIBS "ubsan")
endif()

if(SANITIZER_OPTIONS)
  list(APPEND SANITIZER_OPTIONS "-fno-omit-frame-pointer")
  string(REPLACE ";" " " SANITIZER_OPTIONS_STR "${SANITIZER_OPTIONS}")
endif()

enable_testing()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/opt/hiplz" CACHE PATH "Install path prefix" FORCE)
endif()

message(STATUS "HipCL will be installed to: ${CMAKE_INSTALL_PREFIX}")

set(HIPLZ_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib" CACHE PATH "where libhiplz will be installed")
set(HIPLZ_LLVM_DIR "${CMAKE_INSTALL_PREFIX}/lib/llvm" CACHE PATH "where hip LLVM passes will be installed")
set(HIPLZ_DATA_DIR "${CMAKE_INSTALL_PREFIX}/share" CACHE PATH "where kernel library bitcode will be installed")
set(HIPLZ_INC_DIR "${CMAKE_INSTALL_PREFIX}/include" CACHE PATH "where kernel hiplz headers will be installed")
set(HIPLZ_BIN_DIR "${CMAKE_INSTALL_PREFIX}/bin" CACHE PATH "where hiplz_config will be installed")
set(HIPLZ_SAMPLE_BINDIR "${CMAKE_INSTALL_PREFIX}/bin/samples" CACHE PATH "where samples will be installed")
set(HIPLZ_CMAKE_DIR "${CMAKE_INSTALL_PREFIX}/cmake" CACHE PATH "where cmake files will be installed")

if(UNIX AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
    message(FATAL_ERROR "This project requires Clang >= 8.0")
  endif()
else()
  message(FATAL_ERROR "This project must be compiled with Clang")
endif()

message(STATUS "Using Clang: ${CMAKE_CXX_COMPILER}")
set(HIPLZ_COMPILER "${CMAKE_CXX_COMPILER}")

get_filename_component(CLANG_BIN_PATH "${CMAKE_CXX_COMPILER}" DIRECTORY)

if(NOT DEFINED LLVM_LINK)
  if(EXISTS "${CLANG_BIN_PATH}/llvm-link")
    set(LLVM_LINK "${CLANG_BIN_PATH}/llvm-link" CACHE PATH "llvm-link")
  else()
    message(FATAL_ERROR "Can't find llvm-link at ${CLANG_BIN_PATH}. Please provide CMake argument -DLLVM_LINK=<path/to/llvm-link>")
  endif()
endif()

message(STATUS "Using llvm-link: ${LLVM_LINK}")

if(NOT DEFINED LLVM_SPIRV)
  if(EXISTS "${CLANG_BIN_PATH}/llvm-spirv")
    set(LLVM_SPIRV "${CLANG_BIN_PATH}/llvm-spirv" CACHE PATH "llvm-spirv")
  else()
    message(FATAL_ERROR "Can't find llvm-spirv at ${CLANG_BIN_PATH}. Please copy llvm-spirv to ${CLANG_BIN_PATH}, Clang expects it there!")
  endif()
endif()

message(STATUS "Using llvm-spirv: ${LLVM_SPIRV}")

if(NOT DEFINED LLVM_CONFIG)
  if(EXISTS "${CLANG_BIN_PATH}/llvm-config")
    set(LLVM_CONFIG "${CLANG_BIN_PATH}/llvm-config" CACHE PATH "llvm-config")
  else()
    message(FATAL_ERROR "Can't find llvm-config at ${CLANG_BIN_PATH}. Please provide CMake argument -DLLVM_CONFIG=<path/to/llvm-config>")
  endif()
endif()

message(STATUS "Using llvm-config: ${LLVM_CONFIG}")

execute_process(COMMAND "${LLVM_CONFIG}" "--obj-root"
  RESULT_VARIABLE RES
  OUTPUT_VARIABLE CLANG_ROOT_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND "${LLVM_CONFIG}" "--version"
  RESULT_VARIABLE RES
  OUTPUT_VARIABLE LLVM_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(DEFAULT_BUILD_TYPE "Debug")
else()
  set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMake build type not set, setting to ${DEFAULT_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "CMake build type" FORCE)
endif()

message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
set(PTHREAD_LIBRARY Threads::Threads)

##########################################

add_subdirectory(lib)

# hiplz_config
add_subdirectory(bin)

# LLVM pass
add_subdirectory(llvm_passes)

##########################################

get_target_property(HIPLZ_CFLAGS_PRIV "hiplz" COMPILE_OPTIONS)
message(STATUS "hiplz private compile options: ${HIPLZ_CFLAGS_PRIV}")
get_target_property(HIPLZ_IDIR_PRIV "hiplz" INCLUDE_DIRECTORIES)
message(STATUS "hiplz private include directories: ${HIPLZ_IDIR_PRIV}")


get_target_property(HIPLZ_CFLAGS_IFC "hiplz" INTERFACE_COMPILE_OPTIONS)
message(STATUS "hiplz public compile options: ${HIPLZ_CFLAGS_IFC}")
get_target_property(HIPLZ_IDIR_IFC "hiplz" INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "hiplz public include directories: ${HIPLZ_IDIR_IFC}")

string(REPLACE "${CMAKE_INSTALL_PREFIX}" "-I${CMAKE_INSTALL_PREFIX}" HIPLZ_IDIR_IFC "${HIPLZ_IDIR_IFC}")
# HIPLZ_CXX_OPTIONS is what's reported by hiplz-config.
string(REPLACE ";" " " HIPLZ_CXX_OPTIONS "${HIPLZ_CFLAGS_IFC};${HIPLZ_IDIR_IFC}")

# The hiplz "COMPILE_OPTIONS" target properties contain some generator expressions;
# we have to get rid of those for hiplz_config.h
# this is the way Boost does it (until there's a better way)
# https://gitlab.kitware.com/cmake/cmake/issues/17984
string(REPLACE "$<BUILD_INTERFACE:" "$<0:" HIPLZ_CXX_OPTIONS "${HIPLZ_CXX_OPTIONS}")
string(REPLACE "$<INSTALL_INTERFACE:" "$<1:" HIPLZ_CXX_OPTIONS "${HIPLZ_CXX_OPTIONS}")
message(STATUS "HIPLZ_CXX_OPTIONS: ${HIPLZ_CXX_OPTIONS}")

##########################################

configure_file(
  "${CMAKE_SOURCE_DIR}/hiplz_config.h.in"
  "${CMAKE_BINARY_DIR}/hiplz_config.h.gen"
  @ONLY)

file(GENERATE
     OUTPUT "${CMAKE_BINARY_DIR}/hiplz_config.h"
     INPUT "${CMAKE_BINARY_DIR}/hiplz_config.h.gen")

install(FILES
  "${CMAKE_SOURCE_DIR}/cmake/FindHIP.cmake"
  "${CMAKE_SOURCE_DIR}/cmake/run_hipcc.cmake"
  "${CMAKE_SOURCE_DIR}/cmake/run_make2cmake.cmake"
  "${CMAKE_BINARY_DIR}/hipConfigVersion.cmake"
  "${CMAKE_BINARY_DIR}/hipConfig.cmake"
  DESTINATION "${HIPLZ_CMAKE_DIR}")

##########################################

add_subdirectory(samples)
