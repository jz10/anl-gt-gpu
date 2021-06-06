add_subdirectory(bitcode)

###################################################################################

find_package(OpenCL 2.0 REQUIRED)

set(HIPLZ_SOURCES hiplz.cc clbackend.cc lzbackend.cc log.cc spirv.cc)

set_source_files_properties(${HIPLZ_SOURCES} PROPERTIES LANGUAGE CXX)

add_library(hiplz SHARED ${HIPLZ_SOURCES})

set_target_properties(hiplz PROPERTIES
                      CXX_STANDARD_REQUIRED ON
                      SOVERSION "${LIB_API_VERSION}"
                      VERSION "${LIB_BUILD_VERSION}")

target_link_libraries(hiplz ${SANITIZER_LIBS} ${PTHREAD_LIBRARY} ${OpenCL_LIBRARIES} ${Level_0_LIBRARIES})

add_dependencies("hiplz" "kernellib_bc")

if(DEBUG)
    target_compile_definitions(hiplz PRIVATE "_GLIBCXX_DEBUG")
endif()

if(LOGLEVEL)
  set(VALID_LEVELS "DEBUG;INFO;WARN;ERROR;CRITICAL;OFF")
  if(LOGLEVEL IN_LIST VALID_LEVELS)
    target_compile_definitions(hiplz PRIVATE "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${LOGLEVEL}")
  else()
    message(WARNING "Unknown loglevel: ${LOGLEVEL}, ignoring")
  endif()
endif()

target_compile_options(hiplz PRIVATE "-Wno-unused-parameter")

target_compile_options(hiplz INTERFACE "-x" "hip")

if(SANITIZER_OPTIONS)
  target_compile_options(hiplz INTERFACE ${SANITIZER_OPTIONS})
  target_compile_options(hiplz PRIVATE ${SANITIZER_OPTIONS})
endif()

target_compile_options(hiplz INTERFACE
        "$<INSTALL_INTERFACE:--hip-device-lib-path=${HIPLZ_DATA_DIR}>"
        "$<BUILD_INTERFACE:--hip-device-lib-path=${CMAKE_BINARY_DIR}>"
        "--hip-device-lib=kernellib.bc")

# for LLVM passes
target_compile_options(hiplz INTERFACE
        "$<INSTALL_INTERFACE:--hip-llvm-pass-path=${HIPLZ_LLVM_DIR}>"
        "$<BUILD_INTERFACE:--hip-llvm-pass-path=${CMAKE_BINARY_DIR}/llvm_passes>")

target_include_directories(hiplz
    PUBLIC
        "$<INSTALL_INTERFACE:${HIPLZ_INC_DIR}>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
    PRIVATE
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/include"
        "${Level_0_INCLUDE_DIR}"
)

install(TARGETS hiplz
        EXPORT "hip-targets"
        LIBRARY DESTINATION "${HIPLZ_LIB_DIR}"
        ARCHIVE DESTINATION "${HIPLZ_LIB_DIR}"
        )

install(EXPORT "hip-targets"
        NAMESPACE "hip::"
        DESTINATION "${HIPLZ_CMAKE_DIR}")

install(FILES
        "${CMAKE_SOURCE_DIR}/include/hip/hipcl.hh"
        "${CMAKE_SOURCE_DIR}/include/hip/hipcl_mathlib.hh"
        "${CMAKE_SOURCE_DIR}/include/hip/hip_fatbin.h"
        "${CMAKE_SOURCE_DIR}/include/hip/hip_fp16.h"
        "${CMAKE_SOURCE_DIR}/include/hip/hip_runtime.h"
        "${CMAKE_SOURCE_DIR}/include/hip/hip_vector_types.h"
  DESTINATION "${HIPLZ_INC_DIR}/hip")

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/hiplz-config.cmake.in"
  "${CMAKE_BINARY_DIR}/hipConfig.cmake"
  INSTALL_DESTINATION "${HIPLZ_CMAKE_DIR}"
  PATH_VARS HIPLZ_INC_DIR HIPLZ_LIB_DIR HIPLZ_BIN_DIR HIPLZ_COMPILER
  )

write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/hipConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion
  )
