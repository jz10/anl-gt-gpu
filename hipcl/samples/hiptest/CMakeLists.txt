# hiptest

if( ${LLVM_VERSION} VERSION_GREATER "9.0" )
  # Clang 10 introduced new HIP launch API option.
  # Clang 11 switched to use the new HIP launch API option by default.
  add_hiplz_test(hiptest-old-launch-api hiptest-old-launch-api PASSED hiptest.cc)
  target_compile_options(hiptest-old-launch-api PRIVATE "-fno-hip-new-launch-api")
  add_hiplz_test(hiptest-new-launch-api hiptest-new-launch-api PASSED hiptest.cc)
  target_compile_options(hiptest-new-launch-api PRIVATE "-fhip-new-launch-api")
else()
  add_hiplz_test(hiptest hiptest PASSED hiptest.cc)
endif()
