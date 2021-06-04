
option(SAVE_TEMPS "Save temporary compilation products" OFF)
option(VERBOSE "Verbose compilation" OFF)

if(SAVE_TEMPS)
  add_compile_options("--save-temps")
endif()

if(VERBOSE)
  add_compile_options("-v")
endif()

# ARGN = test args
function(add_hipcl_test EXEC_NAME TEST_NAME TEST_PASS SOURCE)

    set(TEST_EXEC_ARGS ${ARGN})
    set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCE})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_link_libraries("${EXEC_NAME}" "${SANITIZER_LIBS}" "hipcl" "OpenCL"
                          "ze_loader")

    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")

    add_test(NAME "${TEST_NAME}"
             COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME}"
                     ${TEST_EXEC_ARGS})

    set_tests_properties("${TEST_NAME}" PROPERTIES
                         PASS_REGULAR_EXPRESSION "${TEST_PASS}")


endfunction()

# ARGN = sources
function(add_hipcl_binary EXEC_NAME)

    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCES})

    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_link_libraries("${EXEC_NAME}" "${SANITIZER_LIBS}" "hipcl" "OpenCL"
                          "ze_loader")

    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${HIPCL_SAMPLE_BINDIR}")

endfunction()

add_hipcl_test(regression-maxgridsize regression-maxgridsize
               PASSED regression-maxgridsize.cc)
