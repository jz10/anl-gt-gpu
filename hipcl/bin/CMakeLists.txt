
set_source_files_properties(hiplz_config.cc PROPERTIES LANGUAGE CXX )

add_executable(hiplz_config hiplz_config.cc)

set_target_properties(hiplz_config PROPERTIES CXX_STANDARD_REQUIRED ON)

target_include_directories(hiplz_config PRIVATE "${CMAKE_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/cxxopts/include")

target_link_libraries(hiplz_config ${PTHREAD_LIBRARY})

install(TARGETS "hiplz_config"
        RUNTIME DESTINATION "${HIPLZ_BIN_DIR}")
