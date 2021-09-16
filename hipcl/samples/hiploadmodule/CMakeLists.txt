# Test symbol access

add_hiplz_binary(
	hipModuleLoad
	main.cpp)

add_hiplz_device_binary(
	hipModuleLoadBinary
	kernel.cpp)

add_dependencies(hipModuleLoad hipModuleLoadBinary)

add_test(NAME hipModuleLoad
	 COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hipModuleLoad"
	 )

set_tests_properties(hipModuleLoad PROPERTIES
	PASS_REGULAR_EXPRESSION PASSED)
