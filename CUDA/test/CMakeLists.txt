cmake_minimum_required(VERSION 3.16)

set(This NeuralNetworkTests)

set(Headers
	NeuralNetworkTest.h
)

set(Sources
	RunTests.cpp
)

add_executable(${This})
target_sources(${This} PRIVATE RunTests.cpp)

target_include_directories(${This} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${This} PUBLIC
	gtest
	gtest_main
	pthread
	NeuralNetwork
)

add_test(
	NAME ${This}
	COMMAND ${This}
)
