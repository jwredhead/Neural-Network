cmake_minimum_required(VERSION 3.16)

set(This NeuralNetworkTests)

set(Headers
	MatrixTest.h
)

set(Sources
	RunTests.cpp
)

add_executable(${This} ${Sources} ${Headers})

target_include_directories(${This} PRIVATE ../include)

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
