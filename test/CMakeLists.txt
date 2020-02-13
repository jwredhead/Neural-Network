cmake_minimum_required(VERSION 3.16)

set(This NeuralNetworkTests)

set(Sources
	MatrixTest.cpp
)

add_executable(${This} ${Sources})

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
