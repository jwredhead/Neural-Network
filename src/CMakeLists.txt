cmake_minimum_required(VERSION 3.16)

set(This NeuralNetwork)

project(${This} C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)



set(Headers
	../include/Matrix.h
	../include/MatrixException.h
)

set(Sources
	Matrix.cpp
	NeuralNetwork.cpp
)

add_executable(${This} ${Sources} ${Headers})

target_include_directories(${This} PRIVATE ../include)