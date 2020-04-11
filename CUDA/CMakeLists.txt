cmake_minimum_required(VERSION 3.16)

set(This NeuralNetwork)

project(${This} C CXX CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#enable_testing()

set(Headers
	cuda_extension.h
	NeuralNetworkTypes.h
	NeuralNetwork.h
)
list(TRANSFORM Headers PREPEND "include/")

set(Sources
	cuda_extension.cu
	NeuralNetwork.cpp
)

list(TRANSFORM Sources PREPEND "src/")

add_library(${This} STATIC ${Sources} ${Headers})

target_include_directories(${This} PRIVATE include)

#add_subdirectory(test)
