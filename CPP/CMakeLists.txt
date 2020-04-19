cmake_minimum_required(VERSION 3.16)

set(This NeuralNetwork)

project(${This} C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)


enable_testing()

set(Headers
	Matrix.h
	MatrixException.h
	NeuralNetworkTypes.h
	NeuralNetwork.h
)
list(TRANSFORM Headers PREPEND "include/")

set(Sources
	Matrix.cpp
	NeuralNetwork.cpp
)

list(TRANSFORM Sources PREPEND "src/")

add_library(${This} SHARED ${Sources} ${Headers})

target_include_directories(${This} PRIVATE include)

add_subdirectory(test)
