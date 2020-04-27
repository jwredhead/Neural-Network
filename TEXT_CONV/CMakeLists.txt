cmake_minimum_required(VERSION 3.16)

set(This TextConv)

project(${This} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
include(FetchContent)
FetchContent_Declare(
	mnist
	GIT_REPOSITORY https://github.com/wichtounet/mnist.git
	GIT_TAG master
	)
FetchContent_GetProperties(mnist)
if(NOT mnist_POPULATED)
	FetchContent_MakeAvailable(mnist)
endif()
find_package(MNIST PATHS ${CMAKE_BINARY_DIR}/_deps/mnist-src)
if(NOT MNIST_FOUND)
	message(FATAL_ERROR "MNIST loader could not be found")
endif(NOT MNIST_FOUND)

include_directories(${MNIST_INCLUDE_DIR})

add_executable(${This})
target_sources(${This} PRIVATE src/main.cpp)
target_include_directories(${This} PUBLIC "include")

find_library(NN_LIBRARY NAMES NeuralNetwork PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib)

target_link_libraries(${This} PRIVATE
	${NN_LIBRARY}
)

target_include_directories(${This} PRIVATE include)
set_property(TARGET ${This} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_definitions(${This} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
