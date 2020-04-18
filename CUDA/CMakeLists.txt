cmake_minimum_required(VERSION 3.16)

set(This NeuralNetwork)

project(${This} LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

enable_language(CUDA)

enable_testing()

add_library(${This} SHARED)
target_sources(${This} PRIVATE src/NeuralNetwork.cpp)
target_sources(${This} PRIVATE src/cuda_extension.cu)
target_include_directories(${This} PUBLIC "include")

target_link_libraries(${This} PRIVATE
	${CUDA_LIBRARIES}
	${CUDA_cudart_shared_LIBRARY}
	${CUDA_cublas_LIBRARY}
	${CUDA_curand_LIBRARY}
)

target_include_directories(${This} PRIVATE include)
set_property(TARGET ${This} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_subdirectory(test)
