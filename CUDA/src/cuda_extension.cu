#include "cuda_extension.h"
#include "helper_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define  CUDA_CALL(x) do { if((x) !=  cudaSuccess) { \
	printf ("Error  at %s:%d\n",__FILE__ ,__LINE__); \
	return  EXIT_FAILURE ;}}  while (0)

namespace cuda_extension {

cublasHandle_t handle;

inline int initDevice(IN_Layer inLayer, std::vector<NN_Layer> hiddenLayers, NN_Layer outLayer) {

	cublasStatus_t status;
	cudaError_t err;
	const char *str = "";

	curandState *devStates;

	// Get CUDA device
	int dev = findCudaDevice(0, &str);
	if (dev == -1) {
		return EXIT_FAILURE;
	}

	// Create CUBLAS Handle
	printf("INITIALIZING CUBLASS...");

	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLASS FAILED TO CREATE HANDLE!");
		return EXIT_FAILURE;
	}

	printf("CUBLASS INITIALIZED!");


	// Allocate Memory on device for layers
	printf("ALLOCATING DEVICE MEMORY...");

	// Input Layer
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(inLayer.inputs)), inLayer.Nodes * sizeof(float)));

	// Output Layer
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(outLayer.output)), outLayer.Nodes * sizeof(float)));

	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(outLayer.bias)), outLayer.Nodes * sizeof(float)));

	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(outLayer.error)), outLayer.Nodes * sizeof(float)));

	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(outLayer.weights)), *outLayer.wRows * outLayer.wCols * sizeof(float)));

	// Hidden Layers
	for (auto i : hiddenLayers) {
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(i.output)), i.Nodes * sizeof(float)));

		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(i.bias)), i.Nodes * sizeof(float)));

		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(i.error)), i.Nodes * sizeof(float)));

		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(i.weights)), *i.wRows * i.wCols * sizeof(float)));

	}

	printf("MEMORY ALLOCATED!");

//	// Fill device weight and bias matrices with random data with CURAND
//	printf("FILLING WEIGHTS AND BIAS WITH RANDOM VALUES...");
//
//	printf("INITIALIZING CURAND...");
//
//	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&devStates), sizeof(curandState)));
//
//
	return 0;


}

//__global__
//void deviceRandomFill (float *matrix, unsigned size, curandState *globalState) {
//	curandState localState;
//	localState = global_State[]
//}



}
