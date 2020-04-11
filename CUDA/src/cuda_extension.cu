#include "cuda_extension.h"
#include "helper_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define  CUDA_CALL(x) do { if((x) !=  cudaSuccess) { \
	printf ("Error  at %s:%d\n",__FILE__ ,__LINE__); \
	return  EXIT_FAILURE ;}}  while (0)

namespace cuda_extension {

cublasHandle_t handle;

__global__
void initRand (curandState_t *state) {
	int id = threadIdx.x + blockIdx.x * 256;

	curand_init(1234, id, 0, &state[id]);
}

__global__
void deviceRandomFill (float *matrix, unsigned size, curandState_t *globalState) {

	int id = threadIdx.x + blockIdx.x * 256;

	curandState_t localState;
	localState = globalState[id];


	if(id < size) {
		matrix[id] = curand_uniform(&localState);
	}

	globalState[id] = localState;
}

inline int initDevice(IN_Layer inLayer, std::vector<NN_Layer> hiddenLayers, NN_Layer outLayer) {

	cublasStatus_t status;
	const char *str = "";

	curandState_t *devStates;

	unsigned maxNodes;

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

	maxNodes = inLayer.Nodes;

	// Output Layer
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(outLayer.output)), outLayer.Nodes * sizeof(float)));

	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(outLayer.bias)), outLayer.Nodes * sizeof(float)));

	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(outLayer.error)), outLayer.Nodes * sizeof(float)));

	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(outLayer.weights)), *outLayer.wRows * outLayer.wCols * sizeof(float)));

	if (outLayer.Nodes > maxNodes) {
		maxNodes = outLayer.Nodes;
	}

	// Hidden Layers
	for (auto i : hiddenLayers) {
		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(i.output)), i.Nodes * sizeof(float)));

		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(i.bias)), i.Nodes * sizeof(float)));

		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(i.error)), i.Nodes * sizeof(float)));

		CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&(i.weights)), *i.wRows * i.wCols * sizeof(float)));

		if (i.Nodes > maxNodes) {
			maxNodes = i.Nodes;
		}

	}

	printf("MEMORY ALLOCATED!");

	// Fill device weight and bias matrices with random data with CURAND
	printf("FILLING WEIGHTS AND BIAS WITH RANDOM VALUES...");

	printf("INITIALIZING CURAND...");

	unsigned numBlocks = ceil(maxNodes * maxNodes / MAX_THREADS_PER_BLOCK);
	dim3 dimGrid(numBlocks, 1, 1);
	dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1, 1);

	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&devStates), numBlocks * MAX_THREADS_PER_BLOCK * sizeof(curandState_t)));

	initRand<<<dimGrid,dimBlock>>>(devStates);

	printf("CURAND INITIALIZED!");

	deviceRandomFill<<<dimGrid, dimBlock>>>(outLayer.weights, *outLayer.wRows * outLayer.wCols, devStates);
	deviceRandomFill<<<dimGrid, dimBlock>>>(outLayer.bias, outLayer.Nodes, devStates);

	for (auto i : hiddenLayers) {
		deviceRandomFill<<<dimGrid, dimBlock>>>(i.weights, *i.wRows * i.wCols, devStates);
		deviceRandomFill<<<dimGrid, dimBlock>>>(i.bias, i.Nodes, devStates);
	}

	printf("WEIGHT AND BIAS MATRICES FILLED!");

	return 0;
}

}
