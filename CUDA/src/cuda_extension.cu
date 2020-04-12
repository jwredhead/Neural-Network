#include "cuda_extension.h"
#include "helper_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <map>
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

inline cublasOperation_t convertToCublasOp (MATRIX_OP t) {

	cublasOperation_t s;

	std::map<MATRIX_OP, cublasOperation_t> map = {
			{MATRIX_OP::NORMAL, cublasOperation_t::CUBLAS_OP_N},
			{MATRIX_OP::TRANSPOSE, cublasOperation_t::CUBLAS_OP_N}
	};

	s = map.at(t);

	return s;
}

inline int cudaDelete (float *a) {
	if (a) {
		CUDA_CALL(cudaFree(a));
	}
	return 0;
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
		fprintf(stderr, "CUBLASS FAILED TO CREATE HANDLE!\nERROR: %d", status);
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

inline void deleteLayers(IN_Layer inLayer, std::vector<NN_Layer> hiddenLayers, NN_Layer outLayer) {
	printf("DELETING LAYERS FROM DEVICE...");
	int error;

	error = cudaDelete(inLayer.inputs);

	error += cudaDelete(outLayer.weights);
	error +=cudaDelete(outLayer.bias);
	error +=cudaDelete(outLayer.error);
	error +=cudaDelete(outLayer.output);

	for ( auto i : hiddenLayers) {
		error +=cudaDelete(i.weights);
		error +=cudaDelete(i.bias);
		error +=cudaDelete(i.error);
		error +=cudaDelete(i.output);
	}
	if (error ==0) {
		printf("LAYERS DELETED FROM DEVICE!");
	} else {
		printf("UNABLE TO DELETE LAYERS FROM DEVICE!/nNUMBER OF ERRORS: %d", error);
	}

	printf("CLOSING CUBLAS...");
	cublasStatus_t status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLASS FAILED TO DESTROY HANDLE!/nERROR: %d", status);
	} else
	{
		fprintf(stderr, "CUBLASS CLOSED!");
	}

}

inline void multiplyAccumulate(MATRIX_OP transA, float *A, MATRIX_OP transB, float *B, float *C, int m, int n, int k) {
	int lda=m;
	int ldb=k;
	int ldc=m;
	const float alf = 1;
	const float *alpha = &alf;
	const float bet = 1;
	const float *beta = &bet;

	cublasStatus_t status;
	status = cublasSgemm(handle,convertToCublasOp(transA), convertToCublasOp(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLASS MATRIX MULTIPLY ERROR: %d", status);
	}
}

inline int copyVector(float *x, float *y, unsigned size) {
	CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(x), reinterpret_cast<void*>(y), size, cudaMemcpyDeviceToDevice));
	return 0;
}

inline void copyInputs(float *x, float *y, unsigned size) {
	cublasStatus_t status;
	status = cublasSetVector(size, sizeof(float), reinterpret_cast<void*>(x), 1, reinterpret_cast<void*>(y), 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLASS SET VECTOR ERROR: %d", status);
	}

}

}

