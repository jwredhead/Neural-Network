// CUSTOM INCLUDES
#include "cuda_extension.h"
#include "helper_cuda.h"

// CPP INCLUDES
#include <stdio.h>
#include <stdlib.h>
#include <map>

// CUDA INCLUDES
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Macro for output of cuda errors
#define  CUDA_CALL(x) do { if((x) !=  cudaSuccess) { \
	printf ("Error  at %s:%d\n",__FILE__ ,__LINE__); \
	exit(EXIT_FAILURE);}}  while (0)

namespace cuda_extension {

// Namespace-wide variables
cublasHandle_t handle; // Handle for Cublas
unsigned maxNodes; // Max number of nodes

// Sigmoid functor
struct sigmoid {

	__device__ float operator() (float x)
	{
		return 1 / (1 + expf(-x));
	}

};

// Bisigmoid functor
struct bisigmoid {

	__device__ float operator() (float x)
	{
		return (1 - expf(-x) / (1 + expf(-x)));
	}

};

struct sigmoidDerivative {

	__device__ float operator() (float x) {
		return x * ( 1 - x);
	}

};

struct bisigmoidDerivative {

	__device__ float operator() (float x) {
		return 2 * x * (1 - x);
	}

};

struct tanhDerivative {

	__device__ float operator() (float x) {
		return 1 - x * x;
	}

};

__global__ void initRand (curandState_t *state) {
	int id = threadIdx.x + blockIdx.x * 256;

	curand_init(1234, id, 0, &state[id]);
}

__global__ void deviceRandomFill (float *matrix, unsigned size, curandState_t *globalState) {

	int id = threadIdx.x + blockIdx.x * 256;

	curandState_t localState;
	localState = globalState[id];

	float s_value = curand_uniform(&localState);

	__syncthreads();

	if(id < size) {
		matrix[id] = s_value;
	}

	globalState[id] = localState;
}

__global__ void calcError (float *targets, float *output, float *error, unsigned size) {
	int id = threadIdx.x + blockIdx.x * 256;

	float t = 0, e = 0, o = 0;

	if (id < size) {
		t = targets[id];
		e = error[id];
		o = output[id];
	}

	__syncthreads();

	e = t - o;

	__syncthreads();

	if (id < size) {
		error[id] = e;
	}
}

__global__ void matrixAdd (float *x, float *y, unsigned size) {
	int id = threadIdx.x + blockIdx.x * 256;

	float  s_x = 0, s_y = 0;

	if (id < size) {
		s_x = x[id];
		s_y = y[id];
	}

	__syncthreads();

	s_x += s_y;

	__syncthreads();

	if(id < size) {
		x[id] = s_x;
	}
}

template <typename F>
__global__ void runFunction(F f, float *x, unsigned size) {
	int id = threadIdx.x + blockIdx.x * 256;

	float t = 0;

	if (id < size) {
		t = x[id];
	}

	__syncthreads();

	t = f(t);

	__syncthreads();

	if (id < size) {
		x[id] = t;
	}
}

template <typename F>
__global__ void calcGradient(F f, float *gradient, float *x, float *y, unsigned size, float lr) {
	int id = threadIdx.x + blockIdx.x * 256;

	float t = 0, e = 0, g = 0;
	if (id < size) {
		t = x[id];
		e = y[id];
	}

	__syncthreads();

	g = f(t);
	g = g * e;
	g = g * lr;

	__syncthreads();

	if (id < size) {
		gradient[id] = g;
	}
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

inline void initDevice(IN_Layer inLayer, std::vector<NN_Layer> hiddenLayers, NN_Layer outLayer) {

	cublasStatus_t status;
	const char *str = "";

	curandState_t *devStates;

	// Get CUDA device
	int dev = findCudaDevice(0, &str);
	if (dev == -1) {
		fprintf(stderr, "FINDCUDADEVICE ERROR: -1");
		exit(EXIT_FAILURE);
	}

	// Create CUBLAS Handle
	printf("INITIALIZING CUBLASS...");

	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLASS FAILED TO CREATE HANDLE!\nERROR: %d", status);
		exit(EXIT_FAILURE);
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
	unsigned numThreads = (maxNodes > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : maxNodes;
	dim3 dimGrid(numBlocks, 1, 1);
	dim3 dimBlock(numThreads, 1, 1);

	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&devStates), numBlocks * MAX_THREADS_PER_BLOCK * sizeof(curandState_t)));

	initRand<<<dimGrid,dimBlock>>>(devStates);

	printf("CURAND INITIALIZED!");

	deviceRandomFill<<<dimGrid, dimBlock>>>(outLayer.weights, *outLayer.wRows * outLayer.wCols, devStates);
	deviceRandomFill<<<dimGrid, dimBlock>>>(outLayer.bias, outLayer.Nodes, devStates);

	for (auto i : hiddenLayers) {
		deviceRandomFill<<<dimGrid, dimBlock>>>(i.weights, *i.wRows * i.wCols, devStates);
		deviceRandomFill<<<dimGrid, dimBlock>>>(i.bias, i.Nodes, devStates);
	}

	cudaDeviceSynchronize();
	printf("WEIGHT AND BIAS MATRICES FILLED!");

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

inline void multiply(MATRIX_OP transA, float *A, MATRIX_OP transB, float *B, float *C, int m, int n, int k) {
	int lda=m;
	int ldb=k;
	int ldc=m;
	const float alf = 1;
	const float *alpha = &alf;
	const float bet = 0;
	const float *beta = &bet;

	cublasStatus_t status;
	status = cublasSgemm(handle,convertToCublasOp(transA), convertToCublasOp(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLASS MATRIX MULTIPLY ERROR: %d", status);
	}
}

inline void copyVector(float *src, float *dst, unsigned size) {
	CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), size * sizeof(float), cudaMemcpyDeviceToDevice));
}

inline void copyInputs(float *src, float *dst, unsigned size) {
	cublasStatus_t status;
	status = cublasSetVector(size, sizeof(float), reinterpret_cast<void*>(src), 1, reinterpret_cast<void*>(dst), 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLASS SET VECTOR ERROR: %d", status);
	}

}

inline void activationFunction(float *x, unsigned size, Activation_Function f) {

	sigmoid sfunct;
	bisigmoid bfunct;

	unsigned numBlocks = ceil(maxNodes * maxNodes / MAX_THREADS_PER_BLOCK);
	unsigned numThreads = (maxNodes > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : maxNodes;
	dim3 dimGrid(numBlocks, 1, 1);
	dim3 dimBlock(numThreads, 1, 1);

	switch (f) {
		case Activation_Function::SIGMOID: runFunction<<<dimGrid, dimBlock>>>(sfunct, x, size); break;
		case Activation_Function::BI_SIGMOID: runFunction<<<dimGrid, dimBlock>>>(bfunct, x, size); break;
		case Activation_Function::TANH: runFunction<<<dimGrid, dimBlock>>>(tanhf, x, size); break;
	}

	cudaDeviceSynchronize();
}

inline void calculateError(float *targets, float *outputs, float *error, unsigned size) {
	unsigned numBlocks = ceil(maxNodes * maxNodes / MAX_THREADS_PER_BLOCK);
	unsigned numThreads = (maxNodes > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : maxNodes;
	dim3 dimGrid(numBlocks, 1, 1);
	dim3 dimBlock(numThreads, 1, 1);

	calcError<<<dimGrid, dimBlock>>>(targets, outputs, error, size);

	cudaDeviceSynchronize();
}

void adjustWeightsBias(NN_Layer layer, float *inputs, unsigned inputSize, Activation_Function f, float learningRate) {

	float *gradient, *deltaWeight;
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&gradient), layer.Nodes * sizeof(float)));
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&deltaWeight), layer.Nodes * inputSize *sizeof(float)));

	unsigned numBlocks = ceil(maxNodes * maxNodes / MAX_THREADS_PER_BLOCK);
	unsigned numThreads = (maxNodes > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : maxNodes;
	dim3 dimGrid(numBlocks, 1, 1);
	dim3 dimBlock(numThreads, 1, 1);

	sigmoidDerivative sfunct;
	bisigmoidDerivative bfunct;
	tanhDerivative dfunct;

	switch (f) {
		case Activation_Function::SIGMOID: calcGradient<<<dimGrid, dimBlock>>>(sfunct, gradient, layer.output, layer.error, layer.Nodes, learningRate); break;
		case Activation_Function::BI_SIGMOID: calcGradient<<<dimGrid, dimBlock>>>(bfunct, gradient, layer.output, layer.error, layer.Nodes, learningRate); break;
		case Activation_Function::TANH: calcGradient<<<dimGrid, dimBlock>>>(dfunct, gradient, layer.output, layer.error, layer.Nodes, learningRate); break;
	}

	cudaDeviceSynchronize();

	multiply(MATRIX_OP::NORMAL, gradient, MATRIX_OP::TRANSPOSE, inputs, deltaWeight, layer.Nodes, inputSize, 1);

	matrixAdd<<<dimGrid, dimBlock>>>(layer.weights, deltaWeight, layer.Nodes);
	matrixAdd<<<dimGrid, dimBlock>>>(layer.bias, gradient, layer.Nodes);

	cudaDeviceSynchronize();

	cudaDelete(gradient);
	cudaDelete(deltaWeight);
}

void getOutputs(float *d_outputs, float *h_outputs, unsigned size) {

	CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(h_outputs), reinterpret_cast<void*>(d_outputs), size * sizeof(float), cudaMemcpyDeviceToHost));
}

} /* namespace cuda_extension */

#undef CUDA_CALL
