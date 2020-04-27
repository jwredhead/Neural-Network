// CUSTOM INCLUDES
#include "cuda_extension.h"


// CPP INCLUDES
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>

// CUDA INCLUDES
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper_cuda.h"

// Macro for output of cuda errors
#define  CUDA_CALL(x) do { if((x) !=  cudaSuccess) { \
	printf ("Error  at %s:%d\nError Value: %s\n",__FILE__ ,__LINE__, cudaGetErrorString(x)); \
	exit(EXIT_FAILURE);}}  while (0)

namespace cuda_extension {

	// Namespace-wide variables
	cublasHandle_t handle; // Handle for Cublas
	unsigned maxNodes; // Max number of nodes

	// Initialization kernel for GPU Random Number Generator
	__global__ void initRand (curandState_t *state) {
		int id = threadIdx.x + blockIdx.x * blockDim.x;

		curand_init(1234, id, 0, &state[id]);
	}

	// Kernel to fill a given matrix with random numbers
	__global__ void deviceRandomFill (float *matrix, unsigned size, curandState_t *globalState) {

		int id = threadIdx.x + blockIdx.x * blockDim.x;

		curandState_t localState;
		localState = globalState[id];

		float s_value = curand_uniform(&localState);

		__syncthreads();

		if(id < size) {
			matrix[id] = s_value;
		}
		globalState[id] = localState;
	}

	// Kernel to calculate the error between a target matrix and an error matrix
	__global__ void calcError (float *targets, float *output, float *error, unsigned size) {
		int id = threadIdx.x + blockIdx.x * blockDim.x;

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

	__global__ void calcSoftmaxCE_Error(float *targets, float *outputs, float *error, unsigned size) {
		int id = threadIdx.x + blockIdx.x * blockDim.x;
		float t = 0, e = 0, o = 0;

		if (id < size) {
			t = targets[id];
			e = error[id];
			o = outputs[id];
		}
		__syncthreads();

		e = o - t;

		__syncthreads();

		if (id < size) {
			error[id] = e;
		}

	}

	// Kernel to add 2 matrices
	__global__ void matrixSub (float *x, float *y, unsigned size) {
		int id = threadIdx.x + blockIdx.x * blockDim.x;

		float  s_x = 0, s_y = 0;

		if (id < size) {
			s_x = x[id];
			s_y = y[id];
		}

		__syncthreads();

		s_x -= s_y;

		__syncthreads();

		if(id < size) {
			x[id] = s_x;
		}
	}

	// Kernel to run sigmoid function over every element of a matrix
	__global__ void runSigmoidFunction(float *x, unsigned size) {
		int id = threadIdx.x + blockIdx.x * blockDim.x;

		float t = 0;

		if (id < size) {
			t = x[id];
		}

		__syncthreads();

		t = (1 - expf(-t) / (1 + expf(-t)));

		__syncthreads();

		if (id < size) {
			x[id] = t;
		}
	}

	// Kernel to run bisigmoid function over every element of a matrix
	__global__ void runBisigmoidFunction(float *x, unsigned size) {
		int id = threadIdx.x + blockIdx.x * blockDim.x;

		float t = 0;

		if (id < size) {
			t = x[id];
		}

		__syncthreads();

		t = 1 - expf(-t) / (1 + expf(-t));

		__syncthreads();

		if (id < size) {
			x[id] = t;
		}
	}

	// Kernel to run tanh function over every element of a matrix
	__global__ void runTanhFunction(float *x, unsigned size) {
		int id = threadIdx.x + blockIdx.x * blockDim.x;

		float t = 0;

		if (id < size) {
			t = x[id];
		}

		__syncthreads();

		t = tanh(t);

		__syncthreads();

		if (id < size) {
			x[id] = t;
		}
	}

	// Kernel to calculate the gradient between outputs, x, and errors, y,
	__global__ void calcGradientSigmoid(float *output, float *error, unsigned size) {
		int id = threadIdx.x + blockIdx.x * blockDim.x;

		float o = 0, e = 0;
		if (id < size) {
			o = output[id];
			e = error[id];
		}

		__syncthreads();

		e = (o * ( 1 - o)) * e;

		__syncthreads();

		if (id < size) {
			error[id] = e;
		}
	}

	// Kernel to calculate the gradient between outputs, x, and errors, y,
	__global__ void calcGradientBisigmoid(float *output, float *error, unsigned size) {
		int id = threadIdx.x + blockIdx.x * blockDim.x;

		float o = 0, e = 0;
		if (id < size) {
			o = output[id];
			e = error[id];
		}

		__syncthreads();

		e = (2 * o * (1 - o)) * e;

		__syncthreads();

		if (id < size) {
			error[id] = e;
		}
	}

	// Kernel to calculate the gradient between outputs, x, and errors, y,
	__global__ void calcGradientTanh(float *output, float *error, unsigned size) {
		int id = threadIdx.x + blockIdx.x * blockDim.x;

		float o = 0, e = 0;
		if (id < size) {
			o = output[id];
			e = error[id];
		}

		__syncthreads();

		e = (1 - (o * o)) * e;

		__syncthreads();

		if (id < size) {
			error[id] = e;
		}
	}

	__global__ void calcExp(float *x, unsigned size){
		int id = threadIdx.x + blockIdx.x * blockDim.x;

		float t = 0;
		if (id < size) {
			t = x[id];
		}

		__syncthreads();

		t = expf(t);

		__syncthreads();

		if (id < size) {
			x[id] = t;
		}
	}

	__global__ void runSoftmaxFunction(float *x, unsigned size, float *sum){
		int id = threadIdx.x + blockIdx.x * blockDim.x;

		float t = 0;
		if (id < size) {
			t = x[id];
		}

		__syncthreads();


			t = t / *sum;


		__syncthreads();

		if (id < size) {
			x[id] = t;
		}
	}

	// Convert MATRIX_OP to cublasOperation_t
	inline cublasOperation_t convertToCublasOp (MATRIX_OP t) {

		cublasOperation_t s;

		std::map<MATRIX_OP, cublasOperation_t> map = {
				{MATRIX_OP::NORMAL, cublasOperation_t::CUBLAS_OP_N},
				{MATRIX_OP::TRANSPOSE, cublasOperation_t::CUBLAS_OP_N}
		};

		s = map.at(t);

		return s;
	}

	// If device ptr isn't null, free it
	inline int cudaDelete (float *a) {
		if (a) {
			CUDA_CALL(cudaFree(a));
		}
		return 0;
	}

	// Initialize Neural Network Layers on the device
	void initLayers(IN_Layer *inLayer, std::vector<NN_Layer> *hiddenLayers, NN_Layer *outLayer) {

		cublasStatus_t status;
		const char *str = "";

		curandState_t *devStates;

		// Get CUDA device
		int dev = findCudaDevice(0, &str);
		if (dev == -1) {
			fprintf(stderr, "FINDCUDADEVICE ERROR: -1\n");
			exit(EXIT_FAILURE);
		}

		// Create CUBLAS Handle
		printf("INITIALIZING CUBLASS...\n");

		status = cublasCreate(&handle);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "CUBLASS FAILED TO CREATE HANDLE!\nERROR: %s\n", _cudaGetErrorEnum(status));
			exit(EXIT_FAILURE);
		}

		printf("CUBLASS INITIALIZED!\n");


		// Allocate Memory on device for layers
		printf("ALLOCATING DEVICE MEMORY...\n");

		// Input Layer
		CUDA_CALL(cudaMalloc(&(inLayer->inputs), inLayer->Nodes * sizeof(float)));

		maxNodes = inLayer->Nodes;

		// Output Layer
		CUDA_CALL(cudaMalloc(&(outLayer->output), outLayer->Nodes * sizeof(float)));

		CUDA_CALL(cudaMalloc(&(outLayer->bias), outLayer->Nodes * sizeof(float)));

		CUDA_CALL(cudaMalloc(&(outLayer->error), outLayer->Nodes * sizeof(float)));

		CUDA_CALL(cudaMalloc(&(outLayer->weights), outLayer->wRows * outLayer->wCols * sizeof(float)));

		if (outLayer->Nodes > maxNodes) {
			maxNodes = outLayer->Nodes;
		}

		// Hidden Layers
		for (auto&& i : *hiddenLayers) {
			CUDA_CALL(cudaMalloc(&(i.output), i.Nodes * sizeof(float)));

			CUDA_CALL(cudaMalloc(&(i.bias), i.Nodes * sizeof(float)));

			CUDA_CALL(cudaMalloc(&(i.error), i.Nodes * sizeof(float)));

			CUDA_CALL(cudaMalloc(&(i.weights), i.wRows * i.wCols * sizeof(float)));

			if (i.Nodes > maxNodes) {
				maxNodes = i.Nodes;
			}

		}

		printf("MEMORY ALLOCATED!\n");

		// Fill device weight and bias matrices with random data with CURAND
		printf("FILLING WEIGHTS AND BIAS WITH RANDOM VALUES...\n");

		printf("INITIALIZING CURAND...\n");

		unsigned numBlocks = ceil(maxNodes * maxNodes / MAX_THREADS_PER_BLOCK);
		unsigned numThreads = (maxNodes > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : maxNodes;
		dim3 dimGrid(numBlocks, 1, 1);
		dim3 dimBlock(numThreads, 1, 1);
		printf("numBlocks: %d, numThreads %d\n" , numBlocks, numThreads);
		CUDA_CALL(cudaMalloc(&devStates, numBlocks * MAX_THREADS_PER_BLOCK * sizeof(curandState_t)));

		initRand<<<dimGrid,dimBlock>>>(devStates);

		printf("CURAND INITIALIZED!\n");

		deviceRandomFill<<<dimGrid, dimBlock>>>(outLayer->weights, outLayer->wRows * outLayer->wCols, devStates);
		deviceRandomFill<<<dimGrid, dimBlock>>>(outLayer->bias, outLayer->Nodes, devStates);

		for (auto i : *hiddenLayers) {
			deviceRandomFill<<<dimGrid, dimBlock>>>(i.weights, i.wRows * i.wCols, devStates);
			deviceRandomFill<<<dimGrid, dimBlock>>>(i.bias, i.Nodes, devStates);
		}

		CUDA_CALL(cudaDeviceSynchronize());
		printf("WEIGHT AND BIAS MATRICES FILLED!\n");

	}

	// Delete Neural Network from device
	void deleteLayers(IN_Layer *inLayer, std::vector<NN_Layer> *hiddenLayers, NN_Layer *outLayer) {
		printf("DELETING LAYERS FROM DEVICE...\n");
		int error;

		error = cudaDelete(inLayer->inputs);

		error += cudaDelete(outLayer->weights);
		error +=cudaDelete(outLayer->bias);
		error +=cudaDelete(outLayer->error);
		error +=cudaDelete(outLayer->output);

		for ( auto&& i : *hiddenLayers) {
			error +=cudaDelete(i.weights);
			error +=cudaDelete(i.bias);
			error +=cudaDelete(i.error);
			error +=cudaDelete(i.output);
		}
		if (error ==0) {
			printf("LAYERS DELETED FROM DEVICE!\n");
		} else {
			printf("UNABLE TO DELETE LAYERS FROM DEVICE!/nNUMBER OF ERRORS: %d\n", error);
		}

		printf("CLOSING CUBLAS...\n");
		cublasStatus_t status = cublasDestroy(handle);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "CUBLASS FAILED TO DESTROY HANDLE!\nERROR: %s\n", _cudaGetErrorEnum(status));
		} else
		{
			printf("CUBLASS CLOSED!\n");
		}

	}

	// Wrapper for CUBLAS SGEMM matrix multiplication, calculates C = A * B + C
	void multiplyAccumulate(MATRIX_OP transA, float *A, float *B, float *C, int A_rows, int A_cols) {
		float alpha = 1.0;
		float beta = 1.0;


		cublasStatus_t status;
		status = cublasSgemv(handle,convertToCublasOp(transA), A_rows, A_cols, &alpha, A, A_rows, B, 1, &beta, C, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "CUBLASS MATRIX MULTIPLY ERROR: %s\n", _cudaGetErrorEnum(status));
		}
	}

	// Wrapper for CUBLAS SGEMM matrix multiplication, calculates C = A * B
	void multiply(MATRIX_OP transA, float *A, float *B, float *C, int A_rows, int A_cols) {
		float alpha= 1.0;
		float beta = 0.0;


		cublasStatus_t status;
		status = cublasSgemv(handle,convertToCublasOp(transA), A_rows, A_cols, &alpha, A, A_rows, B, 1, &beta, C, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "CUBLASS MATRIX MULTIPLY ERROR: %s\n", _cudaGetErrorEnum(status));
		}
	}

	// Copies data from src to dst, where both pointers are device ptrs
	void copyVector(NN_Layer *layer) {
		CUDA_CALL(cudaMemcpy(layer->output, layer->bias,  layer->Nodes * sizeof(float), cudaMemcpyDeviceToDevice));
	}

	// Copies inputs over to device
	void copyInputs(float *src, IN_Layer *inLayer) {
		CUDA_CALL(cudaMemcpy(inLayer->inputs, src, inLayer->Nodes * sizeof(float), cudaMemcpyHostToDevice));
	}

	// Runs activation function over matrix
	void activationFunction(float *x, unsigned size, Activation_Function f) {
		cublasStatus_t status;
		float *softSum;
		CUDA_CALL(cudaMalloc(&softSum, sizeof(float)));


		unsigned numBlocks = ceil(maxNodes * maxNodes / MAX_THREADS_PER_BLOCK);
		unsigned numThreads = (maxNodes > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : maxNodes;
		dim3 dimGrid(numBlocks, 1, 1);
		dim3 dimBlock(numThreads, 1, 1);

		switch (f) {
			case Activation_Function::SIGMOID: runSigmoidFunction<<<dimGrid, dimBlock>>>(x, size); break;
			case Activation_Function::BI_SIGMOID: runBisigmoidFunction<<<dimGrid, dimBlock>>>(x, size); break;
			case Activation_Function::TANH: runTanhFunction<<<dimGrid, dimBlock>>>(x, size); break;
			case Activation_Function::SOFTMAX: 	calcExp<<<dimGrid, dimBlock>>>(x,size);
												CUDA_CALL(cudaDeviceSynchronize());
												cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);
												status = cublasSasum(handle,size,x,1,softSum);
												if (status != CUBLAS_STATUS_SUCCESS) {
													fprintf(stderr, "CUBLAS ASUM ERROR: %s\n", _cudaGetErrorEnum(status));
												}
												cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST);
												runSoftmaxFunction<<<dimGrid, dimBlock>>>(x, size, softSum);
												break;
		}

		CUDA_CALL(cudaDeviceSynchronize());
		cudaDelete(softSum);
	}

	// Calculates the error between given targets and outputs
	void calculateError(float *targets, float *outputs, float *error, unsigned size, Activation_Function funct) {
		float * d_targets;
		CUDA_CALL(cudaMalloc(&d_targets, size * sizeof(float)));
		CUDA_CALL(cudaMemcpy(d_targets, targets, size * sizeof(float), cudaMemcpyHostToDevice));


		unsigned numBlocks = ceil(maxNodes * maxNodes / MAX_THREADS_PER_BLOCK);
		unsigned numThreads = (maxNodes > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : maxNodes;
		dim3 dimGrid(numBlocks, 1, 1);
		dim3 dimBlock(numThreads, 1, 1);
		if (funct == Activation_Function::SOFTMAX) {
			calcSoftmaxCE_Error<<<dimGrid, dimBlock>>>(d_targets, outputs, error, size);
		} else {
			calcError<<<dimGrid, dimBlock>>>(d_targets, outputs, error, size);
		}

		CUDA_CALL(cudaDeviceSynchronize());
	}

	void backPropogate(NN_Layer *thisLayer, NN_Layer *backLayer, Activation_Function f) {
		unsigned numBlocks = ceil(maxNodes * maxNodes / MAX_THREADS_PER_BLOCK);
		unsigned numThreads = (maxNodes > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : maxNodes;
		dim3 dimGrid(numBlocks, 1, 1);
		dim3 dimBlock(numThreads, 1, 1);

		multiply(MATRIX_OP::TRANSPOSE, backLayer->weights,
						backLayer->error,
						thisLayer->error,
						backLayer->wCols,
						backLayer->wRows);

		switch (f) {
			case Activation_Function::SIGMOID: calcGradientSigmoid<<<dimGrid, dimBlock>>>(thisLayer->output, thisLayer->error, thisLayer->Nodes); break;
			case Activation_Function::BI_SIGMOID: calcGradientBisigmoid<<<dimGrid, dimBlock>>>(thisLayer->output, thisLayer->error, thisLayer->Nodes); break;
			case Activation_Function::TANH: calcGradientTanh<<<dimGrid, dimBlock>>>(thisLayer->output, thisLayer->error, thisLayer->Nodes); break;
		}

		CUDA_CALL(cudaDeviceSynchronize());
	}

	// Adjusts a given layer's weights and bias back propogating gradient descent
	void adjustWeightsBias(NN_Layer *layer, float *inputs, unsigned inputSize, float learningRate) {
		unsigned numBlocks = ceil(maxNodes * maxNodes / MAX_THREADS_PER_BLOCK);
		unsigned numThreads = (maxNodes > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : maxNodes;
		dim3 dimGrid(numBlocks, 1, 1);
		dim3 dimBlock(numThreads, 1, 1);
		cublasStatus_t status;

		float  *gradient;
		CUDA_CALL(cudaMalloc(&gradient, layer->Nodes *sizeof(float)));
		CUDA_CALL(cudaMemcpy(gradient, layer->error,  layer->Nodes * sizeof(float), cudaMemcpyDeviceToDevice));
		cublasSscal(handle, layer->Nodes, &learningRate, gradient, 1);
		matrixSub<<<dimGrid, dimBlock>>>(layer->bias, gradient, layer->Nodes);
		CUDA_CALL(cudaDeviceSynchronize());
		cudaDelete(gradient);

		float lr = -(learningRate);
		status = cublasSger(handle,layer->wRows, layer->wCols, &lr, layer->error, 1, inputs, 1, layer->weights, layer->wRows);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "CUBLAS ASUM ERROR: %s\n", _cudaGetErrorEnum(status));
		}


	}

	// Retrieves Neural Network outputs from device to host
	void getOutputs(float *d_outputs, float *h_outputs, unsigned size) {

		CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(h_outputs), reinterpret_cast<void*>(d_outputs), size * sizeof(float), cudaMemcpyDeviceToHost));
	}

} /* namespace cuda_extension */

#undef CUDA_CALL
