#include "cuda_extension.h"
#include "helper_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cuda_extension {

cublasHandle_t handle;

inline int initDevice(IN_Layer inLayer, std::vector<NN_Layer> hiddenLayers, NN_Layer outLayer) {

	cublasStatus_t status;
	cudaError_t err;
	const char *str = "";

	// Get CUDA device
	int dev = findCudaDevice(0, &str);
	if (dev == -1) {
		return EXIT_FAILURE;
	}

	// Create CUBLAS Handle
	printf("INITIALIZING CUBLASS....");

	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLASS FAILED TO CREATE HANDLE!");
		return EXIT_FAILURE;
	}

	printf("CUBLASS INITIALIZED!");


	// Allocate Memory on device for layers
	printf("ALLOCATING DEVICE MEMORY...");

	err = cudaMalloc(reinterpret_cast<void**>(&(inLayer.inputs)), inLayer.Nodes * sizeof(float));

	if( err != cudaSuccess) {
		fprintf(stderr, "DEVICE MEMORY ALLOCATION ERROR: UNABLE TO ALLOCATE INPUT LAYER");
		return EXIT_FAILURE;
	}

	err = cudaMalloc(reinterpret_cast<void**>(&(outLayer.output)), outLayer.Nodes * sizeof(float));

	if( err != cudaSuccess) {
		fprintf(stderr, "DEVICE MEMORY ALLOCATION ERROR: UNABLE TO ALLOCATE OUTPUT LAYER");
		return EXIT_FAILURE;
	}

	err = cudaMalloc(reinterpret_cast<void**>(&(outLayer.bias)), outLayer.Nodes * sizeof(float));

	if( err != cudaSuccess) {
		fprintf(stderr, "DEVICE MEMORY ALLOCATION ERROR: UNABLE TO ALLOCATE OUTPUT LAYER");
		return EXIT_FAILURE;
	}

	err = cudaMalloc(reinterpret_cast<void**>(&(outLayer.error)), outLayer.Nodes * sizeof(float));

	if( err != cudaSuccess) {
		fprintf(stderr, "DEVICE MEMORY ALLOCATION ERROR: UNABLE TO ALLOCATE OUTPUT LAYER");
		return EXIT_FAILURE;
	}

	err = cudaMalloc(reinterpret_cast<void**>(&(outLayer.weights)), *outLayer.wRows * outLayer.wCols * sizeof(float));

	if( err != cudaSuccess) {
		fprintf(stderr, "DEVICE MEMORY ALLOCATION ERROR: UNABLE TO ALLOCATE OUTPUT LAYER");
		return EXIT_FAILURE;
	}


	for (auto i : hiddenLayers) {
		err = cudaMalloc(reinterpret_cast<void**>(&(i.output)), i.Nodes * sizeof(float));

		if( err != cudaSuccess) {
			fprintf(stderr, "DEVICE MEMORY ALLOCATION ERROR: UNABLE TO ALLOCATE HIDDEN LAYER %d", i);
			return EXIT_FAILURE;
		}

		err = cudaMalloc(reinterpret_cast<void**>(&(i.bias)), i.Nodes * sizeof(float));

		if( err != cudaSuccess) {
			fprintf(stderr, "DEVICE MEMORY ALLOCATION ERROR: UNABLE TO ALLOCATE OUTPUT LAYER");
			return EXIT_FAILURE;
		}

		err = cudaMalloc(reinterpret_cast<void**>(&(i.error)), i.Nodes * sizeof(float));

		if( err != cudaSuccess) {
			fprintf(stderr, "DEVICE MEMORY ALLOCATION ERROR: UNABLE TO ALLOCATE OUTPUT LAYER");
			return EXIT_FAILURE;
		}

		err = cudaMalloc(reinterpret_cast<void**>(&(i.weights)), *i.wRows * i.wCols * sizeof(float));

		if( err != cudaSuccess) {
			fprintf(stderr, "DEVICE MEMORY ALLOCATION ERROR: UNABLE TO ALLOCATE OUTPUT LAYER");
			return EXIT_FAILURE;
		}
	}

	printf("MEMORY ALLOCATED!");

	return 0;



}




}
