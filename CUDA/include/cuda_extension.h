#pragma once

#include "NeuralNetworkTypes.h"
#include <vector>

namespace cuda_extension {

enum class MATRIX_OP {
	NORMAL,
	TRANSPOSE
};

const unsigned MAX_THREADS_PER_BLOCK = 256;

int initLayers(IN_Layer inLayer, std::vector<NN_Layer> hiddenLayers, NN_Layer outLayer);

void deleteLayers(IN_Layer inLayer, std::vector<NN_Layer> hiddenLayers, NN_Layer outLayer);

void multiplyAccumulate(MATRIX_OP transA, float *A, MATRIX_OP transB, float *B, float *C, int m, int n, int k);

int copyVector(float *x, float *, unsigned size);

void copyInputs(float *x, float *y, unsigned size);



}
