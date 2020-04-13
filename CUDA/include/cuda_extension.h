#pragma once

#include "NeuralNetworkTypes.h"
#include <vector>

namespace cuda_extension {

enum class MATRIX_OP {
	NORMAL,
	TRANSPOSE
};

const unsigned MAX_THREADS_PER_BLOCK = 256;

void initLayers(IN_Layer inLayer, std::vector<NN_Layer> hiddenLayers, NN_Layer outLayer);

void deleteLayers(IN_Layer inLayer, std::vector<NN_Layer> hiddenLayers, NN_Layer outLayer);

void multiplyAccumulate(MATRIX_OP transA, float *A, MATRIX_OP transB, float *B, float *C, int m, int n, int k);

void multiply(MATRIX_OP transA, float *A, MATRIX_OP transB, float *B, float *C, int m, int n, int k);

void copyVector(float *src, float *dst, unsigned size);

void copyInputs(float *src, float *dst, unsigned size);

void activationFunction(float *x, unsigned size, Activation_Function f);

void adjustWeightsBias(NN_Layer layer, float *inputs, unsigned inputSize, Activation_Function f, float learningRate);

void calculateError(float *targets, float *outputs, float *error, unsigned size);

void getOutputs(float *d_outputs, float *h_outputs, unsigned size);
}
