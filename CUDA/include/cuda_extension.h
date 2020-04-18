#pragma once

#include "NeuralNetworkTypes.h"
#include <vector>

namespace cuda_extension {

enum class MATRIX_OP {
	NORMAL,
	TRANSPOSE
};

const float MAX_THREADS_PER_BLOCK = 256.0;

void initLayers(IN_Layer *inLayer, std::vector<NN_Layer> *hiddenLayers, NN_Layer *outLayer);

void deleteLayers(IN_Layer *inLayer, std::vector<NN_Layer> *hiddenLayers, NN_Layer *outLayer);

void multiplyAccumulate(MATRIX_OP transA, float *A, MATRIX_OP transB, float *B, float *C, int m, int n, int k);

void multiply(MATRIX_OP transA, float *A, MATRIX_OP transB, float *B, float *C, int A_rows, int A_cols, int B_rows);

void copyVector(NN_Layer *layer);

void copyInputs(float *src, IN_Layer *inLayer);

void activationFunction(float *x, unsigned size, Activation_Function f);

void adjustWeightsBias(NN_Layer *layer, float *inputs, unsigned inputSize, Activation_Function f, float learningRate);

void calculateError(float *targets, float *outputs, float *error, unsigned size);

void getOutputs(float *d_outputs, float *h_outputs, unsigned size);
}
