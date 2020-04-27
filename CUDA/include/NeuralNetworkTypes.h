/*
 * NeuralNetworkTypes.h
 *
 *  Created on: Mar 1, 2020
 *      Author: jwredhead
 */

#pragma once


enum class Activation_Function{
	SIGMOID,
	BI_SIGMOID,
	TANH,
	SOFTMAX
};

struct NN_Layer {
	unsigned Nodes;
	unsigned wRows;
	unsigned wCols;
	float *weights;
	float *bias;
	float *output;
	float *error;
};

struct IN_Layer {
	unsigned Nodes;
	float *inputs;
};


