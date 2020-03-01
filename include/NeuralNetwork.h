/*
 * NeuralNetwork.h
 *
 *  Created on: Feb 16, 2020
 *      Author: jwredhead
 */

#ifndef SRC_NEURALNETWORK_H_
#define SRC_NEURALNETWORK_H_

#include <iostream>
#include <vector>
#include <random>
#include "Matrix.h"


enum class Activation_Function{
	SIGMOID,
	BI_SIGMOID,
	TANH
};

struct NN_Layer {
	unsigned Nodes;
	Matrix<float> weights;
	Matrix<float> bias;
	Matrix<float> output;
	Matrix<float> error;
};

struct IN_Layer {
	unsigned Nodes;
	Matrix<float> inputs;
};

class NeuralNetwork {

public:
	NeuralNetwork(unsigned inputNodes, unsigned hiddenNodes, unsigned outputNodes);
	NeuralNetwork(unsigned inputNodes, unsigned* hiddenNodes, unsigned hiddenLayers, unsigned outputNodes);
	~NeuralNetwork() = default;

	void setActivation(Activation_Function funct);
	Activation_Function getActivation();

	void feedForward(float* inputs);

	void trainNetwork(float* inputs, float* targets);

private:
	Activation_Function m_activation;
	IN_Layer m_inputLayer;
	NN_Layer m_outputLayer;
	std::vector<NN_Layer> m_hiddenLayers;

	const float learningRate = 0.1;

	// Activation Functions not defined in cmath
	float sigmoid (float x);
	float bi_sigmoid(float x);

	void initialize();
	void randomFill(std::uniform_real_distribution<float> dist, std::mt19937 mt, Matrix<float>* m);
	Matrix<float> runActivationFunction(Matrix<float> m);

};

// Ostream Operator
template<int T>
inline std::ostream& operator<<(std::ostream& os, const NeuralNetwork nn) {


	return os;
}

#endif /* SRC_NEURALNETWORK_H_ */
