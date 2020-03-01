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
#include "Matrix.h"


enum class Activation_Function{
	SIGMOID,
	BI_SIGMOID,
	TANH
};

class NeuralNetwork {

public:
	NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes);
	NeuralNetwork(int inputNodes, int* hiddenNodes, int hiddenLayers, int outputNodes);
	~NeuralNetwork();

	void setActivation(Activation_Function funct);
	Activation_Function getActivation();

	Matrix<float> feedForward(float* inputs);

	void trainNetwork(float* inputs, float* targets);

private:
	Activation_Function m_activation;
	Matrix<float>* m_inWeights;
	Matrix<float>* m_inBias;
	Matrix<float>* m_outWeights;
	Matrix<float>* m_outBias;
	std::vector<Matrix<float>*> m_hiddenWeights;
	std::vector<Matrix<float>*> m_hiddenBias;

	int m_inputNodes;
	int m_outputNodes;
	int* m_hiddenNodes;
	int m_hiddenLayers;

	const float learningRate = 0.1;

	// Activation Functions not defined in cmath
	float sigmoid (float x);
	float bi_sigmoid(float x);

	Matrix<float> runActivationFunction(Matrix<float> m);
};

// Ostream Operator
template<int T>
inline std::ostream& operator<<(std::ostream& os, const NeuralNetwork nn) {


	return os;
}

#endif /* SRC_NEURALNETWORK_H_ */
