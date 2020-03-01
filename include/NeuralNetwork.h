/*
 * NeuralNetwork.h
 *
 *  Created on: Feb 16, 2020
 *      Author: jwredhead
 */

#pragma once

#include <iostream>
#include <vector>
#include <random>
#include "Matrix.h"
#include "NeuralNetworkTypes.h"

class NeuralNetwork {

public:
	NeuralNetwork(unsigned inputNodes, unsigned hiddenNodes, unsigned outputNodes);
	NeuralNetwork(unsigned inputNodes, unsigned* hiddenNodes, unsigned hiddenLayers, unsigned outputNodes);
	~NeuralNetwork() = default;

	void setActivationFunction(Activation_Function funct);
	Activation_Function getActivationFunction();

	void setLearningRate(float lr);
	float getLearningRate();

	void trainNetwork(float* inputs, float* targets);

private:
	IN_Layer m_inputLayer;
	NN_Layer m_outputLayer;
	std::vector<NN_Layer> m_hiddenLayers;

	// Default Activation Function is tanh
	Activation_Function m_activation = Activation_Function::TANH;

	// Default Learning Rate is 0.1
	float m_learningRate = 0.1;

	// Activation Functions not defined in cmath
	float sigmoid (float x);
	float bi_sigmoid(float x);

	void initialize();
	void feedForward(float* inputs);
	void randomFill(std::uniform_real_distribution<float> dist, std::mt19937 mt, Matrix<float>* m);
	Matrix<float> runActivationFunction(const Matrix<float>& m);
	Matrix<float> calcGradient(const NN_Layer& l);

	friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork nn);

};

#define INDENT "   "

// Ostream Operator
inline std::ostream& operator<<(std::ostream& os, const NeuralNetwork nn) {

	os << "*****Neural Network*****" << std::endl
		<< "Learning Rate: " << nn.m_learningRate << std::endl
		<< "Activation Function: " << nn.m_activation << std::endl
		<< "Layers: " << (nn.m_hiddenLayers.size() + 2) << std::endl
		<< INDENT << "Input Layer: " << std::endl
		<< nn.m_inputLayer << std::endl
		<< INDENT << "Output Layer: " << std::endl
		<< nn.m_outputLayer << std::endl;

	for (unsigned i=0; i < nn.m_hiddenLayers.size(); i++) {
		os << INDENT << "Hidden Layer " << i << ":" << std::endl
			<< nn.m_hiddenLayers[i] << std::endl;
	}

	return os;
}

#undef INDENT
