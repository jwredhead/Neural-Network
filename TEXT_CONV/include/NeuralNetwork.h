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

namespace NeuralNetworkLib {
	class NeuralNetwork {

	public:
		NeuralNetwork(unsigned inputNodes, unsigned hiddenNodes, unsigned outputNodes);
		NeuralNetwork(unsigned inputNodes, unsigned* hiddenNodes, unsigned hiddenLayers, unsigned outputNodes);
		~NeuralNetwork() = default;

		void setActivationFunction(Activation_Function funct);
		Activation_Function getActivationFunction();

		void setOutputActivation(Activation_Function funct);
		Activation_Function getOutputActivation();

		void setLossFunction(Loss_Function funct);
		Loss_Function getLossFunction();

		void setLearningRate(float lr);
		float getLearningRate();

		void trainNetwork(float* inputs, float* targets);

		void predict(float* inputs, float* outputs);

		float getError();


	private:
		IN_Layer m_inputLayer;
		NN_Layer m_outputLayer;
		std::vector<NN_Layer> m_hiddenLayers;

		// Default Activation Function is tanh, default Loss is cross-entropy
		Activation_Function m_activation = Activation_Function::SIGMOID;
		Activation_Function m_outputActivation = Activation_Function::SOFTMAX;

		// Default Learning Rate is 0.1
		float m_learningRate = 0.1;

		// NN Internal functions
		void initialize();
		void feedForward(float* inputs);
		void randomFill(std::uniform_real_distribution<float> dist, std::mt19937 mt, Matrix<float>* m);
		Matrix<float> runActivationFunction(const Matrix<float>& m, Activation_Function funct);
		Matrix<float> runActivationDerivative(const Matrix<float>& m, Activation_Function funct);
		Matrix<float> calcOutputError(const Matrix<float>& output, const Matrix<float>& target, Activation_Function funct);

		friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork nn);

	};

#define INDENT "   "

	// Ostream Operator
	inline std::ostream& operator<<(std::ostream& os, const NeuralNetwork nn) {

		os << "*****Neural Network*****" << '\n'
			<< "Learning Rate: " << nn.m_learningRate << '\n'
			<< "Activation Function: " << nn.m_activation << '\n'
			<< "Layers: " << (nn.m_hiddenLayers.size() + 2) << '\n'
			<< INDENT << "Input Layer: " << '\n'
			<< nn.m_inputLayer << '\n'
			<< INDENT << "Output Layer: " << '\n'
			<< nn.m_outputLayer << '\n';

		for (unsigned i=0; i < nn.m_hiddenLayers.size(); i++) {
			os << INDENT << "Hidden Layer " << i << ":" << '\n'
				<< nn.m_hiddenLayers[i] << '\n';
		}

		return os;
	}

#undef INDENT
}
