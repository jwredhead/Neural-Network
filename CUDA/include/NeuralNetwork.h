/*
 * NeuralNetwork.h
 *
 *  Created on: Feb 16, 2020
 *      Author: jwredhead
 */

#pragma once

#include "NeuralNetworkTypes.h"
#include <iostream>
#include <vector>
#include <random>

namespace NeuralNetworkLib {
	class NeuralNetwork {

	public:
		NeuralNetwork(unsigned inputNodes, unsigned hiddenNodes, unsigned outputNodes);
		NeuralNetwork(unsigned inputNodes, unsigned* hiddenNodes, unsigned hiddenLayers, unsigned outputNodes);
		~NeuralNetwork();

		void setActivationFunction(Activation_Function funct);
		Activation_Function getActivationFunction();

		void setLearningRate(float lr);
		float getLearningRate();

		void trainNetwork(float* inputs, float* targets);

		void predict(float* inputs, float* outputs);


	private:
		IN_Layer m_inputLayer;
		NN_Layer m_outputLayer;
		std::vector<NN_Layer> m_hiddenLayers;


		// Default Activation Function is tanh
		Activation_Function m_activation = Activation_Function::TANH;
		Activation_Function m_outputActivation = Activation_Function::SOFTMAX;

		// Default Learning Rate is 0.1
		float m_learningRate = 0.1;

		void feedForward(float* inputs);

		friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork nn);

	};
}

