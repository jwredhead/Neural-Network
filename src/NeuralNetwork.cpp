/*
 * NeuralNetwork.cpp
 *
 *  Created on: Feb 16, 2020
 *      Author: jwredhead
 */

#include "NeuralNetwork.h"
#include <string.h>
#include <math.h>



NeuralNetwork::NeuralNetwork(unsigned inputNodes, unsigned hiddenNodes, unsigned outputNodes) {

	// Initialize Activation function, default = tanh function
	m_activation = Activation_Function::TANH;

	m_inputLayer.Nodes = inputNodes;
	m_outputLayer.Nodes = outputNodes;
	NN_Layer hiddenLayer;
	hiddenLayer.Nodes = hiddenNodes;
	m_hiddenLayers.insert(m_hiddenLayers.begin(), hiddenLayer);

	initialize();

}

NeuralNetwork::NeuralNetwork(unsigned inputNodes, unsigned *hiddenNodes, unsigned hiddenLayers, unsigned outputNodes) {

	// Initialize Activation function, default = tanh function
	m_activation = Activation_Function::TANH;

	m_inputLayer.Nodes = inputNodes;
	m_outputLayer.Nodes = outputNodes;

	for (int i=0; i<hiddenLayers; i++) {
		NN_Layer hiddenLayer;
		hiddenLayer.Nodes = hiddenNodes[i];
		m_hiddenLayers.push_back(hiddenLayer);
	}

	initialize();

}

 void NeuralNetwork::setActivation(Activation_Function funct) {
	m_activation =  funct;
}

Activation_Function NeuralNetwork::getActivation() {
	return m_activation;
}


void NeuralNetwork::feedForward(float* inputs) {

	for (unsigned i=0; i < m_inputLayer.Nodes; i++) {
		m_inputLayer.inputs(i,0) = inputs[i];
	}

	m_hiddenLayers.front().output = m_hiddenLayers.front().weights * m_inputLayer.inputs;
	m_hiddenLayers.front().output += (m_hiddenLayers.front().bias);
	m_hiddenLayers.front().output = runActivationFunction(m_hiddenLayers.front().output);

	if (m_hiddenLayers.size() > 1) {
		for (unsigned i=0; i < m_hiddenLayers.size(); i++) {
			m_hiddenLayers[i].output = m_hiddenLayers[i].weights * m_hiddenLayers[i].output;
			m_hiddenLayers[i].output += m_hiddenLayers[i].bias;
			m_hiddenLayers[i].output = runActivationFunction(m_hiddenLayers[i].output);
		}
	}

	m_outputLayer.output = m_outputLayer.weights * m_hiddenLayers.back().output;
	m_outputLayer.output += m_outputLayer.bias;
	m_outputLayer.output = runActivationFunction(m_outputLayer.output);
}

void NeuralNetwork::trainNetwork(float *inputs, float *targets) {

	feedForward(inputs);

	Matrix<float> matTargets(targets, m_outputLayer.Nodes);

	m_outputLayer.error = matTargets - m_outputLayer.output;

	m_hiddenLayers.back().error = m_outputLayer.weights.transpose() * m_outputLayer.error;

	for (unsigned i=(m_hiddenLayers.size()-2); i > -1; i--) {
		m_hiddenLayers[i].error = m_hiddenLayers[i+1].weights.transpose() * m_hiddenLayers[i+1].error;
	}

}

float NeuralNetwork::sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

float NeuralNetwork::bi_sigmoid(float x) {
	return (1 - exp(-x) / (1 + exp(-x)));
}

Matrix<float> NeuralNetwork::runActivationFunction(Matrix<float> m) {

	for (int i=0; i< m.getRows(); i++) {
		for (int j=0; j< m.getCols(); j++) {
			switch (m_activation) {
				case Activation_Function::SIGMOID: 		m(i,j) = sigmoid(m(i,j)); break;
				case Activation_Function::BI_SIGMOID: 	m(i,j) = bi_sigmoid(m(i,j)); break;
				case Activation_Function::TANH: 		m(i,j) = tanh(m(i,j)); break;
			}
		}
	}

	return m;
}

void NeuralNetwork::initialize() {

	// SEED Random Number Generator
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> dist(-1.0, 1.0);

	// Intialize Input layer with null input
	Matrix<float> x(m_inputLayer.Nodes, 1);
	x.fill(0.0);
	m_inputLayer.inputs = x;

	// Initialize output layer with random weights, random bias, null output, and null error
	Matrix<float> randOutWeights(m_outputLayer.Nodes, m_hiddenLayers.back().Nodes);
	randomFill(dist, mt, &randOutWeights);
	m_outputLayer.weights = randOutWeights;

	Matrix<float> randOutBias(m_outputLayer.Nodes, 1);
	randomFill(dist, mt, &randOutBias);
	m_outputLayer.bias = randOutBias;

	Matrix<float> nullOutput(m_outputLayer.Nodes, 1);
	nullOutput.fill(0.0);
	m_outputLayer.output = nullOutput;

	Matrix<float> nullOutErr(m_outputLayer.Nodes, 1);
	nullOutErr.fill(0.0);
	m_outputLayer.error = nullOutErr;

	// Initialize first hidden layer with random weights, random bias, null output, and null error
	Matrix<float> randInWeights(m_hiddenLayers.front().Nodes, m_inputLayer.Nodes);
	randomFill(dist, mt, &randInWeights);
	m_hiddenLayers.front().weights = randInWeights;

	Matrix<float> randInBias(m_hiddenLayers.front().Nodes, 1);
	randomFill(dist, mt, &randInBias);
	m_hiddenLayers.front().bias = randInBias;

	Matrix<float> nullHdnOut(m_hiddenLayers.front().Nodes, 1);
	nullHdnOut.fill(0.0);
	m_hiddenLayers.front().output = nullHdnOut;

	Matrix<float> nullHdnErr(m_hiddenLayers.front().Nodes, 1);
	nullHdnErr.fill(0.0);
	m_outputLayer.error = nullHdnErr;

	// If more hidden layers exist, initialize all hidden layers with random wieghts, reandom bias, null output, and null error
	if (m_hiddenLayers.size() >1) {
		for (unsigned i=1; i<m_hiddenLayers.size(); i++) {
			Matrix<float> randHdnWeights(m_hiddenLayers[i].Nodes, m_hiddenLayers[i-1].Nodes);
			randomFill(dist, mt, &randHdnWeights);
			m_hiddenLayers[i].weights = randHdnWeights;

			Matrix<float> randHdnBias(m_hiddenLayers[i].Nodes, 1);
			randomFill(dist, mt, &randHdnBias);
			m_hiddenLayers[i].bias = randHdnBias;

			Matrix<float> nullHdn(m_hiddenLayers[i].Nodes, 1);
			nullHdn.fill(0.0);
			m_hiddenLayers[i].output = nullHdn;

			Matrix<float> nullHdnErr(m_hiddenLayers[i].Nodes, 1);
			nullHdnErr.fill(0.0);
			m_outputLayer.error = nullHdnErr;
		}
	}

}

void NeuralNetwork::randomFill(std::uniform_real_distribution<float> dist, std::mt19937 mt, Matrix<float>* m) {

	for (unsigned i=0; i < m->getRows(); i++) {
		for (unsigned j=0; j < m->getCols(); j++) {
			(*m)(i,j) = dist(mt);
		}
	}
}

