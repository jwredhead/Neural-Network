/*
 * NeuralNetwork.cpp
 *
 *  Created on: Feb 16, 2020
 *      Author: jwredhead
 */

#include "NeuralNetwork.h"
#include <string.h>

namespace ce = cuda_extension;

NeuralNetwork::NeuralNetwork(unsigned inputNodes, unsigned hiddenNodes, unsigned outputNodes) {

	m_inputLayer.Nodes = inputNodes;
	m_outputLayer.Nodes = outputNodes;
	m_outputLayer.wCols = hiddenNodes;

	NN_Layer hiddenLayer;
	hiddenLayer.Nodes = hiddenNodes;
	m_hiddenLayers.insert(m_hiddenLayers.begin(), hiddenLayer);

	ce::initLayers(m_inputLayer, m_hiddenLayers, m_outputLayer);
}

NeuralNetwork::NeuralNetwork(unsigned inputNodes, unsigned *hiddenNodes, unsigned hiddenLayers, unsigned outputNodes) {

	m_inputLayer.Nodes = inputNodes;
	m_outputLayer.Nodes = outputNodes;

	NN_Layer hiddenLayer0;
	hiddenLayer0.Nodes = hiddenNodes[0];
	hiddenLayer0.wCols = inputNodes;
	m_hiddenLayers.push_back(hiddenLayer0);

	for (int i=1; i<hiddenLayers; i++) {
		NN_Layer hiddenLayer;
		hiddenLayer.Nodes = hiddenNodes[i];
		hiddenLayer.wCols = hiddenNodes[i-1];
		m_hiddenLayers.push_back(hiddenLayer);
	}

	m_outputLayer.wCols = m_hiddenLayers.back().Nodes;

	ce::initLayers(m_inputLayer, m_hiddenLayers, m_outputLayer);
}

NeuralNetwork::~NeuralNetwork() {
	ce::deleteLayers(m_inputLayer, m_hiddenLayers, m_outputLayer);
}


 void NeuralNetwork::setActivationFunction(Activation_Function funct) {
	m_activation =  funct;
}

Activation_Function NeuralNetwork::getActivationFunction() {
	return m_activation;
}


void NeuralNetwork::setLearningRate(float lr) {
	m_learningRate = lr;
}

float NeuralNetwork::getLearningRate() {
	return m_learningRate;
}

void NeuralNetwork::trainNetwork(float *inputs, float *targets) {

	feedForward(inputs);

	ce::calculateError(targets, m_outputLayer.output, m_outputLayer.error, m_outputLayer.Nodes);

	ce::multiply(ce::MATRIX_OP::TRANSPOSE, m_outputLayer.weights,
					ce::MATRIX_OP::NORMAL, m_outputLayer.error,
					m_hiddenLayers.back().error,
					m_outputLayer.wCols,
					m_hiddenLayers.back().Nodes,
					*(m_outputLayer.wRows));

	for (unsigned i=(m_hiddenLayers.size()-2); i > -1; i--) {
		ce::multiply(ce::MATRIX_OP::TRANSPOSE, m_hiddenLayers[i+1].weights,
						ce::MATRIX_OP::NORMAL, m_hiddenLayers[i+1].error,
						m_hiddenLayers[i].error,
						m_hiddenLayers[i+1].wCols,
						m_hiddenLayers[i].Nodes,
						*(m_hiddenLayers[i+1].wRows));
	}

	ce::adjustWeightsBias(m_outputLayer, m_hiddenLayers.back().output, m_hiddenLayers.back().Nodes, m_activation, m_learningRate);

	ce::adjustWeightsBias(m_hiddenLayers.front(), m_inputLayer.inputs, m_inputLayer.Nodes, m_activation, m_learningRate);

	if (m_hiddenLayers.size() > 1) {
		for (unsigned i=1; i < m_hiddenLayers.size(); i++) {
			ce::adjustWeightsBias(m_hiddenLayers[i], m_hiddenLayers[i-1].output, m_hiddenLayers[i-1].Nodes, m_activation, m_learningRate);
		}
	}

}

void NeuralNetwork::predict(float* inputs, float* outputs) {

	feedForward(inputs);

	ce::getOutputs(m_outputLayer.output, outputs, m_outputLayer.Nodes);

}

float NeuralNetwork::sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

float NeuralNetwork::bi_sigmoid(float x) {
	return (1 - exp(-x) / (1 + exp(-x)));
}

void NeuralNetwork::feedForward(float* inputs) {

	ce::copyInputs(inputs, m_inputLayer.inputs, m_inputLayer.Nodes);

	ce::copyVector(m_hiddenLayers.front().bias, m_hiddenLayers.front().output, m_hiddenLayers.front().Nodes);
	ce::multiplyAccumulate(ce::MATRIX_OP::NORMAL,m_hiddenLayers.front().weights,
							ce::MATRIX_OP::NORMAL, m_inputLayer.inputs,
							m_hiddenLayers.front().output,
							*(m_hiddenLayers.front().wRows),
							m_inputLayer.Nodes,
							m_hiddenLayers.front().wCols);
	ce::activationFunction(m_hiddenLayers.front().output, m_hiddenLayers.front().Nodes, m_activation);

	if (m_hiddenLayers.size() > 1) {
		for (unsigned i=1; i < m_hiddenLayers.size(); i++) {
			ce::copyVector(m_hiddenLayers[i].bias, m_hiddenLayers[i].output, m_hiddenLayers[i].Nodes);
			ce::multiplyAccumulate(ce::MATRIX_OP::NORMAL,m_hiddenLayers[i].weights,
					ce::MATRIX_OP::NORMAL, m_hiddenLayers[i-1].output,
					m_hiddenLayers[i].output,
					*(m_hiddenLayers[i].wRows),
					m_hiddenLayers[i-1].Nodes,
					m_hiddenLayers[i].wCols);
			ce::activationFunction(m_hiddenLayers[i].output, m_hiddenLayers[i].Nodes, m_activation);

		}
	}

	ce::copyVector(m_outputLayer.bias, m_outputLayer.output, m_hiddenLayers.back().Nodes);
	ce::multiplyAccumulate(ce::MATRIX_OP::NORMAL,m_outputLayer.weights,
							ce::MATRIX_OP::NORMAL, m_hiddenLayers.back().output,
							m_outputLayer.output,
							*(m_outputLayer.wRows),
							m_hiddenLayers.back().Nodes,
							m_outputLayer.wCols);
	ce::activationFunction(m_outputLayer.output, m_outputLayer.Nodes, m_activation);

}
