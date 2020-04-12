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

	int error = ce::initLayers(m_inputLayer, m_hiddenLayers, m_outputLayer);

	if(error) {
		std::cout << "LAYERS FAILED TO INITIALIZE" << "\n";
	}
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

	int error = ce::initLayers(m_inputLayer, m_hiddenLayers, m_outputLayer);
	if(error) {
		std::cout << "LAYERS FAILED TO INITIALIZE" << "\n";
	}

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

//	feedForward(inputs);
//
//	Matrix<float> matTargets(targets, m_outputLayer.Nodes);
//
//	m_outputLayer.error = matTargets - m_outputLayer.output;
//
//	m_hiddenLayers.back().error = m_outputLayer.weights.transpose() * m_outputLayer.error;
//
//	for (unsigned i=(m_hiddenLayers.size()-2); i > -1; i--) {
//		m_hiddenLayers[i].error = m_hiddenLayers[i+1].weights.transpose() * m_hiddenLayers[i+1].error;
//	}
//
//	Matrix<float> gradients = calcGradient(m_outputLayer);
//	Matrix<float> deltaWeight = gradients * m_hiddenLayers.back().output.transpose();
//	m_outputLayer.weights = m_outputLayer.weights + deltaWeight;
//	m_outputLayer.bias += gradients;
//
//	gradients = calcGradient(m_hiddenLayers.front());
//	deltaWeight = gradients * m_inputLayer.inputs.transpose();
//	m_hiddenLayers.front().weights = m_hiddenLayers.front().weights + deltaWeight;
//	m_hiddenLayers.front().bias += gradients;
//
//	if (m_hiddenLayers.size() > 1) {
//		for (unsigned i=1; i < m_hiddenLayers.size(); i++) {
//			gradients = calcGradient(m_hiddenLayers[i]);
//			deltaWeight = gradients * m_hiddenLayers[i-1].output.transpose();
//			m_hiddenLayers[i].weights = m_hiddenLayers[i].weights + deltaWeight;
//			m_hiddenLayers[i].bias += gradients;
//		}
//	}

}

void NeuralNetwork::predict(float* inputs, float* outputs) {
//	feedForward(inputs);
//	unsigned num_outputs =  m_outputLayer.output.getSize();
//	for ( unsigned i=0; i < num_outputs; ++i) {
//		outputs[i] = m_outputLayer.output(i,0);
//	}
}

float NeuralNetwork::sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

float NeuralNetwork::bi_sigmoid(float x) {
	return (1 - exp(-x) / (1 + exp(-x)));
}

void NeuralNetwork::feedForward(float* inputs) {



//	for (unsigned i=0; i < m_inputLayer.Nodes; i++) {
//		m_inputLayer.inputs(i,0) = inputs[i];
//	}
//
//	m_hiddenLayers.front().output = m_hiddenLayers.front().weights * m_inputLayer.inputs;
//	m_hiddenLayers.front().output += (m_hiddenLayers.front().bias);
//	m_hiddenLayers.front().output = runActivationFunction(m_hiddenLayers.front().output);
//
//	if (m_hiddenLayers.size() > 1) {
//		for (unsigned i=0; i < m_hiddenLayers.size(); i++) {
//			m_hiddenLayers[i].output = m_hiddenLayers[i].weights * m_hiddenLayers[i].output;
//			m_hiddenLayers[i].output += m_hiddenLayers[i].bias;
//			m_hiddenLayers[i].output = runActivationFunction(m_hiddenLayers[i].output);
//		}
//	}
//
//	m_outputLayer.output = m_outputLayer.weights * m_hiddenLayers.back().output;
//	m_outputLayer.output += m_outputLayer.bias;
//	m_outputLayer.output = runActivationFunction(m_outputLayer.output);
}

//Matrix<float> NeuralNetwork::runActivationFunction(const Matrix<float>& m) {
//
//	Matrix<float> n(m.getRows(), m.getCols());
//	for (unsigned i=0; i< m.getRows(); i++) {
//		for (unsigned j=0; j< m.getCols(); j++) {
//			switch (m_activation) {
//				case Activation_Function::SIGMOID: 		n(i,j) = sigmoid(m(i,j)); break;
//				case Activation_Function::BI_SIGMOID: 	n(i,j) = bi_sigmoid(m(i,j)); break;
//				case Activation_Function::TANH: 		n(i,j) = tanh(m(i,j)); break;
//			}
//		}
//	}
//
//	return n;
//}
//
//Matrix<float> NeuralNetwork::calcGradient(const NN_Layer& l) {
//
//	Matrix<float> n(l.output.getRows(), l.output.getCols());
//	for (unsigned i=0; i< l.output.getRows(); i++) {
//		for (unsigned j=0; j< l.output.getCols(); j++) {
//			switch (m_activation) {
//				case Activation_Function::SIGMOID: 		n(i,j) = l.output(i,j) * (1 - l.output(i,j)); break;
//				case Activation_Function::BI_SIGMOID: 	n(i,j) = 2 * l.output(i,j) * (1 - l.output(i,j)) ;break;
//				case Activation_Function::TANH: 		n(i,j) = 1 - ( l.output(i,j) * l.output(i,j) );break;
//			}
//		}
//	}
//	n = n.hadamardProduct(l.error);
//	n = n * m_learningRate;
//
//	return n;
//}

