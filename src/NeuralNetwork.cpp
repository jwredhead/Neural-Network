/*
 * NeuralNetwork.cpp
 *
 *  Created on: Feb 16, 2020
 *      Author: jwredhead
 */

#include "NeuralNetwork.h"
#include <string.h>
#include <math.h>
#include <random>



NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) :
							m_inputNodes(inputNodes),
							m_outputNodes(outputNodes),
							m_hiddenLayers(1) {

	// SEED Random Number Generator
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> dist(-1.0, 1.0);

	// Initialize Activation function, default = tanh function
	m_activation = Activation_Function::TANH;

	// Allocate Memory for Weights and Bias Matrices
	m_inWeights = new Matrix<float>(hiddenNodes,inputNodes);
	m_inBias = new Matrix<float>(hiddenNodes,1);
	m_outWeights = new Matrix<float>(outputNodes,hiddenNodes);
	m_outBias = new Matrix<float>(outputNodes,1);

	m_hiddenNodes[0] = hiddenNodes;

	// Only 1 hidden layer so m_hiddenWeights stays empty
	if (!m_hiddenWeights.empty()) {
		m_hiddenWeights.clear();
	}

	// Set Weight matrices elements to random number between -1 and 1
	for (unsigned i=0; i < m_inWeights->getRows(); i++) {
		for(unsigned j=0; j < m_inWeights->getCols(); j++ ) {
			(*m_inWeights)(i,j) = dist(mt);
		}
	}

	for (unsigned i=0; i < m_inWeights->getRows(); i++) {
		for(unsigned j=0; j < m_inWeights->getCols(); j++ ) {
			(*m_outWeights)(i,j) = dist(mt);
		}
	}

	// Set Bias matrices elements to random number between -1 and 1
	for(unsigned i = 0; i < m_inBias->getSize(); i++) {
		(*m_inBias)(i,0) = dist(mt);
	}

	for(unsigned i = 0; i < m_outBias->getSize(); i++) {
		(*m_outBias)(i,0) = dist(mt);
	}

}

NeuralNetwork::NeuralNetwork(int inputNodes, int *hiddenNodes, int hiddenLayers, int outputNodes) :
									m_inputNodes(inputNodes),
									m_hiddenNodes(hiddenNodes),
									m_hiddenLayers(hiddenLayers),
									m_outputNodes(outputNodes) {

	// SEED Random Number Generator
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> dist(-1.0, 1.0);

	// Initialize Activation function, default = tanh function
	m_activation = Activation_Function::TANH;


	// Allocate Memory for Weights and Bias Matrices
	m_inWeights = new Matrix<float>(hiddenNodes[0],inputNodes);
	m_inBias = new Matrix<float>(hiddenNodes[0],1);
	m_outWeights = new Matrix<float>(outputNodes,hiddenNodes[hiddenLayers-1]);
	m_outBias = new Matrix<float>(outputNodes,1);
	if (!m_hiddenWeights.empty()) {
		m_hiddenWeights.clear();
	}
	if (hiddenLayers > 1) {
		Matrix<float>* temp;
		for (int i=1; i<hiddenLayers; i++) {
			temp = new Matrix<float>(hiddenNodes[i], hiddenNodes[i-1]);
			m_hiddenWeights.push_back(temp);
		}
	}
	if (!m_hiddenBias.empty()) {
		m_hiddenBias.clear();
	}
	if (hiddenLayers > 1) {
		Matrix<float>* temp;
		for (int i=1; i<hiddenLayers; i++) {
			temp = new Matrix<float>(hiddenNodes[i], 1);
			m_hiddenBias.push_back(temp);
		}
	}

	// Set Weight matrices elements to random number between -1 and 1
	for (unsigned i=0; i < m_inWeights->getRows(); i++) {
		for(unsigned j=0; j < m_inWeights->getCols(); j++ ) {
			(*m_inWeights)(i,j) = dist(mt);
		}
	}

	for (unsigned i=0; i < m_inWeights->getRows(); i++) {
		for(unsigned j=0; j < m_inWeights->getCols(); j++ ) {
			(*m_outWeights)(i,j) = dist(mt);
		}
	}

	for ( auto i : m_hiddenWeights) {
		for (unsigned j=0; j < i->getRows(); j++) {
			for(unsigned k=0; k < i->getCols(); k++ ) {
				(*i)(j,k) = dist(mt);
			}
		}
	}

	// Set Bias matrices elements to random number between -1 and 1
	for(unsigned i = 0; i < m_inBias->getSize(); i++) {
		(*m_inBias)(i,0) = dist(mt);
	}

	for(unsigned i = 0; i < m_outBias->getSize(); i++) {
		(*m_outBias)(i,0) = dist(mt);
	}

	for(auto i : m_hiddenBias) {
		for (unsigned j=0; j < i->getSize(); j++) {
			(*i)(j,0) = dist(mt);
		}
	}


}

NeuralNetwork::~NeuralNetwork() {

	// De-allocate memory for Weight matrices
	if (m_inWeights != nullptr) {
		delete m_inWeights;
		m_inWeights = nullptr;
	}
	if (m_outWeights) {
		delete m_outWeights;
		m_outWeights = nullptr;
	}
	if (!m_hiddenWeights.empty()) {
		for (int i=0; i<m_hiddenWeights.size(); i++) {
			delete m_hiddenWeights[i];
		}
		m_hiddenWeights.clear();
	}

	// De-allocate memory for Bias matrices
	if (m_inBias != nullptr) {
		delete m_inBias;
		m_inWeights = nullptr;
	}
	if (m_outBias) {
		delete m_outBias;
		m_outBias = nullptr;
	}
	if (!m_hiddenBias.empty()) {
		for (int i=0; i<m_hiddenBias.size(); i++) {
			delete m_hiddenBias[i];
		}
		m_hiddenBias.clear();
	}

}

 void NeuralNetwork::setActivation(Activation_Function funct) {
	m_activation =  funct;
}

Activation_Function NeuralNetwork::getActivation() {
	return m_activation;
}


Matrix<float> NeuralNetwork::feedForward(float* inputs) {

	Matrix<float> matInputs(inputs, m_inputNodes);

	Matrix<float> hidden = (*m_inWeights) * matInputs;
	hidden += (*m_inBias);
	hidden = runActivationFunction(hidden);

	if(!m_hiddenWeights.empty()) {

		for (unsigned i=0; i<m_hiddenWeights.size(); i++) {
			hidden = (*(m_hiddenWeights[i])) * hidden;
			hidden += (*(m_hiddenBias[i]));
		}
	}

	Matrix<float> output = (*m_outWeights) * hidden;
	output += (*m_outBias);
	output = runActivationFunction(output);

	return output;
}

void NeuralNetwork::trainNetwork(float *inputs, float *targets) {

	Matrix<float> guesses = feedForward(inputs);

	Matrix<float> matTargets(targets, m_outputNodes);

	Matrix<float> outErrors = matTargets - guesses;

	std::vector<Matrix<float>*> hiddenErrors;

	Matrix<float> lastHiddenError = (*m_outWeights).transpose() * outErrors;

	hiddenErrors.insert(hiddenErrors.begin(), &lastHiddenError);

	if (!m_hiddenWeights.empty()) {
		for ( auto i : m_hiddenWeights) {
			Matrix <float> hiddenError_i = i->transpose() * *(hiddenErrors.front());
			hiddenErrors.insert(hiddenErrors.begin(), &hiddenError_i);
		}
	}

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


float NeuralNetwork::sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

float NeuralNetwork::bi_sigmoid(float x) {
	return (1 - exp(-x) / (1 + exp(-x)));
}

