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



NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) {

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

NeuralNetwork::NeuralNetwork(int inputNodes, int *hiddenNodes, int hiddenLayers, int outputNodes) {

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

//TODO Finish Feed Forward
void NeuralNetwork::feedForward(float* inputs, unsigned size) {

	float in[size+1] = {1};
	for (unsigned i; i < size; i++) {
		in[i] = inputs[i];
	}

	Matrix<float> matInputs(in, size+1);

}

float NeuralNetwork::sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

float NeuralNetwork::bi_sigmoid(float x) {
	return (1 - exp(-x) / (1 + exp(-x)));
}

