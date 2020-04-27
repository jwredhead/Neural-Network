/*
 * NeuralNetwork.cpp
 *
 *  Created on: Feb 16, 2020
 *      Author: jwredhead
 */

#include "NeuralNetwork.h"
#include "Util.h"
#include <string.h>


namespace NeuralNetworkLib {

	NeuralNetwork::NeuralNetwork(unsigned inputNodes, unsigned hiddenNodes, unsigned outputNodes) {

		m_inputLayer.Nodes = inputNodes;
		m_outputLayer.Nodes = outputNodes;
		NN_Layer hiddenLayer;
		hiddenLayer.Nodes = hiddenNodes;
		m_hiddenLayers.insert(m_hiddenLayers.begin(), hiddenLayer);

		initialize();

	}

	NeuralNetwork::NeuralNetwork(unsigned inputNodes, unsigned *hiddenNodes, unsigned hiddenLayers, unsigned outputNodes) {

		m_inputLayer.Nodes = inputNodes;
		m_outputLayer.Nodes = outputNodes;

		for (int i=0; i<hiddenLayers; i++) {
			NN_Layer hiddenLayer;
			hiddenLayer.Nodes = hiddenNodes[i];
			m_hiddenLayers.push_back(hiddenLayer);
		}

		initialize();

	}

	void NeuralNetwork::setActivationFunction(Activation_Function funct) {
		m_activation =  funct;
	}

	Activation_Function NeuralNetwork::getActivationFunction() {
		return m_activation;
	}

	void NeuralNetwork::setOutputActivation(Activation_Function funct) {
		m_outputActivation = funct;
	}

	Activation_Function NeuralNetwork::getOutputActivation() {
		return m_outputActivation;
	}

	void NeuralNetwork::setLearningRate(float lr) {
		m_learningRate = lr;
	}

	float NeuralNetwork::getLearningRate() {
		return m_learningRate;
	}

	void NeuralNetwork::trainNetwork(float *inputs, float *targets) {

		feedForward(inputs);

		Matrix<float> matTargets(1,m_outputLayer.Nodes);
		for (unsigned i=0; i < m_outputLayer.Nodes; i++) {
			matTargets(0,i) = targets[i];
		}
		m_outputLayer.Error = calcOutputError(m_outputLayer.Output, matTargets, m_outputActivation);
//		std::cout << m_outputLayer.Weights << m_outputLayer.Error << m_hiddenLayers.back().Output;
		m_hiddenLayers.back().Error = (m_outputLayer.Weights * m_outputLayer.Error).hadamardProduct(runActivationDerivative(m_hiddenLayers.back().Output.transpose(), m_activation));
		if(m_hiddenLayers.size() > 1) {
			for (unsigned i = (m_hiddenLayers.size() - 2); i >= 0; i--) {
					m_hiddenLayers[i].Error = (m_hiddenLayers[i+1].Weights * m_hiddenLayers[i+1].Error).hadamardProduct(runActivationDerivative(m_hiddenLayers[i].Output.transpose(), m_activation));
			}
		}

//		std::cout << m_hiddenLayers.front().Weights << m_hiddenLayers.front().Error << m_inputLayer.inputs << m_hiddenLayers.front().Bias;
		m_hiddenLayers.front().Weights -= (m_hiddenLayers.front().Error * m_inputLayer.inputs).transpose() * m_learningRate;
		m_hiddenLayers.front().Bias -= m_hiddenLayers.front().Error.transpose() * m_learningRate;
		if(m_hiddenLayers.size() > 1) {
			for (unsigned i = 1; i < m_hiddenLayers.size(); i++) {
				m_hiddenLayers[i].Weights -= (m_hiddenLayers[i].Error * m_hiddenLayers[i-1].Output).transpose() * m_learningRate;
				m_hiddenLayers[i].Bias -= m_hiddenLayers[i].Error.transpose() * m_learningRate;
			}
		}
//		std::cout << m_outputLayer.Weights << m_outputLayer.Error << m_hiddenLayers.back().Output << m_outputLayer.Bias;
		m_outputLayer.Weights -= (m_outputLayer.Error * m_hiddenLayers.back().Output).transpose() * m_learningRate;
		m_outputLayer.Bias -= m_outputLayer.Error.transpose() * m_learningRate;
	}

	void NeuralNetwork::predict(float* inputs, float* outputs) {
		feedForward(inputs);
		for ( unsigned i=0; i < m_outputLayer.Nodes; ++i) {
			outputs[i] = m_outputLayer.Output(0,i);
		}
	}

	float NeuralNetwork::getError() {
		float e;
		for (unsigned i=0; i < m_outputLayer.Nodes; ++i) {
			e += m_outputLayer.Error(0,i);
		}
		return e;
	}

	void NeuralNetwork::initialize() {

		// SEED Random Number Generator
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<float> dist(-1.0, 1.0);

		// Intialize Input layer with null input
		Matrix<float> x(1, m_inputLayer.Nodes);
		x.fill(0.0);
		m_inputLayer.inputs = x;

		// Initialize output layer with random weights, random bias, null output, and null error
		Matrix<float> randOutWeights(m_hiddenLayers.back().Nodes, m_outputLayer.Nodes);
		randomFill(dist, mt, &randOutWeights);
		m_outputLayer.Weights = randOutWeights;

		Matrix<float> randOutBias(1, m_outputLayer.Nodes);
		randomFill(dist, mt, &randOutBias);
		m_outputLayer.Bias = randOutBias;

		Matrix<float> nullOutput(1, m_outputLayer.Nodes);
		nullOutput.fill(0.0);
		m_outputLayer.Output = nullOutput;

		Matrix<float> nullOutErr(m_outputLayer.Nodes, 1);
		nullOutErr.fill(0.0);
		m_outputLayer.Error = nullOutErr;

		// Initialize first hidden layer with random weights, random bias, null output, and null error
		Matrix<float> randInWeights(m_inputLayer.Nodes, m_hiddenLayers.front().Nodes);
		randomFill(dist, mt, &randInWeights);
		m_hiddenLayers.front().Weights = randInWeights;

		Matrix<float> randInBias(1, m_hiddenLayers.front().Nodes);
		randomFill(dist, mt, &randInBias);
		m_hiddenLayers.front().Bias = randInBias;

		Matrix<float> nullHdnOut(1, m_hiddenLayers.front().Nodes);
		nullHdnOut.fill(0.0);
		m_hiddenLayers.front().Output = nullHdnOut;

		Matrix<float> nullHdnErr(m_hiddenLayers.front().Nodes, 1);
		nullHdnErr.fill(0.0);
		m_hiddenLayers.front().Error = nullHdnErr;

		// If more hidden layers exist, initialize all hidden layers with random weights, random bias, null output, and null error
		if (m_hiddenLayers.size() >1) {
			for (unsigned i=1; i<m_hiddenLayers.size(); i++) {
				Matrix<float> randHdnWeights(m_hiddenLayers[i].Nodes, m_hiddenLayers[i-1].Nodes);
				randomFill(dist, mt, &randHdnWeights);
				m_hiddenLayers[i].Weights = randHdnWeights;

				Matrix<float> randHdnBias(1, m_hiddenLayers[i].Nodes);
				randomFill(dist, mt, &randHdnBias);
				m_hiddenLayers[i].Bias = randHdnBias;

				Matrix<float> nullHdn(1, m_hiddenLayers[i].Nodes);
				nullHdn.fill(0.0);
				m_hiddenLayers[i].Output = nullHdn;

				Matrix<float> nullHdnErr(m_hiddenLayers[i].Nodes, 1);
				nullHdnErr.fill(0.0);
				m_hiddenLayers[i].Error = nullHdnErr;
			}
		}

	}

	void NeuralNetwork::feedForward(float* inputs) {

		for (unsigned i=0; i < m_inputLayer.Nodes; i++) {
			m_inputLayer.inputs(0,i) = inputs[i];
		}

		m_hiddenLayers.front().Output =  m_inputLayer.inputs * m_hiddenLayers.front().Weights;
		m_hiddenLayers.front().Output += (m_hiddenLayers.front().Bias);
		m_hiddenLayers.front().Output = runActivationFunction(m_hiddenLayers.front().Output, m_activation);

		if (m_hiddenLayers.size() > 1) {
			for (unsigned i=0; i < m_hiddenLayers.size(); i++) {
				m_hiddenLayers[i].Output = m_hiddenLayers[i].Output * m_hiddenLayers[i].Weights;
				m_hiddenLayers[i].Output += m_hiddenLayers[i].Bias;
				m_hiddenLayers[i].Output = runActivationFunction(m_hiddenLayers[i].Output, m_activation);
			}
		}

		m_outputLayer.Output = m_hiddenLayers.back().Output * m_outputLayer.Weights;
		m_outputLayer.Output += m_outputLayer.Bias;
		m_outputLayer.Output = runActivationFunction(m_outputLayer.Output, m_outputActivation);

	}

	void NeuralNetwork::randomFill(std::uniform_real_distribution<float> dist, std::mt19937 mt, Matrix<float>* m) {

		for (unsigned i=0; i < m->getRows(); i++) {
			for (unsigned j=0; j < m->getCols(); j++) {
				(*m)(i,j) = dist(mt);
			}
		}
	}

	Matrix<float> NeuralNetwork::runActivationFunction(const Matrix<float>& m, Activation_Function funct) {

		float sum = 0;
		if (funct == Activation_Function::SOFTMAX) {
			sum = expSum(m);
		}

		Matrix<float> n(m.getRows(), m.getCols());
		for (unsigned i=0; i< m.getRows(); i++) {
			for (unsigned j=0; j< m.getCols(); j++) {
				switch (funct) {
					case Activation_Function::SIGMOID: 		n(i,j) = sigmoid(m(i,j)); break;
					case Activation_Function::BI_SIGMOID: 	n(i,j) = bi_sigmoid(m(i,j)); break;
					case Activation_Function::TANH: 		n(i,j) = tanh(m(i,j)); break;
					case Activation_Function::SOFTMAX:		n(i,j) = exp(m(i,j))/ sum; break;
				}
			}
		}

		return n;
	}

	Matrix<float> NeuralNetwork::runActivationDerivative(const Matrix<float>& output, Activation_Function funct) {

		Matrix<float> n(output.getRows(), output.getCols());
		for (unsigned i=0; i< output.getRows(); i++) {
			for (unsigned j=0; j< output.getCols(); j++) {
				switch (funct) {
					case Activation_Function::SIGMOID: 		n(i,j) = sigmoid_derivative(output(i,j)); break;
					case Activation_Function::BI_SIGMOID: 	n(i,j) = bi_sigmoid_derivative(output(i,j)); break;
					case Activation_Function::TANH: 		n(i,j) = tanh_derivative(output(i,j)); break;
				}
			}
		}

		return n;
	}

	Matrix<float> NeuralNetwork::calcOutputError(const Matrix<float>& output, const Matrix<float>& target, Activation_Function funct) {
		Matrix<float> n(output.getCols(), 1);
		if (funct == Activation_Function::SOFTMAX) {
			for (unsigned i=0; i< output.getCols(); i++) {
					n(i,0) = softmaxCE_derivative(output(0,i), target(0,i));
			}
		} else {
			for (unsigned i=0; i< output.getCols(); i++) {
					n(i,0) = output(0,i) * (output(0,i) - target(0,i));
			}
		}

		return n;
	}
}
