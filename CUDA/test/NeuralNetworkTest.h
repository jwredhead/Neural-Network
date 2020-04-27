/*
 * NeuralNetworkTest.cpp
 *
 *  Created on: Feb 16, 2020
 *      Author: jwredhead
 */
// TODO Write Unit Tests For Neural Network
#include <iostream>
#include <array>
#include <gtest/gtest.h>
#include "NeuralNetwork.h"

namespace nn = NeuralNetworkLib;

struct data {
	float input[2];
	float answer[2];
};

unsigned deciferGuess(float* input, unsigned size) {
	unsigned output =0;
	float maxval= 0;
	for (unsigned i=0; i < size; ++i) {
		std::cout << "Output[" << i << "]= " << input[i] << std::endl;
		if (input[i] > maxval) {
			maxval = input[i];
			output = i;
		}
	}
	return output;
}

class NeuralNetworkTest : public ::testing::Test {
public:
	virtual void SetUp () {
		// Instantiate Neural Network
		m_NN = new nn::NeuralNetwork(2, 8, 2);

		// Uncomment to print out Neural Network
		//std::cout << *nn << std::endl;

		// Setup Training Data for the XOR problem
		// Set Inputs
		training_data[0].input[0] = 0;
		training_data[0].input[1] = 0;
		training_data[1].input[0] = 0;
		training_data[1].input[1] = 1;
		training_data[2].input[0] = 1;
		training_data[2].input[1] = 0;
		training_data[3].input[0] = 1;
		training_data[3].input[1] = 1;


		// Set Answers
		training_data[0].answer[0] = 1;
		training_data[0].answer[1] = 0;
		training_data[1].answer[0] = 0;
		training_data[1].answer[1] = 1;
		training_data[2].answer[0] = 0;
		training_data[2].answer[1] = 1;
		training_data[3].answer[0] = 1;
		training_data[3].answer[1] = 0;

	}

	virtual void TearDown () {
		if (m_NN != nullptr) {
			delete m_NN;
			m_NN =  nullptr;
		}
	}

	nn::NeuralNetwork* m_NN;
	data training_data[4];
};

TEST_F(NeuralNetworkTest, TestTraining) {

	// SEED Random Number Generator
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<> dist(0, 3);

	data tSet;
	for (unsigned i=0; i < 100000; ++i) {
		tSet = training_data[dist(mt)];
		m_NN->trainNetwork(tSet.input, tSet.answer);
	}

	float pred[2];
	unsigned guess;
	m_NN->predict(training_data[0].input, pred);
	guess = deciferGuess(pred, 2);
	EXPECT_EQ(0,guess);
	m_NN->predict(training_data[1].input, pred);
	guess = deciferGuess(pred, 2);
	EXPECT_EQ(1,guess);
	m_NN->predict(training_data[2].input, pred);
	guess = deciferGuess(pred, 2);
	EXPECT_EQ(1,guess);
	m_NN->predict(training_data[3].input, pred);
	guess = deciferGuess(pred, 2);
	EXPECT_EQ(0,guess);
}
