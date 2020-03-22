/*
 * NeuralNetworkTest.cpp
 *
 *  Created on: Feb 16, 2020
 *      Author: jwredhead
 */
// TODO Write Unit Tests For Neural Network
#include <iostream>
#include <array>
#include <math.h>
#include <gtest/gtest.h>
#include "NeuralNetwork.h"


struct data {
	float input[2];
	float answer;
};

class NeuralNetworkTest : public ::testing::Test {
public:
	virtual void SetUp () {
		// Instantiate Neural Network
		nn = new NeuralNetwork(2, 4, 1);

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
		training_data[0].answer = 0;
		training_data[1].answer = 1;
		training_data[2].answer = 1;
		training_data[3].answer = 0;

	}

	virtual void TearDown () {
		if (nn != nullptr) {
			delete nn;
			nn = nullptr;
		}
	}

	NeuralNetwork* nn;
	data training_data[4];
};

TEST_F(NeuralNetworkTest, TestTraining) {

	// SEED Random Number Generator
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<> dist(0, 3);

	data tSet;
	for (unsigned i=0; i < 1000000; ++i) {
		tSet = training_data[dist(mt)];
		nn->trainNetwork(tSet.input, &tSet.answer);
	}

	float pred;
	float tolerance = 0.01;
	nn->predict(training_data[0].input, &pred);
	EXPECT_NEAR(0,pred, tolerance);
	nn->predict(training_data[1].input, &pred);
	EXPECT_NEAR(1,pred, tolerance);
	nn->predict(training_data[2].input, &pred);
	EXPECT_NEAR(1,pred, tolerance);
	nn->predict(training_data[3].input, &pred);
	EXPECT_NEAR(0,pred, tolerance);
}
