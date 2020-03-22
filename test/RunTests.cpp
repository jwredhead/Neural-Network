/*
 * RunTests.cpp
 *
 *  Created on: Feb 14, 2020
 *      Author: jwredhead
 */
#include <iostream>
#include <gtest/gtest.h>
#include "MatrixTest.h"
#include "NeuralNetworkTest.h"


int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

