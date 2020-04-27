/*
 * Util.h
 *
 *  Created on: Apr 25, 2020
 *      Author: jwredhead
 */

#pragma once
#include <math.h>


namespace NeuralNetworkLib {

	float sigmoid(float x) {
		return 1 / (1 + exp(-x));
	}

	float bi_sigmoid(float x) {
		return (1 - exp(-x) / (1 + exp(-x)));
	}

	float sigmoid_derivative(float x) {
		return (x * (1 - x));
	}

	float bi_sigmoid_derivative(float x) {
		return (2 * x * (1 - x));
	}

	float tanh_derivative(float x) {
		return (1 - pow(x, 2.0));
	}

	float softmaxCE_derivative(float output, float target) {
		return (output - target);
	}

	float expSum(const Matrix <float>& m) {
		float sum =0;
		for (unsigned i=0; i< m.getRows(); i++) {
			for (unsigned j = 0; j < m.getCols(); ++j) {
				sum += exp(m(i,j));
			}
		}
		return sum;
	}


}
