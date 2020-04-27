/*
 * NeuralNetworkTypes.h
 *
 *  Created on: Mar 1, 2020
 *      Author: jwredhead
 */

#pragma once

#include <map>
#include <string>
#include "Matrix.h"

#define INDENT "   "
namespace NeuralNetworkLib {

	enum class Activation_Function {
		SIGMOID,
		BI_SIGMOID,
		TANH,
		SOFTMAX
	};

	enum class Loss_Function {
		MSE,
		MAE,
		SVM,
		CEL
	};

	// Ostream Operator for Activation Function Enum
	inline std::ostream& operator<<(std::ostream& os, const Activation_Function af) {

		std::map<Activation_Function, std::string> map = {
				{Activation_Function::SIGMOID, "Sigmoid"},
				{Activation_Function::BI_SIGMOID, "Bipolar Sigmoid"},
				{Activation_Function::TANH, "Hyperbolic Tangent"},
				{Activation_Function::SOFTMAX, "Softmax"}
		};

		std::string s = map.at(af);
		os << s;
		return os;
	}

	struct NN_Layer {
		unsigned Nodes;
		Matrix<float> Weights;
		Matrix<float> Bias;
		Matrix<float> Output;
		Matrix<float> Error;
	};

	// Ostream Operator for Neural Network Layer Struct
	inline std::ostream& operator<<(std::ostream& os, const NN_Layer l) {

		os << INDENT << INDENT << "Nodes: " << l.Nodes << '\n'
			<< INDENT << INDENT << "Weight Matrix" << l.Weights << '\n'
			<< INDENT << INDENT << "Bias Matrix" << l.Bias << '\n'
			<< INDENT << INDENT << "Output Matrix" << l.Output << '\n'
			<< INDENT << INDENT << "Error Matrix" << l.Error << '\n';
		return os;
	}

	struct IN_Layer {
		unsigned Nodes;
		Matrix<float> inputs;
	};

	// Ostream Operator for Input Layer Struct
	inline std::ostream& operator<<(std::ostream& os, const IN_Layer l) {

		os << INDENT << INDENT << "Nodes: " << l.Nodes << '\n'
			<< INDENT << INDENT << "Input_Matrix" << l.inputs << '\n';
		return os;
	}
}
#undef INDENT

