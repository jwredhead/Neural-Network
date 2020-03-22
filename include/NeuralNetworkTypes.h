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

enum class Activation_Function{
	SIGMOID,
	BI_SIGMOID,
	TANH
};

// Ostream Operator for Activation Function Enum
inline std::ostream& operator<<(std::ostream& os, const Activation_Function af) {

	std::map<Activation_Function, std::string> map = {
			{Activation_Function::SIGMOID, "Sigmoid"},
			{Activation_Function::BI_SIGMOID, "Bipolar Sigmoid"},
			{Activation_Function::TANH, "Hyperbolic Tangent"}
	};

	std::string s = map.at(af);
	os << s;
	return os;
}

struct NN_Layer {
	unsigned Nodes;
	Matrix<float> weights;
	Matrix<float> bias;
	Matrix<float> output;
	Matrix<float> error;
};

// Ostream Operator for Neural Network Layer Struct
inline std::ostream& operator<<(std::ostream& os, const NN_Layer l) {

	os << INDENT << INDENT << "Nodes: " << l.Nodes << '\n'
		<< INDENT << INDENT << "Weight Matrix" << l.weights << '\n'
		<< INDENT << INDENT << "Bias Matrix" << l.bias << '\n'
		<< INDENT << INDENT << "Output Matrix" << l.output << '\n'
		<< INDENT << INDENT << "Error Matrix" << l.error << '\n';
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

#undef INDENT

