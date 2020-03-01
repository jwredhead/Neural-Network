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

	os << INDENT << INDENT << "Nodes: " << l.Nodes << std::endl
		<< INDENT << INDENT << "Weight Matrix" << l.weights << std::endl
		<< INDENT << INDENT << "Bias Matrix" << l.bias << std::endl
		<< INDENT << INDENT << "Output Matrix" << l.output << std::endl
		<< INDENT << INDENT << "Error Matrix" << l.error << std::endl;
	return os;
}

struct IN_Layer {
	unsigned Nodes;
	Matrix<float> inputs;
};

// Ostream Operator for Input Layer Struct
inline std::ostream& operator<<(std::ostream& os, const IN_Layer l) {

	os << INDENT << INDENT << "Nodes: " << l.Nodes << std::endl
		<< INDENT << INDENT << "Input_Matrix" << l.inputs << std::endl;
	return os;
}

#undef INDENT

