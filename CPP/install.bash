#!/bin/bash
echo "INSTALLING LIBRARY"

FILE=lib/libNeuralNetwork.so

if [ -f "$FILE" ]; then
	cp include/NeuralNetwork.h ../TEXT_CONV/include
	cp include/NeuralNetworkTypes.h ../TEXT_CONV/include
	cp include/Matrix.h ../TEXT_CONV/include
	cp include/MatrixException.h ../TEXT_CONV/include
	cp $FILE ../TEXT_CONV/lib
	echo "$FILE INSTALLED"
else 
	echo "$FILE DOESN'T EXIST, ENSURE LIBRARY IS BUILT"
fi
