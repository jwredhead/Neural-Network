#!/bin/bash
echo "INSTALLING LIBRARY"

FILE=lib/libNeuralNetwork.so

if [ -f "$FILE" ]; then
	cp include/NeuralNetwork.h ../TEXT_CONV/include
	cp $FILE ../TEXT_CONV/lib
	echo "$FILE INSTALLED"
else 
	echo "$FILE DOESN'T EXIST, ENSURE LIBRARY IS BUILT"
fi
