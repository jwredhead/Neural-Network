#include "NeuralNetwork.h"
#include "mnist/mnist_reader.hpp"
#include <iostream>
#include <chrono>

namespace nn = NeuralNetworkLib;

struct data {
	float input[784] = {0};
	float answers[10] = {0};
};

int main (int argc, char* argv[]) {

	// Initialize RNG
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<> dist(0, 3);

	// Load MNIST data
	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

	std::cout << "# of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "# of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "# of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "# of test labels = " << dataset.test_labels.size() << std::endl;

//
//    unsigned x;
//    float targets[dataset.training_labels.size()][10] = {0};
//    for ( auto i : dataset.training_labels) {
//    	x = i;
//    	targets[i][x] = 1;
//    }
//
//    float training_data[dataset.training_images.size()][dataset.training_images[0].size()];
//    for (unsigned i=0; i < dataset.training_images.size(); ++i) {
//    	std::copy(dataset.training_images[i].begin(), dataset.training_images[i].end(), training_data[i]);
//    }

    auto start = std::chrono::high_resolution_clock::now();
    int index, label;
    std::vector<uint8_t> thisRun;

    nn::NeuralNetwork *NN = new nn::NeuralNetwork(784, 100, 10);
    float data[784], answers[10];
    for (unsigned i=0; i < 100000; ++i) {
    	index = dist(mt);
    	thisRun = dataset.training_images[index];
    	for (unsigned j =0; j < 784; ++j) {
    		data[j] = thisRun[j];
    	}
    	label = dataset.training_labels[index];
    	answers[label] = 1;

    	NN->trainNetwork(data, answers);
    	answers[label] = 0;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Training Time: " << elapsed_seconds.count() << std::endl;

    float testinput[784];
    float output[10];
    float max = 0;
    int guess;
    int num_correct=0;


    thisRun = dataset.test_images[0];
    for (unsigned i =0; i < 784; ++i) {
    	testinput[i] = thisRun[i];
    }
    int answer = dataset.test_labels[0];
	std::cout << "Answer = "<< answer << std::endl;
	start = std::chrono::high_resolution_clock::now();
    NN->predict(testinput, output);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Prediction Time: " << elapsed_seconds.count() << std::endl;
    for (unsigned i=0; i < 10; ++i) {
    	std::cout << "Output[" << i << "]= " << output[i] << std::endl;
    }
//    for (unsigned i = 0; i < 10000; ++i) {
//        for (unsigned j =0; j < 784; ++j) {
//    		testinput[j] = dataset.test_images[i][j];
//    	}
//        NN->predict(testinput, output);
//
//        for (unsigned k = 0; k < 10; ++k) {
//        	if (output[k] > max) {
//        		max = output[k];
//        		guess = k;
//        	}
//        }
//        if (guess == dataset.test_labels[i]) {
//        	num_correct++;
//        }
//    }
//
//    float acc = num_correct / 10000.0;
//
//    std::cout << "ACCURACY: " << acc << std::endl;

    delete NN;
    return 0;
}
