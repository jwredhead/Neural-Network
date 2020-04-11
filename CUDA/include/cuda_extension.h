#pragma once

#include "NeuralNetworkTypes.h"
#include <vector>

namespace cuda_extension {

const unsigned MAX_THREADS_PER_BLOCK = 256;

int initLayers(IN_Layer inLayer, std::vector<NN_Layer> hiddenLayers, NN_Layer outLayer);

}
