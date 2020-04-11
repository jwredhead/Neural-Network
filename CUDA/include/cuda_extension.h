#pragma once

#include "NeuralNetworkTypes.h"
#include <vector>

namespace cuda_extension {



int initLayers(IN_Layer inLayer, std::vector<NN_Layer> hiddenLayers, NN_Layer outLayer);


}
