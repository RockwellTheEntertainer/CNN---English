#ifndef CNN_FUNC_H
#define CNN_FUNC_H

#include "data_format.h"


// Give batch_size vectors, each vector softmax the probability of forming multiple categories
std::vector<tensor> softmax(const std::vector<tensor>& input);

// batch_size samples, each sample is 0, 1, 2. For example, 1 gives [0.0, 1.0, 0.0, 0.0]
std::vector<tensor> one_hot(const std::vector<int>& labels, const int num_classes);

// Calculate the cross entropy loss for the output probability probs and the label label, return the loss value and the gradient of the return
std::pair<data_type, std::vector<tensor> > cross_entroy_backward(
const std::vector<tensor>& probs, const std::vector<tensor>& labels);

// Decimals become strings
std::string float_to_string(const float value, const int precision);

#endif //CNN_FUNC_H
