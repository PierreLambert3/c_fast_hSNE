#ifndef DATASETS_H
#define DATASETS_H

#include "system.h"

// load the mnist matrix from the file ../src/datasets/MNIST/mnist_train.csv, see ../src/datasets/MNIST/dataset_info.txt for int on the format
// void load_mnist(uint32_t* N, uint32_t* M, float*** X, uint32_t** Y);
void load_mnist(uint32_t* N, uint32_t* M, float** X, uint32_t* Y);

#endif // DATASETS_H