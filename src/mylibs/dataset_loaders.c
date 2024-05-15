#include "probabilities.h"
#include "dataset_loaders.h"

void load_mnist(uint32_t* N, uint32_t* M, float** X, uint32_t* Y) {
    FILE* file = fopen("../datasets/MNIST/mnist_train.csv", "r");
    if (file == NULL) {
        dying_breath("file not found");
    }
    // sets values into Xhd and Y, Y being th first integer on each line
    for (uint32_t i = 0; i < *N; i++) {
        if (fscanf(file, "%d,", &Y[i]) != 1) {
            dying_breath("Error reading Y value");
        }
        for (uint32_t j = 0; j < *M; j++) {
            if (fscanf(file, "%f,", &X[i][j]) != 1) {
                dying_breath("Error reading X value");
            }
            X[i][j] = X[i][j] / 255.0f;
        }
    }
}