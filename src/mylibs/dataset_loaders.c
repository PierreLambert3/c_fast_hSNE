#include "probabilities.h"
#include "dataset_loaders.h"

void load_mnist(uint32_t* N, uint32_t* M, float** X, uint32_t* Y) {
    /* FILE* file = fopen("../datasets/MNIST/mnist_train.csv", "r");
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
    fclose(file); */
    FILE* X_file = fopen("../datasets/MNIST/MNIST_PCA_X.bin", "r");
    FILE* Y_file = fopen("../datasets/MNIST/MNIST_PCA_Y.bin", "r");
    if (X_file == NULL || Y_file == NULL) {
        dying_breath("file not found");
    }
    // there are 50 dimensions in this PCA version of the MNIST dataset, N and M are already allocated
    // Y is encoded is float, but it is actually an integer here because classification
    float* tmp_Y = malloc_float(*N, 0.0f);
    for (uint32_t i = 0; i < *N; i++) {
        if (fread(X[i], sizeof(float), *M, X_file) != *M) {
            dying_breath("Error reading X value");
        }
        if (fread(&tmp_Y[i], sizeof(float), 1, Y_file) != 1) {
            dying_breath("Error reading Y value");
        }
    }
    // convert Y to integer
    for (uint32_t i = 0; i < *N; i++) {
        Y[i] = (uint32_t)tmp_Y[i];
    }
    free(tmp_Y);

    fclose(X_file);
    fclose(Y_file);
}