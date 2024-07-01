#include "probabilities.h"
#include "dataset_loaders.h"


/* void load_mnist(uint32_t* N, uint32_t* M, float*** X, uint32_t** Y) {
    *N = 60000; // Number of samples
    *M = 50;    // Number of features per sample

    *Y   = malloc_uint32_t(*N, 0);
    *X   = malloc_float_matrix(*N, *M, -42.0f); 


    // Open files
    FILE* X_file = fopen("../datasets/MNIST/MNIST_PCA_X.bin", "r");
    FILE* Y_file = fopen("../datasets/MNIST/MNIST_PCA_Y.bin", "r");
    if (X_file == NULL || Y_file == NULL) {
        dying_breath("file not found");
    }

    // Read data
    for (uint32_t i = 0; i < *N; i++) {
        if (fread((*X)[i], sizeof(float), *M, X_file) != *M) {
            dying_breath("Error reading X value");
        }
        if (fread(&(*Y)[i], sizeof(uint32_t), 1, Y_file) != 1) {
            dying_breath("Error reading Y value");
        }
    }

    // Close files
    fclose(X_file);
    fclose(Y_file);
} */

void load_mnist(uint32_t* N, uint32_t* M, float** X, uint32_t* Y) {
    
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