#include "probabilities.h"
#include "dataset_loaders.h"

void load_mnist_2(uint32_t* N, uint32_t* M, float*** X, uint32_t** Y){
    FILE* X_file = fopen("../datasets/MNIST/MNIST_PCA_X.bin", "r");
    FILE* Y_file = fopen("../datasets/MNIST/MNIST_PCA_Y.bin", "r");
    if (X_file == NULL || Y_file == NULL) {
        dying_breath("file not found");
    }

    N[0] = 60000u;
    M[0] = 50u;

    Y[0] = malloc_uint32_t(*N, 0);
    X[0] = malloc_float_matrix(*N, *M, -42.0f);

    float* tmp_Y = malloc_float(*N, 0.0f);
    for (uint32_t i = 0; i < *N; i++) {
        if (fread(X[0][i], sizeof(float), *M, X_file) != *M) {
            dying_breath("Error reading X value");
        }
        if (fread(&tmp_Y[i], sizeof(float), 1, Y_file) != 1) {
            dying_breath("Error reading Y value");
        }
    }
    // convert Y to integer
    for (uint32_t i = 0; i < *N; i++) {
        Y[0][i] = (uint32_t)tmp_Y[i];
    }



    fclose(X_file);
    fclose(Y_file);
    free(tmp_Y);

}

void load_blobs(uint32_t desired_n, uint32_t* N, uint32_t* M, float*** X, uint32_t** Y){
    N[0] = desired_n;
    M[0] = 2u;

    X[0] = malloc_float_matrix(*N, *M, -42.0f);
    Y[0] = malloc_uint32_t(*N, 0);



    /* for (uint32_t i = 0; i < *N; i++) {
        if (i < *N / 2) {
            X[0][i][0] = rand_float_between(&global_rand_state, -1.0f, 0.0f);
            X[0][i][1] = rand_float_between(&global_rand_state, -1.0f, 0.0f);
            Y[0][i] = 0;
        } else {
            X[0][i][0] = rand_float_between(&global_rand_state, 0.0f, 1.0f);
            X[0][i][1] = rand_float_between(&global_rand_state, 0.0f, 1.0f);
            Y[0][i] = 1;
        }
    } */
}
