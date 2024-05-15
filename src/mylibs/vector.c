#include "vector.h"

void normalise_float_matrix(float** X, uint32_t N, uint32_t M){
    // allocate on the stack the memory for the mean and variance
    float mean[M];
    float variance[M];
    
    // calculate the mean of each column
    for (uint32_t j = 0; j < M; j++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < N; i++) {
            sum += X[i][j];
        }
        mean[j] = sum / (float) N;
    }
    
    // calculate the variance of each column
    for (uint32_t j = 0; j < M; j++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < N; i++) {
            float diff = X[i][j] - mean[j];
            sum += diff * diff;
        }
        variance[j] = sum /  (FLOAT_EPS + (float) N);
    }
    
    // normalize each column
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < M; j++) {
            X[i][j] = (X[i][j] - mean[j]) / sqrt(variance[j]);
        }
    }
}