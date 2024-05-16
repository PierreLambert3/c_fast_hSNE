#include "vector.h"

void normalise_float_matrix(float** X, uint32_t N, uint32_t M){
    /* // prints the first 2 observations of the normalised data, new line every 28 pixels
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < M; j++) {
            printf("%f  ", X[i][j]);
            if (j % 28 == 27) {
                printf("\n");
            }
        }
        printf("\n");
    }
    dying_breath("normalise_float_matrix() done"); */


    // allocate on the stack the memory for the mean and variance
    double mean[M];
    double variance[M];
    // initialise the mean and variance
    for (uint32_t j = 0; j < M; j++){
        mean[j] = 0.0;
        variance[j] = 0.0;
    }
    // calculate the mean of each column
    for (uint32_t i = 0; i < N; i++){
        for (uint32_t j = 0; j < M; j++){
            mean[j] += (double) X[i][j];
        }
    }
    for(uint32_t j = 0; j < M; j++){
        mean[j] /= (double) N;
    }
    
    // calculate the variance of each column
    for (uint32_t i = 0; i < N; i++){
        for (uint32_t j = 0; j < M; j++){
            float diff = ((double)X[i][j]) - mean[j];
            variance[j] += diff * diff;
        }
    }
    for (uint32_t j = 0; j < M; j++){
        variance[j] /= (double) N;
        variance[j] = FLOAT_EPS + sqrt(variance[j]);
    }
    
    // normalize each column
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < M; j++) {
            X[i][j] = (X[i][j] - (float)mean[j]) / (float)variance[j];
        }
    }
}

float f_euclidean_sq(float* Xi, float* Xj, uint32_t M){
    float sum = 0.0f;
    for (uint32_t i = 0; i < M; i++) {
        float diff = Xi[i] - Xj[i];
        sum += diff * diff;}
    return sum;
}