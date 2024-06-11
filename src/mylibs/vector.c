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

float f_euclidean_sq_in_embedding(float* Xi, float* Xj){
    float sum = 0.0f;
    for (uint32_t i = 0; i < Mld; i++) {
        float diff = Xi[i] - Xj[i];
        sum += diff * diff;}
    return sum;
}

inline float fast_logf(float a) {
    float x = a - 1;
    float sum = 0;
    float term = x;
    for (int i = 1; i < 10; i++) {
        sum += term / i;
        term *= -x;
    }
    return sum;
}

inline float fast_expf(float x) {
    float sum = 1;
    float term = 1;
    for (int i = 1; i < 10; i++) {
        term *= x / i;
        sum += term;
    }
    return sum;
}

inline float fast_powf(float a, float b) {
    dying_breath("untested, dont use here because the distribution of a and b is not appropriate in this context (a is close to zero, b can be very big)");
    return fast_expf(b * fast_logf(a));
}

inline float kernel_LD(float eucl_sq, float alpha){
    if(USE_CUSTOM_QIJ_KERNEL){
       dying_breath("USE_CUSTOM_QIJ_KERNEL is not implemented");
    }
    else{
        return 1.0f / powf(1.0f + eucl_sq / alpha, alpha);
    }        
}