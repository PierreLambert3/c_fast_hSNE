#ifndef VECTOR_H
#define VECTOR_H
#include "system.h"
#include "constants_global.h"

// for each dimension M, the data is rescaled to have mean 0 and variance 1
void normalise_float_matrix(float** X, uint32_t N, uint32_t M);

// float Euclidean distance between two vectors of size M
float f_euclidean_sq(float* Xi, float* Xj, uint32_t M);

// float Euclidean distance between two vectors of knwon size Mld
float f_euclidean_sq_in_embedding(float* Xi, float* Xj);


// 2d kernel for tsne
float kernel_LD(float eucl_sq, float alpha);

// for fast powf 
float fast_logf(float a);
float fast_expf(float x);
float fast_powf(float a, float b);

#endif // VECTOR_H
