#ifndef CUDA_THINGS_H
#define CUDA_THINGS_H
#include <stdint.h>
#include <stdio.h>
#include "constants_global.h"

extern "C"{

// inline device funtion to calculate the squared euclidean distance between two points
__device__ __forceinline__ float euclidean_sq(float* regis_Xi, float* regis_Xj){
    float eucl_sq = 0.0f;
    #pragma unroll
    for (uint32_t m = 0; m < Mld; m++) {
        float diff = regis_Xi[m] - regis_Xj[m];
        eucl_sq += diff * diff;
    }
    return eucl_sq;
}

// inline device function that computes the simplified Cauchy kernel
// kernel function : 1. / powf(1. + eucl_sq/alpha_cauchy, alpha_cauchy);
// carefull: UNSAFE, alpha needs to be strictly > 0
__device__ __forceinline__ float cauchy_kernel(float eucl_sq, float alpha){
    return 1. / powf(1. + eucl_sq/alpha, alpha);
}

// 1-dimensional grid of 1-dimensional blocks. each thread is a set (i, k) with i < N and k < Khd
__global__ void interactions_K_HD(int N, int Khd, int PAD, float* dvc_Pij, float* dvc_Xld_nester, uint32_t* dvc_neighsHD, float* furthest_neighdists_LD, float Qdenom_EMA, float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_attraction, float* dvc_momenta_repulsion){
    // get i, k and j 
    uint32_t global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_index >= N * Khd) return;
    uint32_t i = global_thread_index / Khd;
    uint32_t k = global_thread_index % Khd;
    uint32_t j = dvc_neighsHD[i * Khd + k];

    // Declare shared memory
    extern __shared__ float smem[];
    uint32_t smem_idx0 = (4u * Mld + PAD) * (uint32_t)threadIdx.x; // Calculate the starting index for this thread
    
    // registers, and fetching
    float reg_Xi[Mld];
    float reg_Xj[Mld];
    #pragma unroll
    for (uint32_t m = 0; m < Mld; m++) {
        reg_Xi[m] = dvc_Xld_nester[i * Mld + m];
        reg_Xj[m] = dvc_Xld_nester[j * Mld + m];
    }

    // compute squared euclidean distance 
    float eucl_sq = euclidean_sq(reg_Xi, reg_Xj);
    // similarity in HD  ("prefetching" to read only cache, it's not a true prefetch but it interleaves fetching with compute)
    float pij     = __ldg(&dvc_Pij[i * Khd + k]);
    // similarity in LD
    float wij     = cauchy_kernel(eucl_sq, alpha_cauchy); // qij = wij / Qdenom_EMA
    
    // update the momenta resulting from attractive forces
    float common_gradient_multiplier = 2.0f * (pij - (wij / Qdenom_EMA)) * powf(wij, 1.0f / alpha_cauchy);
    #pragma unroll
    for(uint32_t m = 0; m < Mld; m++){
        float gradient = (reg_Xi[m] - reg_Xj[m]) * common_gradient_multiplier;
        // attraction (i movement)
        smem[smem_idx0 + m]       = -gradient;
        // attraction (j movement)
        smem[smem_idx0 + Mld + m] = gradient;
    }

    // prepare repulsion
    smem_idx0 += 2u * Mld;
    
    

    // save 2d kernel value 

    // save euclidean value

    // atomic add momenta

    


    if(i == N-1)
        printf("Thread %d: i = %d, k = %d   (Khd %u   Mld: %u)  j: %u   eucl %f and simi: %f \n", threadIdx.x, i, k, Khd, Mld, j, eucl_sq, wij / Qdenom_EMA); 
}

void fill_raw_momenta_launch_cuda(cudaStream_t stream_HD, cudaStream_t stream_LD, cudaStream_t stream_rand, int K_HD_block_size, int K_HD_n_blocks, int N, int Khd, float* dvc_Pij, float* dvc_Xld_nester, uint32_t* dvc_neighsHD, float* furthest_neighdists_LD, float Qdenom_EMA, float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_attraction, float* dvc_momenta_repulsion){
    
    // ajouter cudastream_t x3 en arguments
    
    // ~~~~~~~~~  Kernel 1: HD neighbours  ~~~~~~~~~
    // Calculate padding
    uint32_t PAD = 0;
    while ((4u * Mld + PAD) % 32 == 0) {
        PAD++;}
    // shared memory size
    int sharedMemorySize = (int) (sizeof(float) * ((unsigned long)(((uint32_t)K_HD_block_size) * (4u * Mld + PAD))));
    printf("kernel carachteristics: n blocks %d, block size %d, shared memory size %d  PAD %d\n", K_HD_n_blocks, K_HD_block_size, sharedMemorySize, PAD);   

    interactions_K_HD<<<K_HD_n_blocks, K_HD_block_size, sharedMemorySize, stream_HD>>>(N, Khd, PAD, dvc_Pij, dvc_Xld_nester, dvc_neighsHD, furthest_neighdists_LD, Qdenom_EMA, alpha_cauchy, dvc_Qdenom_elements, dvc_momenta_attraction, dvc_momenta_repulsion);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("\n\nError: %s\n", cudaGetErrorString(err));}


    // ~~~~~~~~~~~  sync streams  ~~~~~~~~~
    cudaStreamSynchronize(stream_HD);





    // TODO: become a god by using pretch CUDA instruction in assembly (Fermi)
    // the prefetch instruction is used to load the data from global memory to the L2 cache
}

}

#endif