#ifndef CUDA_THINGS_H
#define CUDA_THINGS_H
#include <stdint.h>
#include <stdio.h>
#include "constants_global.h"

extern "C"{

// inline device funtion to calculate the squared euclidean distance between two points
__device__ __forceinline__ float cuda_euclidean_sq(float* Xi, float* Xj){
    float eucl_sq = 0.0f;
    #pragma unroll
    for (uint32_t m = 0; m < Mld; m++) {
        float diff = Xi[m] - Xj[m];
        eucl_sq += diff * diff;
    }
    return eucl_sq;
}

// inline device function that computes the simplified Cauchy kernel
// kernel function : 1. / powf(1. + eucl_sq/alpha_cauchy, alpha_cauchy);
// carefull: UNSAFE, alpha needs to be strictly > 0
__device__ __forceinline__ float cuda_cauchy_kernel(float eucl_sq, float alpha){
    return 1.0f / powf(1.0f + eucl_sq/alpha, alpha);
}

// 1-dimensional grid of 1-dimensional blocks. each thread corresponds to a pair (i, k) with i < N and k < Khd
__global__ void interactions_K_HD(uint32_t N, uint32_t Khd, uint32_t N_distinct_i,float* dvc_Pij, float* dvc_Xld_nester, uint32_t* dvc_neighsHD, float* furthest_neighdists_LD, float Qdenom_EMA, float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_attraction, float* dvc_momenta_repulsion){
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // TODO: reduce contention & increase readability by using 2D blocks !!!!!
    // enables an easyt hierarchical approach tp the atomic operations
    // --> most of the the atomicAdd would then be done on the shared memory (much faster)
    // only one thread per 2nd dimension of the block would be responsible for the atomicAdd to global memory,
    // constraint on Khd: multiple of 32 -> would likely be faster
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    // ~~~~~~~~~~~~~~~~~~~ get i, k and j ~~~~~~~~~~~~~~~~~~~
    uint32_t global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_index >= N * Khd) {return;}
    uint32_t i    = global_thread_index / Khd;
    uint32_t i0   = (blockIdx.x * blockDim.x) / Khd; // the value of the smallest i in the block
    uint32_t k    = global_thread_index % Khd;
    uint32_t j    = dvc_neighsHD[i * Khd + k];

    // ~~~~~~~~~~~~~~~~~~~ Initialise shared memory ~~~~~~~~~~~~~~~~~~~
    // shared memory can be divided into 3 local blocks:
    // N_distinct_i*(2u*Mld) floats, contain for each distinct i, the aggregated update to momenta vectors for attraction and repulsion
    // N_distinct_i*Mld floats, contain for each distinct i, the Xi vector
    // (blockDim.x * (4u*Mld) floats).transpose, contain the update to momenta for attraction and repulsion for each thread, both for i and j
    extern __shared__ float smem[];
    float* attrac_momenta_aggregator_i = &smem[(i - i0) * (2u * Mld)];
    float* repuls_momenta_aggregator_i = &attrac_momenta_aggregator_i[Mld];
    float* Xi                          = &smem[(N_distinct_i * (2u * Mld)) + (i - i0) * Mld];
    float* attrac_momentum_update_i_T  = &smem[(N_distinct_i * (3u * Mld)) + threadIdx.x]; // carefull: transposed (stride = blockDim.x)
    float* attrac_momentum_update_j_T  = &attrac_momentum_update_i_T[Mld * blockDim.x]; // carefull: transposed (stride = blockDim.x)
    float* repuls_momentum_update_i_T  = &attrac_momentum_update_j_T[Mld * blockDim.x]; // carefull: transposed (stride = blockDim.x)
    float* repuls_momentum_update_j_T  = &repuls_momentum_update_i_T[Mld * blockDim.x]; // carefull: transposed (stride = blockDim.x)

    // ~~~~~~~~~~~~~~~~~~~ Initialise registers: Xj and furthest LDneighdists for i and j ~~~~~~~~~~~~~~~~~~~
    // also write Xi to shared memory, if it is the first time we see i
    float furthest_LDneighdist_i = __ldg(&furthest_neighdists_LD[i]);
    if(threadIdx.x == 0 || k == 0){ // fetch Xi from DRAM
        #pragma unroll
        for (uint32_t m = 0; m < Mld; m++) {
            Xi[m] = dvc_Xld_nester[i * Mld + m];}
    }
    __syncthreads();
    float  Xj[Mld];
    float furthest_LDneighdist_j = __ldg(&furthest_neighdists_LD[j]);
    #pragma unroll
    for (uint32_t m = 0; m < Mld; m++) { // fetch Xj from DRAM
        Xj[m] = dvc_Xld_nester[j * Mld + m];}

    // compute squared euclidean distance 
    float eucl_sq = cuda_euclidean_sq(Xi, Xj);
    // similarity in HD  ("prefetching" to read only cache, it's not a true prefetch but it interleaves fetching with compute)
    float pij     = __ldg(&dvc_Pij[i * Khd + k]);
    // similarity in LD
    float wij     = cuda_cauchy_kernel(eucl_sq, alpha_cauchy); // qij = wij / Qdenom_EMA

    // ~~~~~~~~~~~~~~~~~~~ attractive forces ~~~~~~~~~~~~~~~~~~~
    float common_gradient_multiplier = 2.0f * (pij - (wij / Qdenom_EMA)) * powf(wij, 1.0f / alpha_cauchy);
    #pragma unroll
    for(uint32_t m = 0; m < Mld; m++){
        float gradient = (Xi[m] - Xj[m]) * common_gradient_multiplier;
        attrac_momentum_update_i_T[m * blockDim.x] = -gradient; // i movement
        attrac_momentum_update_j_T[m * blockDim.x] =  gradient; // j movement
    }
    // atomic agregation of the momenta
    // i : on attrac_momenta_aggregator_i
    for(uint32_t m = 0; m < Mld; m++){
        atomicAdd(&attrac_momenta_aggregator_i[m], attrac_momentum_update_i_T[m * blockDim.x]);}
    // j : directly to global memory
    for(uint32_t m = 0; m < Mld; m++){
        atomicAdd(&dvc_momenta_attraction[j * Mld + m], attrac_momentum_update_j_T[m * blockDim.x]);}

    // ~~~~~~~~~~~~~~~~~~~ repulsive forces ~~~~~~~~~~~~~~~~~~~
    // bool do_repulsion = ... ;
    return;

/* 
    // atomic add momenta


    // prepare repulsion index
    // smem_idx0 += 2u * Mld;
    // if j is NOT part of i's LD_neighs, and i is not part of j's LD_neighs, then do repulsive forces
    bool do_repulsion = eucl_sq > reg_furthest_neighdist_LD_i && eucl_sq > reg_furthest_neighdist_LD_j;
    if(do_repulsion){
        // update the momenta resulting from repulsive forces
        float common_gradient_multiplier = 2.0f * (wij / Qdenom_EMA) * powf(wij, 1.0f / alpha_cauchy);
        #pragma unroll
        for(uint32_t m = 0; m < Mld; m++){
            float gradient = (reg_Xi[m] - reg_Xj[m]) * common_gradient_multiplier;
            // repulsion (i movement)
            smem[smem_idx0 + m]       = gradient;
            // repulsion (j movement)
            smem[smem_idx0 + Mld + m] = -gradient;
        } 
       // atomic add momenta
    }
    

    // save 2d kernel value 

    // save euclidean value

    // attraction: always 
    
    // dont forget synthreads before writing to global memory

    if(i == N-1)
        printf("Thread %d: i = %d, k = %d   (Khd %u   Mld: %u)  j: %u   eucl %f and simi: %f  (wij %f) \n", threadIdx.x, i, k, Khd, Mld, j, eucl_sq, wij / Qdenom_EMA, wij); 
    */
}

void fill_raw_momenta_launch_cuda(cudaStream_t stream_HD, cudaStream_t stream_LD, cudaStream_t stream_rand, uint32_t K_HD_block_size, uint32_t K_HD_n_blocks, uint32_t N, uint32_t Khd, float* dvc_Pij, float* dvc_Xld_nester, uint32_t* dvc_neighsHD, float* furthest_neighdists_LD, float Qdenom_EMA, float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_attraction, float* dvc_momenta_repulsion){
    
    // ajouter cudastream_t x3 en arguments
    
    // ~~~~~~~~~  Kernel 1: HD neighbours  ~~~~~~~~~
    // in the worst case scenario, how many different values of i will be processed by one block
    uint32_t max_nb_different_i = 2u + (K_HD_block_size) / Khd;
    // shared memory size
    uint32_t n_floats_in_smem = (max_nb_different_i * (2u * Mld)) + (max_nb_different_i * Mld) + (K_HD_block_size * (4u * Mld));
    uint32_t sharedMemorySize = (uint32_t) (sizeof(float) * n_floats_in_smem);
    printf("\n\nkernel carachteristics:\nn blocks %d, block size %d \nshared memory size %d bytes (%d floats) \n", K_HD_n_blocks, K_HD_block_size, sharedMemorySize, n_floats_in_smem);   
    interactions_K_HD<<<K_HD_n_blocks, K_HD_block_size, sharedMemorySize, stream_HD>>>(N, Khd, max_nb_different_i, dvc_Pij, dvc_Xld_nester, dvc_neighsHD, furthest_neighdists_LD, Qdenom_EMA, alpha_cauchy, dvc_Qdenom_elements, dvc_momenta_attraction, dvc_momenta_repulsion);
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