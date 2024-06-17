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

__device__ void warpReduce_periodic_sumReduction_on_matrix(volatile float* matrix, uint32_t e, uint32_t prev_len, uint32_t stride, uint32_t Ncol){
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
        if((e + stride < prev_len)){
            #pragma unroll
            for(uint32_t m = 0u; m < Mld; m++){
                matrix[m*Ncol] += matrix[m*Ncol + stride];
            }
        }
    }
}

// this function computes the periodic sum on a matrix of dimension (Nrows, period*Nvectors) (= (Nrows, Ncol))
// accessing the row r, and element e of the nth vector : matrix[r * Ncol + n * period + e]
/*
visual example:

| _A_  1  2  3  4  5  6  7  8  9  | _C_ 11 12 13 14 15 16 17 18 19  | 20 21 22 23 24 25 26 27 28 29 |
| _B_  2  3  4  5  6  7  8  9 10  | 11  12 13 14 15 16 17 18 19 20 | 21 22 23 24 25 26 27 28 29 30 |
| 2    3  4  5  6  7  8  9 10 11  | 12  13 14 15 16 17 18 19 20 21 | 22 23 24 25 26 27 28 29 30 31 |
| 3    4  5  6  7  8  9 10 11 12  | 13  14 15 16 17 18 19 20 21 22 | 23 24 25 26 27 28 29 30 31 32 |
| 4    5  6  7  8  9 10 11 12 13  | 14  15 16 17 18 19 20 21 22 23 | 24 25 26 27 28 29 30 31 32 33 |
| 5    6  7  8  9 10 11 12 13 14  | 15  16 17 18 19 20 21 22 23 24 | 25 26 27 28 29 30 31 32 33 34 |
| 6    7  8  9 10 11 12 13 14 15  | 16  17 18 19 20 21 22 23 24 25 | 26 27 28 29 30 31 32 33 34 35 |

--> after this function, A will contain the sum (A + 1 + 2 + ... + 9) 
                         B will contain the sum (B + 2 + 3 + ... + 10) 
                         C will contain the sum (C + 11 + 12 + ... + 19)
                        ... and so on
*/
__device__ __forceinline__ void periodic_sumReduction_on_matrix(float* matrix, uint32_t Nrows, uint32_t Ncol, uint32_t period, uint32_t e){
    __syncthreads(); // just in case it wasn't done after writing
    uint32_t prev_len = 2u * period;
    uint32_t stride   = period;
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
        if((e + stride < prev_len)){
            #pragma unroll
            for(uint32_t m = 0u; m < Mld; m++){
                matrix[m*Ncol] += matrix[m*Ncol + stride];
            }
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore (volatile float* matrix prevents reordering)
    if(e + stride < prev_len){ 
        warpReduce_periodic_sumReduction_on_matrix(matrix, e, prev_len, stride, Ncol);
    }
}

/*
grid shape : 1-d with total number of threads >= N * Khd
block shape: (Khd, Ni)
*/
__global__ void interactions_K_HD(uint32_t N, float* dvc_Pij, float* dvc_Xld_nester,\
        uint32_t* dvc_neighsHD, float* furthest_neighdists_LD, float Qdenom_EMA,\
        float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_attraction,\
        float* dvc_momenta_repulsion){
    
    
    
    
    printf("revoir tous les indices car changement dans la forme de smem!!\n");
    printf("ensiute il faudra finir d ecrire en global memeory\n");
    printf("et ensuite un dernier check sur la réduction somme GT vs parallel\n");

    // ~~~~~~~~~~~~~~~~~~~ get i, k and j ~~~~~~~~~~~~~~~~~~~
    uint32_t Khd           = blockDim.x; // Khd is guaranteed to be >= 32u
    uint32_t Ni            = blockDim.y;
    uint32_t block_surface = blockDim.x * blockDim.y;
    uint32_t i0            = (block_surface * blockIdx.x) / Khd; // the value of the smallest i in the block
    uint32_t i             = i0 + threadIdx.y;
    if( i >= N ){return;} // out of bounds (no guarantee that N is a multiple of Ni)
    uint32_t k             = threadIdx.x;
    uint32_t j             = dvc_neighsHD[i * Khd + k]; 
    uint32_t tid           = threadIdx.x + threadIdx.y * Khd;

    // ~~~~~~~~~~~~~~~~~~~ Initialise registers: Xj and furthest LDneighdists for i and j ~~~~~~~~~~~~~~~~~~~
    float furthest_LDneighdist_j = __ldg(&furthest_neighdists_LD[j]);
    float Xj[Mld];
    #pragma unroll
    for (uint32_t m = 0u; m < Mld; m++) { // fetch Xj from DRAM
        Xj[m] = dvc_Xld_nester[j * Mld + m];}
    float furthest_LDneighdist_i = __ldg(&furthest_neighdists_LD[i]);

    // ~~~~~~~~~~~~~~~~~~~ Initialise shared memory ~~~~~~~~~~~~~~~~~~~
    /*shared memory can be divided into 3 blocks:
    Ni*(2u*Mld) floats, contain for each distinct i, the aggregated update to momenta vectors for attraction and repulsion
    Ni*Mld floats, contain for each distinct i, the Xi vector
    block_surface * (4u*Mld) floats, transposed. Contains the individual updates to momenta for attraction and repulsion for each thread, both for i and j*/
    extern __shared__ float smem[];
    float* Xi                   = &smem[Ni*(2u*Mld) + (i - i0) * Mld];
    float* momenta_update_i_T   = &smem[Ni*(3u*Mld) + tid];  // stride for changing m: block_surface
    float* momenta_update_j_T   = &momenta_update_i_T[block_surface * Mld]; // stride for changing m: block_surface
    if(k == 0){ // fetch Xi from DRAM  and  initialise aggregator
        #pragma unroll
        for (uint32_t m = 0u; m < Mld; m++) {
            Xi[m] = dvc_Xld_nester[i * Mld + m];
        }
    }
    __syncthreads();

    // ~~~~~~~~~~~~~~~~~~~ now for some computations: prepare pij and wij ~~~~~~~~~~~~~~~~~~~
    // compute squared euclidean distance 
    float eucl_sq = cuda_euclidean_sq(Xi, Xj);
    // similarity in HD
    float pij     = __ldg(&dvc_Pij[i * Khd + k]);
    // similarity in LD (qij = wij / Qdenom_EMA)
    float wij     = cuda_cauchy_kernel(eucl_sq, alpha_cauchy); 

    // ~~~~~~~~~~~~~~~~~~~ save wij for Qdenom computation ~~~~~~~~~~~~~~~~~~~
    dvc_Qdenom_elements[i * Khd + k] = wij; // HD kernel : offset = 0   ( N_elements_of_Qdenom = N * (Khd + Kld + NB_RANDOM_POINTS_FAR_REPULSION);)

    // ~~~~~~~~~~~~~~~~~~~ attractive forces: independant gradients ~~~~~~~~~~~~~~~~~~~
    // individual updates to momenta for attraction
    float powerthing = 2.0f * powf(wij, 1.0f / alpha_cauchy);
    float common_attraction_gradient_multiplier =  pij * powerthing;
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        float gradient = (Xi[m] - Xj[m]) * common_attraction_gradient_multiplier;
        momenta_update_i_T[m*block_surface] = -gradient; // i movement
        momenta_update_j_T[m*block_surface] =  gradient; // j movement
    }
    // aggregate the individual updates
    periodic_sumReduction_on_matrix(momenta_update_i_T, Mld, block_surface, Khd, k);
    // write aggregated to global memory for point i
    if(k == 0u){
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            atomicAdd(&dvc_momenta_attraction[i * Mld + m], momenta_update_i_T[m*block_surface]);
        }
    }
    // write individual updates to j attraction momenta
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        atomicAdd(&dvc_momenta_attraction[j * Mld + m], momenta_update_j_T[m*block_surface]);
    }

    // ~~~~~~~~~~~~~~~~~~~ repulsive forces ~~~~~~~~~~~~~~~~~~~
    // individual updates to momenta for repulsion
    bool do_repulsion = eucl_sq > furthest_LDneighdist_i && eucl_sq > furthest_LDneighdist_j; // do repulsion if not LD neighbours. 
    if(do_repulsion){ // the  conditional is annoying because there is no structure in the decision to do repulsion or not: x2 time taken
        float common_repulsion_gradient_multiplier  = -(wij / Qdenom_EMA) * powerthing;
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            float gradient = (Xi[m] - Xj[m]) * common_repulsion_gradient_multiplier;
            momenta_update_i_T[m*block_surface] = -gradient; // i movement
            momenta_update_j_T[m*block_surface] =  gradient; // j movement
        }
    }
    else{ // important to set to zero, else the aggregation will be wrong
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            momenta_update_i_T[m*block_surface] = 0.0f;
            momenta_update_j_T[m*block_surface] = 0.0f;
        }
    }
    // aggregate the individual updates
    periodic_sumReduction_on_matrix(momenta_update_i_T, Mld, block_surface, Khd, k);
    // TODO: write to the repulsion momentum



    // si pas do_repulsion: faut quand meme mettre des 0 sinon le calcul de la somme des moments est fausse
    // bool do_repulsion = ... ;
     /* 
    
    // probleme: mon parallel reduction utiliser des vecteurs de taille variable


    // atomic agregation of the momenta
    // i : on attrac_momenta_aggregator_i
    for(uint32_t m = 0; m < Mld; m++){
        atomicAdd(&attrac_momenta_aggregator_i[m], attrac_momentum_update_i_T[m * blockDim.x]);}
    
    // j : directly to global memory
    for(uint32_t m = 0; m < Mld; m++){
        atomicAdd(&dvc_momenta_attraction[j * Mld + m], attrac_momentum_update_j_T[m * blockDim.x]);}

    // ~~~~~~~~~~~~~~~~~~~ repulsive forces ~~~~~~~~~~~~~~~~~~~
    // si pas do_repulsion: faut quand meme mettre des 0 sinon le calcul de la somme des moments est fausse
    // bool do_repulsion = ... ; */
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

void fill_raw_momenta_launch_cuda(cudaStream_t stream_HD, cudaStream_t stream_LD, cudaStream_t stream_rand,\
     uint32_t* Kern_HD_blockshape, uint32_t* Kern_HD_gridshape, uint32_t N, uint32_t Khd, float* dvc_Pij,\
      float* dvc_Xld_nester, uint32_t* dvc_neighsHD, float* furthest_neighdists_LD, float Qdenom_EMA,\
       float alpha_cauchy, double* dvc_Qdenom_elements,\
        float* dvc_momenta_attraction, float* dvc_momenta_repulsion, float* dvc_momenta_repulsion_far){
    
    // ~~~~~~~~~  clear momenta (async)  ~~~~~~~~~
    cudaMemsetAsync(dvc_momenta_attraction, 0, N * Mld * sizeof(float), stream_HD);
    cudaMemsetAsync(dvc_momenta_repulsion, 0, N * Mld * sizeof(float), stream_LD);
    cudaMemsetAsync(dvc_momenta_repulsion_far, 0, N * Mld * sizeof(float), stream_rand);

    // ~~~~~~~~~  prepare kernel calls ~~~~~~~~~
    // Kernel 1: HD neighbours
    uint32_t Kern_HD_block_surface = Kern_HD_blockshape[0] * Kern_HD_blockshape[1]; // N threads per block
    if((Kern_HD_block_surface % 32) != 0){printf("\n\nError: block size must be a multiple of 32\n");return;}
    uint32_t Kern_HD_sharedMemorySize = (uint32_t) (sizeof(float) * ((Kern_HD_blockshape[1] * Mld) + (Kern_HD_block_surface * (2u * Mld))));
    dim3 Kern_HD_grid(Kern_HD_gridshape[0], Kern_HD_gridshape[1]);
    dim3 Kern_HD_block(Kern_HD_blockshape[0], Kern_HD_blockshape[1]);
    // Kernel 2: LD neighbours

    
    // ~~~~~~~~~  launch kernels (and wait for async memset to finish)  ~~~~~~~~~
    // kernel 1 : HD neighbours
    printf("kernel gid shape %u %u %u      block shape %u %u %u\n", Kern_HD_grid.x, Kern_HD_grid.y, Kern_HD_grid.z, Kern_HD_block.x, Kern_HD_block.y, Kern_HD_block.z);
    cudaStreamSynchronize(stream_HD); // wait for the momenta to clear
    interactions_K_HD<<<Kern_HD_grid, Kern_HD_block, Kern_HD_sharedMemorySize, stream_HD>>>(N, dvc_Pij, dvc_Xld_nester, dvc_neighsHD, furthest_neighdists_LD, Qdenom_EMA, alpha_cauchy, dvc_Qdenom_elements, dvc_momenta_attraction, dvc_momenta_repulsion);// launch the kernel 1
    



    // ~~~~~~~~~~~  sync all streams  ~~~~~~~~~
    cudaStreamSynchronize(stream_HD);
   
    // TODO: ascend to godhood by using pretch CUDA instruction in assembly (Fermi architecture)
    // the prefetch instruction is used to load the data from global memory to the L2 cache
}

}

#endif