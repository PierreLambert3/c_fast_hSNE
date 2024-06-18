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
        warpReduce_periodic_sumReduction_on_matrix(matrix, e, prev_len, stride, Ncol);}
    __syncthreads(); // this one is not necessary
}

/*
visual representation of shared memory:
        |[----- Mld -----]|    each row contains Xi
        |        .        |                                           <----   this first block is of height Ni and width Mld
        |        .        |
        |[----- Mld -----]|
        --------------------------------------
        | M | M | M | M | ... | M | M | M | M |    
        | l | l | l | l | ... | l | l | l | l |    momentum for i (m1)
        | d | d | d | d | ... | d | d | d | d |        
        --------------------------------------                        <----  this second block is of height 2Mld and width block_surface (= Ni*N_RAND)
        | M | M | M | M | ... | M | M | M | M |    
        | l | l | l | l | ... | l | l | l | l |    momentum updates for j
        | d | d | d | d | ... | d | d | d | d |    
        --------------------------------------
 */
__global__ void interactions_far(uint32_t N, float* dvc_Xld_nester, float Qdenom_EMA,\
        float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_repulsion_far){
    // ~~~~~~~~~~~~~~~~~~~ get i, k and j ~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~ Initialise registers: Xj ~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~ Initialise shared memory ~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~ now for some computations:  wij ~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~ save wij for Qdenom computation (offset is already done) ~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~ repulsive forces far ~~~~~~~~~~~~~~~~~~~
    // DO NOT APPLY MOMENTA ON j, ONLY i (because else we don't have a guaranteed balance on the forces)

    // ~~~~~~~~~~~~~~~~~~~ save the new seed for next iteration ~~~~~~~~~~~~~~~~~~~


    return;
}




/* visual representation of shared memory:
        |[----- Mld -----]|    each row contains Xi
        |        .        |                                           <----   this first block is of height Ni and width Mld
        |        .        |
        |[----- Mld -----]|
        --------------------------------------
        | M | M | M | M | ... | M | M | M | M |    
        | l | l | l | l | ... | l | l | l | l |    momentum for i (m1)
        | d | d | d | d | ... | d | d | d | d |        
        --------------------------------------                        <----  this second block is of height 2Mld and width block_surface (= Ni*Kld)
        | M | M | M | M | ... | M | M | M | M |    
        | l | l | l | l | ... | l | l | l | l |    momentum updates for j
        | d | d | d | d | ... | d | d | d | d |    
        --------------------------------------
  for this 2nd block, each thread is a column (block_surface columns in total, (ie: Ni*Kld columns))
  each row of the second block is organised as such:
        |i0,k0| i0,k1 | i0,k2 | ... | i0,k(Kld-1) | i1,k0 | i1,k1 | i1,k2 | ... | i1,k(Kld-1) | ... | i(Ni-1),k(Kld-1)|
*/

__global__ void interactions_K_LD(uint32_t N, float* dvc_Xld_nester, uint32_t* dvc_neighsLD, float Qdenom_EMA,\
        float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_repulsion, float* all_neighdists_LD){
    // ~~~~~~~~~~~~~~~~~~~ get i, k and j ~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~ Initialise registers: Xj ~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~ Initialise shared memory ~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~ now for some computations:  wij (offset already done) ~~~~~~~~~~~~~~~~~~~
    // dvc_Qdenom_elements[(i * Kld + k)] = (double) wij;  

    // ~~~~~~~~~~~~~~~~~~~ save wij for Qdenom computation ~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~ saved euclidean distance to all_neighdists_LD[i*Kld + k] ~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~ repulsive forces ~~~~~~~~~~~~~~~~~~~

    return;
}


/*
grid shape : 1-d with total number of threads >= N * Khd
block shape: (Khd, Ni)

/* visual representation of shared memory:
        |[----- Mld -----]|    each row contains Xi
        |        .        |                                           <----   this first block is of height Ni and width Mld
        |        .        |
        |[----- Mld -----]|
        --------------------------------------
        | M | M | M | M | ... | M | M | M | M |    
        | l | l | l | l | ... | l | l | l | l |    momentum for i (m1)
        | d | d | d | d | ... | d | d | d | d |        
        --------------------------------------                        <----  this second block is of height 2Mld and width block_surface (= Ni*Khd)
        | M | M | M | M | ... | M | M | M | M |    
        | l | l | l | l | ... | l | l | l | l |    momentum updates for j
        | d | d | d | d | ... | d | d | d | d |    
        --------------------------------------
  for this 2nd block, each thread is a column (block_surface columns in total, (ie: Ni*Khd columns))
  each row of the second block is organised as such:
        |i0,k0| i0,k1 | i0,k2 | ... | i0,k(Khd-1) | i1,k0 | i1,k1 | i1,k2 | ... | i1,k(Khd-1) | ... | i(Ni-1),k(Khd-1)|
*/
__global__ void interactions_K_HD(uint32_t N, float* dvc_Pij, float* dvc_Xld_nester,\
        uint32_t* dvc_neighsHD, float* furthest_neighdists_LD, float Qdenom_EMA,\
        float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_attraction,\
        float* dvc_momenta_repulsion){
    // ~~~~~~~~~~~~~~~~~~~ get i, k and j ~~~~~~~~~~~~~~~~~~~
    uint32_t Khd           = blockDim.x; // block shape: (Khd, Ni);  Khd is guaranteed to be >= 32u
    uint32_t Ni            = blockDim.y; // block shape: (Khd, Ni)
    uint32_t block_surface = blockDim.x * blockDim.y;
    uint32_t i0            = (block_surface * blockIdx.x) / Khd; // the value of the smallest i in the block
    uint32_t k             = threadIdx.x;
    uint32_t i             = i0 + threadIdx.y;
    if( i >= N ){return;} // out of bounds (no guarantee that N is a multiple of Ni)
    uint32_t j             = dvc_neighsHD[i * Khd + k]; 
    uint32_t tid           = threadIdx.x + threadIdx.y * Khd; // index within the block

    // ~~~~~~~~~~~~~~~~~~~ Initialise registers: Xj and furthest LDneighdists for i and j ~~~~~~~~~~~~~~~~~~~
    float furthest_LDneighdist_j = __ldg(&furthest_neighdists_LD[j]);
    float Xj[Mld];
    #pragma unroll
    for (uint32_t m = 0u; m < Mld; m++) { // fetch Xj from DRAM
        Xj[m] = dvc_Xld_nester[j * Mld + m];}
    float furthest_LDneighdist_i = __ldg(&furthest_neighdists_LD[i]);

    // ~~~~~~~~~~~~~~~~~~~ Initialise shared memory ~~~~~~~~~~~~~~~~~~~
    extern __shared__ float smem[];
    float* Xi                   = &smem[(i - i0) * Mld];
    float* momenta_update_i_T   = &smem[Ni*Mld + tid];  // stride for changing m: block_surface
    float* momenta_update_j_T   = &smem[Ni*Mld + block_surface*Mld + tid]; // stride for changing m: block_surface
    if(k == 0){ // fetch Xi from DRAM
        #pragma unroll
        for (uint32_t m = 0u; m < Mld; m++) {
            Xi[m] = dvc_Xld_nester[i * Mld + m];}
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
    dvc_Qdenom_elements[(i * Khd + k)] = (double) wij; 

    // ~~~~~~~~~~~~~~~~~~~ attractive forces ~~~~~~~~~~~~~~~~~~~
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
            atomicAdd(&dvc_momenta_attraction[i * Mld + m], momenta_update_i_T[m*block_surface]);}
    }
    // write individual updates to j attraction momenta
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        atomicAdd(&dvc_momenta_attraction[j * Mld + m], momenta_update_j_T[m*block_surface]);}

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
    // write to global memory for point i
    if(k == 0u){
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            atomicAdd(&dvc_momenta_repulsion[i * Mld + m], momenta_update_i_T[m*block_surface]);}
    }
    // write individual updates to j repulsion momenta
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        atomicAdd(&dvc_momenta_repulsion[j * Mld + m], momenta_update_j_T[m*block_surface]);}
}

void fill_raw_momenta_launch_cuda(cudaStream_t stream_HD, cudaStream_t stream_LD, cudaStream_t stream_FAR,\
     uint32_t* Kern_HD_blockshape, uint32_t* Kern_HD_gridshape,uint32_t* Kern_LD_blockshape, uint32_t* Kern_LD_gridshape,uint32_t* Kern_FAR_blockshape, uint32_t* Kern_FAR_gridshape,\
      uint32_t N, uint32_t Khd, float* dvc_Pij,\
      float* dvc_Xld_nester, uint32_t* dvc_neighsHD, uint32_t* dvc_neighsLD, float* furthest_neighdists_LD, float Qdenom_EMA,\
       float alpha_cauchy, double* dvc_Qdenom_elements,\
        float* dvc_momenta_attraction, float* dvc_momenta_repulsion, float* dvc_momenta_repulsion_far, float* all_neighdists_LD, uint32_t* random_numbers_size_NxRand){
    
    // ~~~~~~~~~  clear momenta (async)  ~~~~~~~~~
    cudaMemsetAsync(dvc_momenta_attraction, 0, N * Mld * sizeof(float), stream_HD);
    cudaMemsetAsync(dvc_momenta_repulsion, 0, N * Mld * sizeof(float), stream_LD);
    cudaMemsetAsync(dvc_momenta_repulsion_far, 0, N * Mld * sizeof(float), stream_FAR);

    // ~~~~~~~~~  prepare kernel calls ~~~~~~~~~
    // Kernel 1: HD neighbours
    uint32_t Kern_HD_block_surface = Kern_HD_blockshape[0] * Kern_HD_blockshape[1]; // N threads per block
    // if((Kern_HD_block_surface % 32) != 0){printf("\n\nError: block size should be a multiple of 32\n");return;}
    uint32_t Kern_HD_sharedMemorySize = (uint32_t) (sizeof(float) * ((Kern_HD_blockshape[1] * Mld) + (Kern_HD_block_surface * (2u * Mld))));
    dim3 Kern_HD_grid(Kern_HD_gridshape[0], Kern_HD_gridshape[1]);
    dim3 Kern_HD_block(Kern_HD_blockshape[0], Kern_HD_blockshape[1]);
    // Kernel 2: LD neighbours
    uint32_t Kern_LD_block_surface = Kern_LD_blockshape[0] * Kern_LD_blockshape[1]; // N threads per block
    uint32_t Kern_LD_sharedMemorySize = (uint32_t) (sizeof(float) * ((Kern_LD_blockshape[1] * Mld) + (Kern_LD_block_surface * (2u * Mld)));
    dim3 Kern_LD_grid(Kern_LD_gridshape[0], Kern_LD_gridshape[1]);
    dim3 Kern_LD_block(Kern_LD_blockshape[0], Kern_LD_blockshape[1]);
    // Kernel 3: FAR neighbours
    uint32_t Kern_FAR_block_surface = Kern_FAR_blockshape[0] * Kern_FAR_blockshape[1]; // N threads per block
    uint32_t Kern_FAR_sharedMemorySize = (uint32_t) (sizeof(float) * ((Kern_FAR_blockshape[1] * Mld) + (Kern_FAR_block_surface * (2u * Mld)));
    dim3 Kern_FAR_grid(Kern_FAR_gridshape[0], Kern_FAR_gridshape[1]);
    dim3 Kern_FAR_block(Kern_FAR_blockshape[0], Kern_FAR_blockshape[1]);

    
    // ~~~~~~~~~  launch kernels (and wait for async memset to finish)  ~~~~~~~~~
    // kernel 1 : HD neighbours
    cudaStreamSynchronize(stream_HD); // wait for the momenta to clear
    interactions_K_HD<<<Kern_HD_grid, Kern_HD_block, Kern_HD_sharedMemorySize, stream_HD>>>(N, dvc_Pij, dvc_Xld_nester, dvc_neighsHD, furthest_neighdists_LD, Qdenom_EMA, alpha_cauchy, dvc_Qdenom_elements, dvc_momenta_attraction, dvc_momenta_repulsion);// launch the kernel 1
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {printf("Error in kernel 1: %s\n", cudaGetErrorString(err));}
    // kernel 2 : LD neighbours
    cudaStreamSynchronize(stream_LD); // wait for the momenta to clear
    interactions_K_LD<<<Kern_LD_grid, Kern_LD_block, Kern_LD_sharedMemorySize, stream_LD>>>\
    (N, dvc_Xld_nester, dvc_neighsLD, Qdenom_EMA, alpha_cauchy, &dvc_Qdenom_elements[N*Khd], dvc_momenta_repulsion, all_neighdists_LD);// launch the kernel 2
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {printf("Error in kernel 2: %s\n", cudaGetErrorString(err));}
    // kernel 3 : FAR neighbours
    cudaStreamSynchronize(stream_FAR); // wait for the momenta to clear
    interactions_far<<<Kern_FAR_grid, Kern_FAR_block, Kern_FAR_sharedMemorySize, stream_FAR>>>\
    (N, dvc_Xld_nester, Qdenom_EMA, alpha_cauchy, &dvc_Qdenom_elements[N*Khd + N*Kld], dvc_momenta_repulsion_far);// launch the kernel 3
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {printf("Error in kernel 3: %s\n", cudaGetErrorString(err));}

    // ~~~~~~~~~~~  sync all streams  ~~~~~~~~~
    cudaStreamSynchronize(stream_HD);
    cudaStreamSynchronize(stream_LD);
    cudaStreamSynchronize(stream_FAR);
   
    // TODO: ascend to godhood by using pretch CUDA instruction in assembly (Fermi architecture)
    // the prefetch instruction is used to load the data from global memory to the L2 cache
}

}

#endif