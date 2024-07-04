#ifndef CUDA_THINGS_H
#define CUDA_THINGS_H
#include <stdint.h>
#include <stdio.h>
#include "constants_global.h"

extern "C"{

__device__ __forceinline__ uint32_t random_uint32_t_xorshift32(uint32_t* rand_state){
    *rand_state ^= *rand_state << 13u;
    *rand_state ^= *rand_state >> 17u;
    *rand_state ^= *rand_state << 5u;
    return *rand_state;
}

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

/***
 *                           _ _      _                      _            _   _             
 *                          | | |    | |                    | |          | | (_)            
 *     _ __   __ _ _ __ __ _| | | ___| |        _ __ ___  __| |_   _  ___| |_ _  ___  _ __  
 *    | '_ \ / _` | '__/ _` | | |/ _ \ |       | '__/ _ \/ _` | | | |/ __| __| |/ _ \| '_ \ 
 *    | |_) | (_| | | | (_| | | |  __/ |       | | |  __/ (_| | |_| | (__| |_| | (_) | | | |
 *    | .__/ \__,_|_|  \__,_|_|_|\___|_|       |_|  \___|\__,_|\__,_|\___|\__|_|\___/|_| |_|
 *    | |                                                                                   
 *    |_|                                                                                   
 */
__device__ void warpReduce_periodic_maxReduction_on_matrix(volatile float* matrix, uint32_t e, uint32_t prev_len, uint32_t stride, uint32_t Ncol){
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if((e + stride < prev_len)){
            #pragma unroll
            for(uint32_t m = 0u; m < Mld; m++){
                float maxval   = fmaxf(matrix[m*Ncol + stride], matrix[m*Ncol]);
                matrix[m*Ncol] = maxval; // CANNOT DO TERNARY OPERATION BECAUSE RACE CONDITION!!!!
            }
        }
    }
}

__device__ __forceinline__ void periodic_maxReduction_on_matrix(float* matrix, uint32_t Nrows, uint32_t Ncol, uint32_t period, uint32_t e){
    __syncthreads(); // just in case it wasn't done after writing
    uint32_t prev_len = 2u * period;
    uint32_t stride   = period;
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if((e + stride < prev_len)){
            #pragma unroll
            for(uint32_t m = 0u; m < Mld; m++){
                float maxval   = fmaxf(matrix[m*Ncol + stride], matrix[m*Ncol]);
                matrix[m*Ncol] = maxval; // CANNOT DO TERNARY OPERATION BECAUSE RACE CONDITION!!!!
            }
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore (volatile float* matrix prevents reordering)
    if(e + stride < prev_len){ 
        warpReduce_periodic_maxReduction_on_matrix(matrix, e, prev_len, stride, Ncol);}
    __syncthreads(); // this one is not necessary
}

__device__ void warpReduce_periodic_sumReduction_on_matrix(volatile float* matrix, uint32_t e, uint32_t prev_len, uint32_t stride, uint32_t Ncol){
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(e + stride < prev_len){
            #pragma unroll
            for(uint32_t m = 0u; m < Mld; m++){
                float to_add    = matrix[m*Ncol + stride];
                matrix[m*Ncol] += to_add; // don't use += on things on the right that can be modified by other threads
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
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(e + stride < prev_len){
            #pragma unroll
            for(uint32_t m = 0u; m < Mld; m++){
                float to_add    = matrix[m*Ncol + stride];
                matrix[m*Ncol] += to_add; // don't use += on things on the right that can be modified by other threads
            }
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore (volatile float* matrix prevents reordering)
    if(e + stride < prev_len){ 
        warpReduce_periodic_sumReduction_on_matrix(matrix, e, prev_len, stride, Ncol);}
    __syncthreads(); // this one is not necessary
}

// on 1d vectors 
__device__ void warpReduce_1dsumReduction_double(volatile double* vector, uint32_t i, uint32_t prev_len, uint32_t stride){
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            double to_add  = vector[i + stride];
            vector[i]     += to_add;
        }
    }
}

__device__ void parallel_1dsumReduction_double(double* vector, uint32_t n, uint32_t i){
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            double to_add  = vector[i + stride];
            vector[i]     += to_add;
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce_1dsumReduction_double(vector, i, prev_len, stride);}
    __syncthreads();
}


/***
 *                    _                _                        _     
 *                   | |              | |                      | |    
 *      ___ _   _  __| | __ _         | | _____ _ __ _ __   ___| |___ 
 *     / __| | | |/ _` |/ _` |        | |/ / _ \ '__| '_ \ / _ \ / __|
 *    | (__| |_| | (_| | (_| |        |   <  __/ |  | | | |  __/ \__ \
 *     \___|\__,_|\__,_|\__,_|        |_|\_\___|_|  |_| |_|\___|_|___/
 *                                                                    
 *                                                                    
 */

/*periodic_sumReduction_on_matrix(momenta_update_i_T, Mld, block_surface, NB_RANDOM_POINTS_FAR_REPULSION, k);        |
        |[----- Mld -----]|
        |[----- Mld -----]|
        |[----- Mld -----]|
        |[----- Mld -----]|
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
__global__ void interactions_far(uint32_t N, float* cu_Xld_nesterov, float Qdenom_EMA,\
        float alpha_cauchy, double* cu_elements_of_Qdenom, float* cu_nudge_FAR,\
        uint32_t* cu_random_numbers_size_NxRand){
    // ~~~~~~~~~~~~~~~~~~~ get i, k and j ~~~~~~~~~~~~~~~~~~~
    uint32_t this_Ni = blockDim.y; // block shape: (NB_RANDOM_POINTS_FAR_REPULSION, Ni)
    // if last block: Ni is not guaranteed to be the same as usual
    if(blockIdx.x == gridDim.x - 1u){
        this_Ni = N - (blockIdx.x * blockDim.y);}
    uint32_t this_block_surface = NB_RANDOM_POINTS_FAR_REPULSION * this_Ni;
    uint32_t i0            = blockDim.y * blockIdx.x; // the value of the smallest i in the block
    uint32_t k             = threadIdx.x;       // block shape: (NB_RANDOM_POINTS_FAR_REPULSION, Ni)
    uint32_t i             = i0 + threadIdx.y;  // block shape: (NB_RANDOM_POINTS_FAR_REPULSION, Ni)
    if( i >= N ){return;} // out of bounds
    uint32_t tid           = threadIdx.x + threadIdx.y * NB_RANDOM_POINTS_FAR_REPULSION; // index within the block
    uint32_t j             = cu_random_numbers_size_NxRand[i * NB_RANDOM_POINTS_FAR_REPULSION + k] % N;


    // ~~~~~~~~~~~~~~~~~~~ Initialise registers: Xj ~~~~~~~~~~~~~~~~~~~
    float Xj[Mld];
    #pragma unroll
    for (uint32_t m = 0u; m < Mld; m++) { // fetch Xj from DRAM
        Xj[m] = cu_Xld_nesterov[j * Mld + m];}

    // ~~~~~~~~~~~~~~~~~~~ Initialise shared memory ~~~~~~~~~~~~~~~~~~~
    extern __shared__ float smem3[];
    float* Xi                   = &smem3[(i - i0) * Mld];
    float* momenta_update_i_T   = &smem3[this_Ni*Mld + tid];  // stride for changing m: block_surface
    if(k == 0){ // fetch Xi from DRAM
        #pragma unroll
        for (uint32_t m = 0u; m < Mld; m++) {
            Xi[m] = cu_Xld_nesterov[i * Mld + m];}
    }
    __syncthreads();

    // ~~~~~~~~~~~~~~~~~~~ now for some computations:  wij (offset already done) ~~~~~~~~~~~~~~~~~~~
    float eucl_sq = cuda_euclidean_sq(Xi, Xj); // compute squared euclidean distance 
    float wij     = cuda_cauchy_kernel(eucl_sq, alpha_cauchy);  // similarity in LD (qij = wij / Qdenom_EMA)

    // ~~~~~~~~~~~~~~~~~~~ repulsive forces far ~~~~~~~~~~~~~~~~~~~
    // DO NOT APPLY MOMENTA ON j, ONLY i (because else we don't have a guaranteed balance on the forces)
    float common_repulsion_gradient_multiplier  = -(wij * __frcp_rn(Qdenom_EMA)) * (2.0f * powf(wij, __frcp_rn(alpha_cauchy)));
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        momenta_update_i_T[m*this_block_surface] = -(Xi[m] - Xj[m]) * common_repulsion_gradient_multiplier; // i movement
    }
    // aggregate the individual updates
    periodic_sumReduction_on_matrix(momenta_update_i_T, Mld, this_block_surface, NB_RANDOM_POINTS_FAR_REPULSION, k);
    // write to global memory for point i
    if(k == 0u){
        float scaling_factor = 0.5f * (float) (N-1u) / (float) NB_RANDOM_POINTS_FAR_REPULSION;
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            float scaled_value = momenta_update_i_T[m*this_block_surface] * scaling_factor;
            atomicAdd(&cu_nudge_FAR[i * Mld + m], scaled_value);}
    }

    // ~~~~~~~~~~~~~~~~~~~ update the new seed for next iteration ~~~~~~~~~~~~~~~~~~~
    cu_random_numbers_size_NxRand[i * NB_RANDOM_POINTS_FAR_REPULSION + k] = random_uint32_t_xorshift32(&cu_random_numbers_size_NxRand[i * NB_RANDOM_POINTS_FAR_REPULSION + k]); // save the new random number
    
    // ~~~~~~~~~~~~~~~~~~~ aggregate wij for this block ~~~~~~~~~~~~~~~~~~~
    __syncthreads();
    double* smem_double3 = (double*) &smem3;
    smem_double3[tid]    = (double) wij;
    parallel_1dsumReduction_double(smem_double3, this_block_surface, tid);
    if(tid == 0u){
        cu_elements_of_Qdenom[blockIdx.x] = smem_double3[0u];}
    return;
}


/* visual representation of shared memory:
        |[----- Mld -----]|    each row contains the Xi momentum source
        |        .        |                            
        |        .        |
        |[----- Mld -----]|
        --------------------------------------
        | M | M | M | M | ... | M | M | M | M |    
        | l | l | l | l | ... | l | l | l | l |     width: Ni x Kld
        | d | d | d | d | ... | d | d | d | d |        
        --------------------------------------

        block shape: (Kld, Ni)
*/
__global__ void leak_kernel(uint32_t N, uint32_t* cu_neighsLD, float* momenta_src, float* momenta_rcv){
    // LEAK_ALPHA : 1 is big leak, 0 is small leak
    uint32_t this_Ni = blockDim.y; // block shape: (Kld, Ni)
    if(blockIdx.x == gridDim.x - 1u){
        this_Ni = N - (blockIdx.x * blockDim.y);}
    uint32_t this_block_surface = Kld * this_Ni;
    uint32_t i0            = blockDim.y * blockIdx.x; // the value of the smallest i in the block
    uint32_t k             = threadIdx.x;       // block shape: (Kld, Ni)
    uint32_t i             = i0 + threadIdx.y;  // block shape: (Kld, Ni)
    if( i >= N ){return;} // out of bounds (no guarantee that N is a multiple of Kld)
    __syncthreads();
    uint32_t j             = cu_neighsLD[i * Kld + k]; 
    uint32_t tid           = threadIdx.x + threadIdx.y * Kld; // index within the block

    // ~~~~~~~~~~~~~~~~~~~ Initialise registers: source from j ~~~~~~~~~~~~~~~~~~~
    float src_j[Mld];
    #pragma unroll
    for (uint32_t m = 0u; m < Mld; m++) { // fetch source from DRAM
        src_j[m] = momenta_src[j*Mld + m];}

    // ~~~~~~~~~~~~~~~~~~~ Initialise shared memory ~~~~~~~~~~~~~~~~~~~
    extern __shared__ float smem5[];
    float* src_i  = &smem5[(i - i0) * Mld];
    float* rcv_ik = &smem5[this_Ni*Mld + tid];  // stride for changing m: block_surface
    if(k == 0){ // fetch source momentum from DRAM
        #pragma unroll
        for (uint32_t m = 0u; m < Mld; m++) {
            src_i[m] = momenta_src[i*Mld + m];}
    }
    __syncthreads();

    // ~~~~~~~~~~~~~~~~~~~ compte leak & save to memory ~~~~~~~~~~~~~~~~~~~
    // raw leak: 0.5 * (src_i + src_j)
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        float mean_v = (0.5 * (src_i[m] + src_j[m])) * __frcp_rn((float)Kld);
        rcv_ik[m*this_block_surface] = mean_v;
    }
    // write to global memory for point j (every thread does this: contention is low because of the random access pattern)
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        float external_contribution = LEAK_ALPHA * rcv_ik[m*this_block_surface];
        atomicAdd(&momenta_rcv[j*Mld + m], external_contribution);
    }
    // parallel sum reduction: prepare aggregation for point i
    periodic_sumReduction_on_matrix(rcv_ik, Mld, this_block_surface, Kld, k); // syncthreads included in 1st instruction
    // for i: do the self contribution here (not in j)
    if(k == 0u){
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            float self_contribution     = (1.0f - LEAK_ALPHA) * src_i[m];
            float external_contribution = LEAK_ALPHA * rcv_ik[m*this_block_surface];
            atomicAdd(&momenta_rcv[i*Mld + m], self_contribution + external_contribution);
        }
    }
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
__global__ void interactions_K_LD(uint32_t N, float* cu_Xld_nesterov, uint32_t* cu_neighsLD, float Qdenom_EMA,\
        float alpha_cauchy, double* cu_elements_of_Qdenom, float* cu_nudge_repuls_HDLD, float* cu_temporary_furthest_neighdists_LD){
    // ~~~~~~~~~~~~~~~~~~~ get i, k and j ~~~~~~~~~~~~~~~~~~~
    uint32_t this_Ni = blockDim.y; // block shape: (Kld, Ni)
    // if last block: Ni is not guaranteed to be the same as usual
    if(blockIdx.x == gridDim.x - 1u){
        this_Ni = N - (blockIdx.x * blockDim.y);}
    uint32_t this_block_surface = Kld * this_Ni;
    uint32_t i0            = blockDim.y * blockIdx.x; // the value of the smallest i in the block
    uint32_t k             = threadIdx.x;       // block shape: (Kld, Ni)
    uint32_t i             = i0 + threadIdx.y;  // block shape: (Kld, Ni)
    if( i >= N ){return;} // out of bounds (no guarantee that N is a multiple of Kld)
    __syncthreads();
    uint32_t j             = cu_neighsLD[i * Kld + k]; 
    uint32_t tid           = threadIdx.x + threadIdx.y * Kld; // index within the block

    // ~~~~~~~~~~~~~~~~~~~ Initialise registers: Xj ~~~~~~~~~~~~~~~~~~~
    float Xj[Mld];
    #pragma unroll
    for (uint32_t m = 0u; m < Mld; m++) { // fetch Xj from DRAM
        Xj[m] = cu_Xld_nesterov[j * Mld + m];}

    // ~~~~~~~~~~~~~~~~~~~ Initialise shared memory ~~~~~~~~~~~~~~~~~~~
    extern __shared__ float smem2[];
    float* Xi                   = &smem2[(i - i0) * Mld];
    float* momenta_update_i_T   = &smem2[this_Ni*Mld + tid];  // stride for changing m: block_surface
    float* momenta_update_j_T   = &smem2[this_Ni*Mld + this_block_surface*Mld + tid]; // stride for changing m: block_surface
    if(k == 0){ // fetch Xi from DRAM
        #pragma unroll
        for (uint32_t m = 0u; m < Mld; m++) {
            Xi[m] = cu_Xld_nesterov[i * Mld + m];}
    }
    __syncthreads();

    // ~~~~~~~~~~~~~~~~~~~ now for some computations:  wij (offset already done) ~~~~~~~~~~~~~~~~~~~
    // compute squared euclidean distance 
    float eucl_sq = cuda_euclidean_sq(Xi, Xj);
    // similarity in LD (qij = wij / Qdenom_EMA)
    float wij     = cuda_cauchy_kernel(eucl_sq, alpha_cauchy); 
    // ~~~~~~~~~~~~~~~~~~~ repulsive forces ~~~~~~~~~~~~~~~~~~~
    // individual updates to momenta for repulsion
    float common_repulsion_gradient_multiplier  = -(wij * __frcp_rn(Qdenom_EMA)) * (2.0f * powf(wij, __frcp_rn(alpha_cauchy)));

    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        float gradient = (Xi[m] - Xj[m]) * common_repulsion_gradient_multiplier;
        momenta_update_i_T[m*this_block_surface] = -gradient; // i movement
        momenta_update_j_T[m*this_block_surface] =  gradient; // j movement
    }
    // aggregate the individual updates
    periodic_sumReduction_on_matrix(momenta_update_i_T, Mld, this_block_surface, Kld, k);
    // write to global memory for point i
    if(k == 0u){
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            atomicAdd(&cu_nudge_repuls_HDLD[i * Mld + m], momenta_update_i_T[m*this_block_surface]);}
    }
    // write individual updates to j repulsion momenta
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        atomicAdd(&cu_nudge_repuls_HDLD[j * Mld + m], momenta_update_j_T[m*this_block_surface]);}

    // ~~~~~~~~~~~~~~~~~~~ find the fursthest neighbour distance in LD and save it ~~~~~~~~~~~~~~~~~~~
    momenta_update_i_T[0] = eucl_sq; // start by writing eucl to shared memory
    periodic_maxReduction_on_matrix(momenta_update_i_T, 1u, this_block_surface, Kld, k); // find the furthest neighbour distance in LD (parallel reduction)
    if(k == 0u){ // save to global memory for point i
        cu_temporary_furthest_neighdists_LD[i] = momenta_update_i_T[0];}
    
    // ~~~~~~~~~~~~~~~~~~~ aggregate wij for this block ~~~~~~~~~~~~~~~~~~~
    __syncthreads();
    double* smem_double2 = (double*) &smem2;
    smem_double2[tid]    = (double) wij;
    parallel_1dsumReduction_double(smem_double2, this_block_surface, tid);
    if(tid == 0u){
        cu_elements_of_Qdenom[blockIdx.x] = smem_double2[0u]; }
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
__global__ void interactions_K_HD(uint32_t N, float* dvc_Pij, float* cu_Xld_nesterov,\
        uint32_t* dvc_neighsHD, float* cu_furthest_neighdists_LD, float Qdenom_EMA,\
        float alpha_cauchy, double* cu_elements_of_Qdenom, float* cu_nudge_attrac_HD,\
        float* cu_nudge_repuls_HDLD){
    // ~~~~~~~~~~~~~~~~~~~ get i, k and j ~~~~~~~~~~~~~~~~~~~
    uint32_t Khd           = blockDim.x; // block shape: (Khd, Ni);  Khd is guaranteed to be >= 32u
    uint32_t this_Ni       = blockDim.y; // block shape: (Khd, Ni)
    // if last block: Ni is not guaranteed to be the same as usual
    if(blockIdx.x == gridDim.x - 1u){
        this_Ni = N - (blockIdx.x * blockDim.y);}
    uint32_t this_block_surface = Khd * this_Ni; 
    uint32_t i0            = blockDim.y * blockIdx.x; // the value of the smallest i in the block
    uint32_t k             = threadIdx.x;
    uint32_t i             = i0 + threadIdx.y;
    if( i >= N ){return;} // out of bounds (no guarantee that N is a multiple of Ni)
    uint32_t j             = dvc_neighsHD[i * Khd + k]; 
    uint32_t tid           = threadIdx.x + threadIdx.y * Khd; // index within the block

    // ~~~~~~~~~~~~~~~~~~~ Initialise registers: Xj and furthest LDneighdists for i and j ~~~~~~~~~~~~~~~~~~~
    float furthest_LDneighdist_j = __ldg(&cu_furthest_neighdists_LD[j]);
    float Xj[Mld];
    #pragma unroll
    for (uint32_t m = 0u; m < Mld; m++) { // fetch Xj from DRAM
        Xj[m] = cu_Xld_nesterov[j * Mld + m];}
    float furthest_LDneighdist_i = __ldg(&cu_furthest_neighdists_LD[i]);

    // ~~~~~~~~~~~~~~~~~~~ Initialise shared memory ~~~~~~~~~~~~~~~~~~~
    extern __shared__ float smem1[];
    float* Xi                   = &smem1[(i - i0) * Mld];
    float* momenta_update_i_T   = &smem1[this_Ni*Mld + tid];  // stride for changing m: block_surface
    float* momenta_update_j_T   = &smem1[this_Ni*Mld + this_block_surface*Mld + tid]; // stride for changing m: block_surface
    if(k == 0){ // fetch Xi from DRAM
        #pragma unroll
        for (uint32_t m = 0u; m < Mld; m++) {
            Xi[m] = cu_Xld_nesterov[i * Mld + m];}
    }
    __syncthreads();

    // ~~~~~~~~~~~~~~~~~~~ now for some computations: prepare pij and wij ~~~~~~~~~~~~~~~~~~~
    // compute squared euclidean distance 
    float eucl_sq = cuda_euclidean_sq(Xi, Xj);
    // similarity in HD
    float pij     = __ldg(&dvc_Pij[i * Khd + k]);
    // similarity in LD (qij = wij / Qdenom_EMA)
    float wij     = cuda_cauchy_kernel(eucl_sq, alpha_cauchy); 

    // ~~~~~~~~~~~~~~~~~~~ attractive forces ~~~~~~~~~~~~~~~~~~~
    // individual updates to momenta for attraction
    float powerthing = 2.0f * powf(wij, __frcp_rn(alpha_cauchy));
    float common_attraction_gradient_multiplier =  pij * powerthing;
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        float gradient = (Xi[m] - Xj[m]) * common_attraction_gradient_multiplier;
        momenta_update_i_T[m*this_block_surface] = -gradient; // i movement
        momenta_update_j_T[m*this_block_surface] =  gradient; // j movement
    }
    // aggregate the individual updates
    periodic_sumReduction_on_matrix(momenta_update_i_T, Mld, this_block_surface, Khd, k);
    // write aggregated to global memory for point i
    if(k == 0u){
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            atomicAdd(&cu_nudge_attrac_HD[i * Mld + m], momenta_update_i_T[m*this_block_surface]);}
    }
    // write individual updates to j attraction momenta (contention should be low because of the random access pattern)
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        atomicAdd(&cu_nudge_attrac_HD[j * Mld + m], momenta_update_j_T[m*this_block_surface]);}

    // ~~~~~~~~~~~~~~~~~~~ repulsive forces ~~~~~~~~~~~~~~~~~~~
    // individual updates to momenta for repulsion
    bool do_repulsion = eucl_sq > furthest_LDneighdist_i && eucl_sq > furthest_LDneighdist_j; // do repulsion if not LD neighbours. 
    if(do_repulsion){ // the  conditional is annoying because there is no structure in the decision to do repulsion or not: x2 time taken
        // float common_repulsion_gradient_multiplier  = -(wij / Qdenom_EMA) * powerthing;
        float common_repulsion_gradient_multiplier  = -(wij * __frcp_rn(Qdenom_EMA)) * powerthing;
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            float gradient = (Xi[m] - Xj[m]) * common_repulsion_gradient_multiplier;
            momenta_update_i_T[m*this_block_surface] = -gradient; // i movement
            momenta_update_j_T[m*this_block_surface] =  gradient; // j movement
        }
    }
    else{ // important to set to zero, else the aggregation will be wrong
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            momenta_update_i_T[m*this_block_surface] = 0.0f;
            momenta_update_j_T[m*this_block_surface] = 0.0f;
        }
    }

    // aggregate the individual updates
    periodic_sumReduction_on_matrix(momenta_update_i_T, Mld, this_block_surface, Khd, k);
    // write to global memory for point i
    if(k == 0u){
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            atomicAdd(&cu_nudge_repuls_HDLD[i * Mld + m], momenta_update_i_T[m*this_block_surface]);}
    }
    // write individual updates to j repulsion momenta
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        atomicAdd(&cu_nudge_repuls_HDLD[j * Mld + m], momenta_update_j_T[m*this_block_surface]);}

    // ~~~~~~~~~~~~~~~~~~~ aggregate wij for this block ~~~~~~~~~~~~~~~~~~~
    __syncthreads();
    double* smem_double1 = (double*) &smem1;
    smem_double1[tid]    = (double) wij;
    parallel_1dsumReduction_double(smem_double1, this_block_surface, tid); 
    if(tid == 0u){
        cu_elements_of_Qdenom[blockIdx.x] = smem_double1[0u];}
    return;
}

__global__ void final_sum_Qdenom_elements(double* cu_elements_of_Qdenom, float* sum_Qdenom_elements, uint32_t n_elements_total){
    extern __shared__ double smem4[];
    uint32_t tid           = threadIdx.x;
    uint32_t global_offset = blockIdx.x * blockDim.x;
    if(global_offset + tid < n_elements_total){
        smem4[tid] = cu_elements_of_Qdenom[global_offset + tid];}
    else{
        smem4[tid] = 0.0;}
    __syncthreads();
    uint32_t n_elements = blockDim.x;
    if(blockIdx.x == gridDim.x - 1u){
        n_elements = n_elements_total - (blockIdx.x * blockDim.x); }
    parallel_1dsumReduction_double(smem4, n_elements, tid);
    if(tid == 0u){
        atomicAdd(sum_Qdenom_elements, (float) smem4[0u]);}
    return;
}


// grid 1d    ;     block 2d : (Mld, Ni)
__global__ void apply_momenta_and_decay(uint32_t N, float* cu_Xld_base, float* cu_Xld_nesterov,\
        float* cu_nudge_attrac_HD, float* cu_nudge_repuls_HDLD, float* cu_nudge_FAR,\
        float* cu_momenta_attrac, float* cu_momenta_repuls_near, float* cu_momentum_far,\
        float repulsion_multiplier, float lr){

    // ~~~~~~~~~ determine which point (i) and ld variable (m) we're talking about ~~~~~~~~~
    uint32_t i = blockIdx.x * blockDim.y + threadIdx.y;
    if(i >= N){return;} // out of bounds
    uint32_t m = threadIdx.x;

    // ~~~~~~~~~  update momenta (nudge & decay) ~~~~~~~~~
    float new_attraction_momentum     = MOMENTUM_ALPHA*cu_momenta_attrac[i * Mld + m]      + lr*cu_nudge_attrac_HD[i * Mld + m];
    float new_repulsion_momentum_near = MOMENTUM_ALPHA*cu_momenta_repuls_near[i * Mld + m] + lr*cu_nudge_repuls_HDLD[i * Mld + m];
    float new_repulsion_momentum_far  = MOMENTUM_ALPHA*cu_momentum_far[i * Mld + m]        + lr*cu_nudge_FAR[i * Mld + m];
    cu_momenta_attrac[i * Mld + m]      = new_attraction_momentum;
    cu_momenta_repuls_near[i * Mld + m] = new_repulsion_momentum_near;
    cu_momentum_far[i * Mld + m]        = new_repulsion_momentum_far;
    
    // ~~~~~~~~~ compute & save in a register the resultant momentum ~~~~~~~~~
    float attraction = cu_momenta_attrac[i * Mld + m];
    float repulsion  = cu_momenta_repuls_near[i * Mld + m] + cu_momentum_far[i * Mld + m];
    float movement   = (attraction + (repulsion_multiplier * repulsion));

    // ~~~~~~~~~ apply momenta to Xld & generated nesterov parameters ~~~~~~~~~
    float new_x_parameter = cu_Xld_base[i * Mld + m] + movement;
    float new_x_nesterov  = cu_Xld_base[i * Mld + m] + 1.9f * movement;
    cu_Xld_base[i * Mld + m]     = new_x_parameter;
    cu_Xld_nesterov[i * Mld + m] = new_x_nesterov;
    return;
}

// grid 1d    ;     block 2d : (Mld, Ni)
__global__ void rescale_embedding(uint32_t N, float* cu_Xld_base, float* cu_Xld_nesterov, float* cu_momenta_attrac, float* cu_momenta_repuls_near, float* cu_momentum_far0, float* cu_momentum_far1,\
            float* cu_nudge_attrac_HD, float* cu_nudge_repuls_HDLD, float* cu_nudge_FAR){
    // ~~~~~~~~~ determine which point (i) and ld variable (m) we're talking about ~~~~~~~~~
    uint32_t i = blockIdx.x * blockDim.y + threadIdx.y;
    if(i >= N){return;} // out of bounds
    uint32_t m = threadIdx.x;

    // ~~~~~~~~~  rescale the embedding ~~~~~~~~~
    float new_x_parameter = cu_Xld_base[i * Mld + m] / 10.0f;
    cu_Xld_base[i * Mld + m]       = new_x_parameter;
    cu_Xld_nesterov[i * Mld + m]   = new_x_parameter;
    cu_momenta_attrac[i * Mld + m]      = 0.0f;
    cu_momenta_repuls_near[i * Mld + m] = 0.0f;
    cu_momentum_far0[i * Mld + m]       = 0.0f;
    cu_momentum_far1[i * Mld + m]       = 0.0f;
    cu_nudge_attrac_HD[i * Mld + m]     = 0.0f;
    cu_nudge_repuls_HDLD[i * Mld + m]   = 0.0f;
    cu_nudge_FAR[i * Mld + m]           = 0.0f;
    return;
}

// grid 1d    ;     block 2d : (Mld, Ni)
// no need for efficiency here, called rarely
__global__ void recompute_LD_neighdists(uint32_t N, float* cu_Xld_nesterov, uint32_t* cu_neighsLD, float* cu_furthest_neighdists_LD){
    // ~~~~~~~~~ determine which point (i) and ld variable (m) we're talking about ~~~~~~~~~
    uint32_t i = blockIdx.x * blockDim.y + threadIdx.y;
    if(i >= N){return;} // out of bounds
    uint32_t m = threadIdx.x;
    
    if(m == 0u){
        float furthest_neighdist = 0.0f;
        float Xi[Mld];
        #pragma unroll
        for (uint32_t m = 0u; m < Mld; m++) { // fetch Xi from DRAM
            Xi[m] = cu_Xld_nesterov[i * Mld + m];}
        for(uint32_t k = 0u; k < Mld; k++){
            float Xj[Mld];
            #pragma unroll
            for (uint32_t m = 0u; m < Mld; m++) { // fetch Xj from DRAM
                Xj[m] = cu_Xld_nesterov[cu_neighsLD[i * Mld + k] * Mld + m];}
            float eucl_sq = cuda_euclidean_sq(Xi, Xj);
            if(eucl_sq > furthest_neighdist){
                furthest_neighdist = eucl_sq;}
        }
        cu_furthest_neighdists_LD[i] = furthest_neighdist;
    } // reset the furthest neighbour distance

    return;
}



void cuda_launch___recompute_LD_neighdists(uint32_t* block_shape, uint32_t* grid_shape,\
            uint32_t N, float* cu_Xld_nesterov, uint32_t* cu_neighsLD, float* cu_furthest_neighdists_LD){
    // sync the whole device
    cudaDeviceSynchronize();

    // ~~~~~~~~~  prepare kernel calls ~~~~~~~~~
    dim3 grid(grid_shape[0], grid_shape[1]);
    dim3 block(block_shape[0], block_shape[1]);

    // ~~~~~~~~~ launch kernel ~~~~~~~~~
    recompute_LD_neighdists<<<grid, block>>>(N, cu_Xld_nesterov, cu_neighsLD, cu_furthest_neighdists_LD);
    cudaDeviceSynchronize();
}

void cuda_launch___rescale_embedding(uint32_t* block_shape, uint32_t* grid_shape,uint32_t N, float* cu_Xld_base, float* cu_Xld_nesterov, float* cu_momenta_attrac, float* cu_momenta_repuls_near, float* cu_momentum_far0, float* cu_momentum_far1, float* cu_nudge_attrac_HD, float* cu_nudge_repuls_HDLD, float* cu_nudge_FAR){
    // sync the whole device
    cudaDeviceSynchronize();

    // ~~~~~~~~~  prepare kernel calls ~~~~~~~~~
    dim3 grid(grid_shape[0], grid_shape[1]);
    dim3 block(block_shape[0], block_shape[1]);

    // ~~~~~~~~~ launch kernel ~~~~~~~~~
    rescale_embedding<<<grid, block>>>(N, cu_Xld_base, cu_Xld_nesterov, cu_momenta_attrac, cu_momenta_repuls_near, cu_momentum_far0, cu_momentum_far1, cu_nudge_attrac_HD, cu_nudge_repuls_HDLD, cu_nudge_FAR);
    cudaDeviceSynchronize();
}

void cuda_launch___apply_momenta_and_decay(cudaStream_t stream_params, uint32_t* block_shape, uint32_t* grid_shape,\
        uint32_t N, float* cu_Xld_base, float* cu_Xld_nesterov,\
        float* cu_nudge_attrac_HD, float* cu_nudge_repuls_HDLD, float* cu_nudge_FAR,\
        float* cu_momenta_attrac, float* cu_momenta_repuls_near, float* cu_momentum_far,\
        float repulsion_multiplier, float lr){

    // ~~~~~~~~~  prepare kernel calls ~~~~~~~~~
    dim3 grid(grid_shape[0], grid_shape[1]);
    dim3 block(block_shape[0], block_shape[1]);

    // ~~~~~~~~~ launch kernel ~~~~~~~~~
    cudaStreamSynchronize(stream_params); // redundant but I'd say it's a safe practice
    apply_momenta_and_decay<<<grid, block, 0, stream_params>>>(N, cu_Xld_base, cu_Xld_nesterov,\
        cu_nudge_attrac_HD, cu_nudge_repuls_HDLD, cu_nudge_FAR,\
        cu_momenta_attrac, cu_momenta_repuls_near, cu_momentum_far,\
        repulsion_multiplier, lr);
    cudaStreamSynchronize(stream_params);
    return;
}

void cuda_launch___fill_nudges_and_leak(cudaStream_t stream_nudge_HD, cudaStream_t stream_nudge_LD, cudaStream_t stream_nudge_FAR, cudaStream_t stream_Qdenomsum, cudaStream_t stream_leak,\
        uint32_t* Kern_HD_blockshape, uint32_t* Kern_HD_gridshape,uint32_t* Kern_LD_blockshape, uint32_t* Kern_LD_gridshape,uint32_t* Kern_FAR_blockshape, uint32_t* Kern_FAR_gridshape, uint32_t* Kern_Qdenomsum_blockshape, uint32_t* Kern_Qdenomsum_gridshape,  uint32_t* Kern_leak_blockshape, uint32_t* Kern_leak_gridshape,\
        uint32_t N, uint32_t Khd, float* cu_P,\
        float* cu_Xld_nesterov, uint32_t* cu_neighsHD, uint32_t* cu_neighsLD, float* cu_furthest_neighdists_LD, float Qdenom_EMA,\
        float alpha_cauchy, double* cu_elements_of_Qdenom, float* cu_sum_Qdenom_elements, float* cpu_sum_Qdenom_elements, uint32_t Qdenom_N_elements,\
        float* cu_nudge_attrac_HD, float* cu_nudge_repuls_HDLD, float* cu_nudge_FAR, float* cu_temporary_furthest_neighdists_LD,\
        uint32_t* cu_random_numbers_size_NxRand,\
        float* momentum_src_leak, float* momentum_rcv_leak){
    
    // ~~~~~~~~~  clear nudges and cu_elements_of_Qdenom (async)  ~~~~~~~~~
    cudaMemsetAsync(cu_nudge_attrac_HD, 0, N * Mld * sizeof(float), stream_nudge_HD);
    cudaMemsetAsync(cu_nudge_repuls_HDLD, 0, N * Mld * sizeof(float), stream_nudge_LD);
    cudaMemsetAsync(cu_nudge_FAR, 0, N * Mld * sizeof(float), stream_nudge_FAR);
    cudaMemsetAsync(cu_elements_of_Qdenom, 0, Qdenom_N_elements * sizeof(double), stream_Qdenomsum);
    cudaMemsetAsync(cu_sum_Qdenom_elements, 0, sizeof(float), stream_Qdenomsum);

    // ~~~~~~~~~  clear the receiving momentum ~~~~~~~~~
    cudaMemsetAsync(momentum_rcv_leak, 0, N * Mld * sizeof(float), stream_leak);

    // ~~~~~~~~~  prepare kernel calls ~~~~~~~~~
    // Kernel 1: HD neighbours
    uint32_t Kern_HD_block_surface = Kern_HD_blockshape[0] * Kern_HD_blockshape[1]; // N threads per block
    if((Kern_HD_block_surface % 32) != 0){printf("\n\nError: block size should be a multiple of 32\n");return;}
    uint32_t Kern_HD_sharedMemorySize = (uint32_t) (sizeof(float) * ((Kern_HD_blockshape[1] * Mld) + (Kern_HD_block_surface * (2u * Mld))));
    dim3 Kern_HD_grid(Kern_HD_gridshape[0], Kern_HD_gridshape[1]);
    dim3 Kern_HD_block(Kern_HD_blockshape[0], Kern_HD_blockshape[1]);
    // Kernel 2: LD neighbours
    uint32_t Kern_LD_block_surface = Kern_LD_blockshape[0] * Kern_LD_blockshape[1]; // N threads per block
    uint32_t Kern_LD_sharedMemorySize = (uint32_t) (sizeof(float) * ((Kern_LD_blockshape[1] * Mld) + (Kern_LD_block_surface * (2u * Mld))));
    dim3 Kern_LD_grid(Kern_LD_gridshape[0], Kern_LD_gridshape[1]);
    dim3 Kern_LD_block(Kern_LD_blockshape[0], Kern_LD_blockshape[1]);
    // Kernel 3: FAR neighbours
    uint32_t Kern_FAR_block_surface = Kern_FAR_blockshape[0] * Kern_FAR_blockshape[1]; // N threads per block
    uint32_t Kern_FAR_sharedMemorySize = (uint32_t) (sizeof(float) * ((Kern_FAR_blockshape[1] * Mld) + (Kern_FAR_block_surface * (2u * Mld))));
    dim3 Kern_FAR_grid(Kern_FAR_gridshape[0], Kern_FAR_gridshape[1]);
    dim3 Kern_FAR_block(Kern_FAR_blockshape[0], Kern_FAR_blockshape[1]);
    // kernel4 : sum wij samples
    uint32_t Kern_Qdenomsum_block_surface = Kern_Qdenomsum_blockshape[0] * Kern_Qdenomsum_blockshape[1]; 
    uint32_t Kern_Qdenomsum_sharedMemorySize = (uint32_t) (sizeof(double) * Kern_Qdenomsum_block_surface);
    dim3 Kern_Qdenomsum_grid(Kern_Qdenomsum_gridshape[0], Kern_Qdenomsum_gridshape[1]);
    dim3 Kern_Qdenomsum_block(Kern_Qdenomsum_blockshape[0], Kern_Qdenomsum_blockshape[1]);
    // kernel 5: leak   block: (Kld, Ni)
    uint32_t leak_block_surface = Kern_leak_blockshape[0] * Kern_leak_blockshape[1]; // N threads per block
    uint32_t leak_sharedMemorySize = (uint32_t) (sizeof(float) * ((Kern_leak_blockshape[1] * Mld) + (Kld * Kern_leak_blockshape[1] * Mld)));
    dim3 Kern_leak_grid(Kern_leak_gridshape[0], Kern_leak_gridshape[1]);
    dim3 Kern_leak_block(Kern_leak_blockshape[0], Kern_leak_blockshape[1]);


    // ~~~~~~~~~  launch kernels (and wait for async memset to finish)  ~~~~~~~~~
    // wait for the Qdenom elements to be reset
    cudaStreamSynchronize(stream_Qdenomsum);

    // kernel 1 : HD neighbours
    cudaStreamSynchronize(stream_nudge_HD); // wait for the momenta to clear
    interactions_K_HD<<<Kern_HD_grid, Kern_HD_block, Kern_HD_sharedMemorySize, stream_nudge_HD>>>(N, cu_P, cu_Xld_nesterov, cu_neighsHD, cu_furthest_neighdists_LD, Qdenom_EMA, alpha_cauchy, cu_elements_of_Qdenom, cu_nudge_attrac_HD, cu_nudge_repuls_HDLD);// launch the kernel 1
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {printf("Error in kernel 1: %s\n", cudaGetErrorString(err1));}

    // kernel 2 : LD neighbours
    cudaStreamSynchronize(stream_nudge_LD); // wait for the momenta to clear
    uint32_t Qdenom_offset = Kern_HD_grid.x * Kern_HD_grid.y;
    interactions_K_LD<<<Kern_LD_grid, Kern_LD_block, Kern_LD_sharedMemorySize, stream_nudge_LD>>>(N, cu_Xld_nesterov, cu_neighsLD, Qdenom_EMA, alpha_cauchy, &cu_elements_of_Qdenom[Qdenom_offset], cu_nudge_repuls_HDLD, cu_temporary_furthest_neighdists_LD);// launch the kernel 2
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {printf("Error in kernel 2: %s\n", cudaGetErrorString(err2));}

    // kernel 3 : FAR neighbours
    cudaStreamSynchronize(stream_nudge_FAR); // wait for the momenta to clear
    Qdenom_offset += Kern_LD_grid.x * Kern_LD_grid.y;
    interactions_far<<<Kern_FAR_grid, Kern_FAR_block, Kern_FAR_sharedMemorySize, stream_nudge_FAR>>>(N, cu_Xld_nesterov, Qdenom_EMA, alpha_cauchy, &cu_elements_of_Qdenom[Qdenom_offset], cu_nudge_FAR, cu_random_numbers_size_NxRand);// launch the kernel 3
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {printf("Error in kernel 3: %s\n", cudaGetErrorString(err3));}

    // kernel 5 : leak the momenta
    cudaStreamSynchronize(stream_leak);
    /* leak_kernel<<<Kern_leak_grid, Kern_leak_block, leak_sharedMemorySize, stream_leak>>>(N, cu_neighsLD, momentum_src_leak, momentum_rcv_leak);
    cudaError_t err5 = cudaGetLastError();
    if (err5 != cudaSuccess) {printf("Error in kernel 5: %s\n", cudaGetErrorString(err5));} */

    // ~~~~~~~~~~~  update cu_furthest_neighdists_LD  ~~~~~~~~~
    cudaStreamSynchronize(stream_nudge_HD); // wait for the 1st kernel to finish (because it USES      cu_furthest_neighdists_LD)
    cudaStreamSynchronize(stream_nudge_LD); // wait for the 2nd kernel to finish (because it WRITES TO cu_temporary_furthest_neighdists_LD)
    cudaMemcpyAsync(cu_furthest_neighdists_LD, cu_temporary_furthest_neighdists_LD, N * sizeof(float), cudaMemcpyDeviceToDevice, stream_nudge_LD);

    // ~~~~~~~~~~~  now compute the Qdenom  ~~~~~~~~~
    cudaStreamSynchronize(stream_nudge_FAR); // sync remaining stream
    final_sum_Qdenom_elements<<<Kern_Qdenomsum_grid, Kern_Qdenomsum_block, Kern_Qdenomsum_sharedMemorySize, stream_Qdenomsum>>>(cu_elements_of_Qdenom, cu_sum_Qdenom_elements, Qdenom_N_elements);
    cudaMemcpyAsync(cpu_sum_Qdenom_elements, cu_sum_Qdenom_elements, sizeof(float), cudaMemcpyDeviceToHost, stream_Qdenomsum);

    // ~~~~~~~~~~~  sync last streams: we want everybody to be done here  ~~~~~~~~~
    cudaStreamSynchronize(stream_Qdenomsum);
    cudaStreamSynchronize(stream_nudge_LD);
    cudaStreamSynchronize(stream_leak);

    // TODO: ascend to godhood by using pretch CUDA instruction in assembly (Fermi architecture). The prefetch instruction is used to load the data from global memory to the L2 cache
}

}

#endif
