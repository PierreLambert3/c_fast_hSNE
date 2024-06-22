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
        float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_repulsion_far,\
        uint32_t* random_numbers_size_NxRand, uint32_t Qdenom_offset){
    // ~~~~~~~~~~~~~~~~~~~ get i, k and j ~~~~~~~~~~~~~~~~~~~
    uint32_t Ni            = blockDim.y; // block shape: (NB_RANDOM_POINTS_FAR_REPULSION, Ni)
    uint32_t block_surface = NB_RANDOM_POINTS_FAR_REPULSION * Ni;
    uint32_t i0            = (block_surface * blockIdx.x) / NB_RANDOM_POINTS_FAR_REPULSION; // the value of the smallest i in the block
    uint32_t k             = threadIdx.x;       // block shape: (NB_RANDOM_POINTS_FAR_REPULSION, Ni)
    uint32_t i             = i0 + threadIdx.y;  // block shape: (NB_RANDOM_POINTS_FAR_REPULSION, Ni)
    if( i >= N ){return;} // out of bounds
    uint32_t tid           = threadIdx.x + threadIdx.y * NB_RANDOM_POINTS_FAR_REPULSION; // index within the block
    uint32_t j             = random_numbers_size_NxRand[i * NB_RANDOM_POINTS_FAR_REPULSION + k] % N;

    // ~~~~~~~~~~~~~~~~~~~ Initialise registers: Xj ~~~~~~~~~~~~~~~~~~~
    float Xj[Mld];
    #pragma unroll
    for (uint32_t m = 0u; m < Mld; m++) { // fetch Xj from DRAM
        Xj[m] = dvc_Xld_nester[j * Mld + m];}

    // ~~~~~~~~~~~~~~~~~~~ Initialise shared memory ~~~~~~~~~~~~~~~~~~~
    extern __shared__ float smem[];
    float* Xi                   = &smem[(i - i0) * Mld];
    float* momenta_update_i_T   = &smem[Ni*Mld + tid];  // stride for changing m: block_surface
    if(k == 0){ // fetch Xi from DRAM
        #pragma unroll
        for (uint32_t m = 0u; m < Mld; m++) {
            Xi[m] = dvc_Xld_nester[i * Mld + m];}
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
        momenta_update_i_T[m*block_surface] = -(Xi[m] - Xj[m]) * common_repulsion_gradient_multiplier; // i movement
    }
    // aggregate the individual updates
    periodic_sumReduction_on_matrix(momenta_update_i_T, Mld, block_surface, NB_RANDOM_POINTS_FAR_REPULSION, k);
    // write to global memory for point i
    if(k == 0u){
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            atomicAdd(&dvc_momenta_repulsion_far[i * Mld + m], momenta_update_i_T[m*block_surface]);}
    }

    // ~~~~~~~~~~~~~~~~~~~ update the new seed for next iteration ~~~~~~~~~~~~~~~~~~~
    random_numbers_size_NxRand[i * NB_RANDOM_POINTS_FAR_REPULSION + k] = random_uint32_t_xorshift32(&random_numbers_size_NxRand[i * NB_RANDOM_POINTS_FAR_REPULSION + k]); // save the new random number
    
    // ~~~~~~~~~~~~~~~~~~~ aggregate wij for this block ~~~~~~~~~~~~~~~~~~~
    __syncthreads();
    double* smem_double = (double*) &smem;
    smem_double[tid]    = (double) wij;
    parallel_1dsumReduction_double(smem_double, block_surface, tid);
    if(tid == 0u){
        dvc_Qdenom_elements[Qdenom_offset + blockIdx.x] = smem_double[0u];}
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
        float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_repulsion, float* temporary_furthest_neighdists, uint32_t Qdenom_offset){
    // ~~~~~~~~~~~~~~~~~~~ get i, k and j ~~~~~~~~~~~~~~~~~~~
    uint32_t Ni            = blockDim.y; // block shape: (Kld, Ni)
    uint32_t block_surface = Kld * Ni;
    uint32_t i0            = (block_surface * blockIdx.x) / Kld; // the value of the smallest i in the block
    uint32_t k             = threadIdx.x;       // block shape: (Kld, Ni)
    uint32_t i             = i0 + threadIdx.y;  // block shape: (Kld, Ni)
    if( i >= N ){return;} // out of bounds (no guarantee that N is a multiple of Kld)
    uint32_t j             = dvc_neighsLD[i * Kld + k]; 
    uint32_t tid           = threadIdx.x + threadIdx.y * Kld; // index within the block

    // ~~~~~~~~~~~~~~~~~~~ Initialise registers: Xj ~~~~~~~~~~~~~~~~~~~
    float Xj[Mld];
    #pragma unroll
    for (uint32_t m = 0u; m < Mld; m++) { // fetch Xj from DRAM
        Xj[m] = dvc_Xld_nester[j * Mld + m];}

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
        momenta_update_i_T[m*block_surface] = -gradient; // i movement
        momenta_update_j_T[m*block_surface] =  gradient; // j movement
    }
    // aggregate the individual updates
    periodic_sumReduction_on_matrix(momenta_update_i_T, Mld, block_surface, Kld, k);
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

    // ~~~~~~~~~~~~~~~~~~~ find the fursthest neighbour distance in LD and save it ~~~~~~~~~~~~~~~~~~~
    momenta_update_i_T[0] = eucl_sq; // start by writing eucl to shared memory
    periodic_maxReduction_on_matrix(momenta_update_i_T, 1u, block_surface, Kld, k); // find the furthest neighbour distance in LD (parallel reduction)
    if(k == 0u){ // save to global memory for point i
        temporary_furthest_neighdists[i] = momenta_update_i_T[0];}
    
    // ~~~~~~~~~~~~~~~~~~~ aggregate wij for this block ~~~~~~~~~~~~~~~~~~~
    __syncthreads();
    double* smem_double = (double*) &smem;
    smem_double[tid]    = (double) wij;
    parallel_1dsumReduction_double(smem_double, block_surface, tid);
    if(tid == 0u){
        dvc_Qdenom_elements[Qdenom_offset + blockIdx.x] = smem_double[0u];}
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
        float* dvc_momenta_repulsion, uint32_t Qdenom_offset){
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

    // ~~~~~~~~~~~~~~~~~~~ attractive forces ~~~~~~~~~~~~~~~~~~~
    // individual updates to momenta for attraction
    float powerthing = 2.0f * powf(wij, __frcp_rn(alpha_cauchy));
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
        // float common_repulsion_gradient_multiplier  = -(wij / Qdenom_EMA) * powerthing;
        float common_repulsion_gradient_multiplier  = -(wij * __frcp_rn(Qdenom_EMA)) * powerthing;
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

    // ~~~~~~~~~~~~~~~~~~~ aggregate wij for this block ~~~~~~~~~~~~~~~~~~~
    __syncthreads();
    double* smem_double = (double*) &smem;
    smem_double[tid]    = (double) wij;
    parallel_1dsumReduction_double(smem_double, block_surface, tid); 
    if(tid == 0u){
        dvc_Qdenom_elements[Qdenom_offset + blockIdx.x] = smem_double[0u];}
    return;
}

void fill_raw_momenta_launch_cuda(cudaStream_t stream_HD, cudaStream_t stream_LD, cudaStream_t stream_FAR,\
     uint32_t* Kern_HD_blockshape, uint32_t* Kern_HD_gridshape,uint32_t* Kern_LD_blockshape, uint32_t* Kern_LD_gridshape,uint32_t* Kern_FAR_blockshape, uint32_t* Kern_FAR_gridshape,\
      uint32_t N, uint32_t Khd, float* dvc_Pij,\
      float* dvc_Xld_nester, uint32_t* dvc_neighsHD, uint32_t* dvc_neighsLD, float* furthest_neighdists_LD, float Qdenom_EMA,\
       float alpha_cauchy, double* dvc_Qdenom_elements, uint32_t Qdenom_N_elements,\
        float* dvc_momenta_attraction, float* dvc_momenta_repulsion, float* dvc_momenta_repulsion_far, float* temporary_furthest_neighdists,\
         uint32_t* random_numbers_size_NxRand){
    
    
    // ~~~~~~~~~  clear momenta and dvc_Qdenom_elements (async)  ~~~~~~~~~
    cudaMemsetAsync(dvc_Qdenom_elements, 0, Qdenom_N_elements * sizeof(double), stream_HD);
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
    uint32_t Kern_LD_sharedMemorySize = (uint32_t) (sizeof(float) * ((Kern_LD_blockshape[1] * Mld) + (Kern_LD_block_surface * (2u * Mld))));
    dim3 Kern_LD_grid(Kern_LD_gridshape[0], Kern_LD_gridshape[1]);
    dim3 Kern_LD_block(Kern_LD_blockshape[0], Kern_LD_blockshape[1]);
    // Kernel 3: FAR neighbours
    uint32_t Kern_FAR_block_surface = Kern_FAR_blockshape[0] * Kern_FAR_blockshape[1]; // N threads per block
    uint32_t Kern_FAR_sharedMemorySize = (uint32_t) (sizeof(float) * ((Kern_FAR_blockshape[1] * Mld) + (Kern_FAR_block_surface * (2u * Mld))));
    dim3 Kern_FAR_grid(Kern_FAR_gridshape[0], Kern_FAR_gridshape[1]);
    dim3 Kern_FAR_block(Kern_FAR_blockshape[0], Kern_FAR_blockshape[1]);
    // kernel4 : sum wij samples
    // uint32_t
    
    // ~~~~~~~~~  launch kernels (and wait for async memset to finish)  ~~~~~~~~~
    uint32_t block_offset = 0u;
    // kernel 1 : HD neighbours
    cudaStreamSynchronize(stream_HD); // wait for the momenta to clear
    interactions_K_HD<<<Kern_HD_grid, Kern_HD_block, Kern_HD_sharedMemorySize, stream_HD>>>(N, dvc_Pij, dvc_Xld_nester, dvc_neighsHD, furthest_neighdists_LD, Qdenom_EMA, alpha_cauchy, dvc_Qdenom_elements, dvc_momenta_attraction, dvc_momenta_repulsion, block_offset);// launch the kernel 1
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {printf("Error in kernel 1: %s\n", cudaGetErrorString(err1));}
    // kernel 2 : LD neighbours
    block_offset += Kern_HD_grid.x * Kern_HD_grid.y;
    cudaStreamSynchronize(stream_LD); // wait for the momenta to clear
    interactions_K_LD<<<Kern_LD_grid, Kern_LD_block, Kern_LD_sharedMemorySize, stream_LD>>>(N, dvc_Xld_nester, dvc_neighsLD, Qdenom_EMA, alpha_cauchy, &dvc_Qdenom_elements[N*Khd], dvc_momenta_repulsion, temporary_furthest_neighdists, block_offset);// launch the kernel 2
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {printf("Error in kernel 2: %s\n", cudaGetErrorString(err2));}
    // kernel 3 : FAR neighbours
    block_offset += Kern_LD_grid.x * Kern_LD_grid.y;
    cudaStreamSynchronize(stream_FAR); // wait for the momenta to clear
    interactions_far<<<Kern_FAR_grid, Kern_FAR_block, Kern_FAR_sharedMemorySize, stream_FAR>>>(N, dvc_Xld_nester, Qdenom_EMA, alpha_cauchy, &dvc_Qdenom_elements[N*Khd + N*Kld], dvc_momenta_repulsion_far, random_numbers_size_NxRand, block_offset);// launch the kernel 3
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {printf("Error in kernel 3: %s\n", cudaGetErrorString(err3));}
    block_offset += Kern_FAR_grid.x * Kern_FAR_grid.y;


    // ~~~~~~~~~~~  memcpy for furthest_neighdists_LD  ~~~~~~~~~
    // wait for the 1st kernel to finish (because it uses furthest_neighdists_LD)
    cudaStreamSynchronize(stream_HD);
    // wait for the 2nd kernel to finish (because it writes to temporary_furthest_neighdists)
    cudaStreamSynchronize(stream_LD);
    // do a memcpy from temporary_furthest_neighdists to furthest_neighdists_LD 
    cudaMemcpyAsync(furthest_neighdists_LD, temporary_furthest_neighdists, N * sizeof(float), cudaMemcpyDeviceToDevice, stream_LD);
    

    // ~~~~~~~~~~~  sync streams  ~~~~~~~~~
    cudaStreamSynchronize(stream_FAR);
    cudaStreamSynchronize(stream_LD);

    // ~~~~~~~~~~~  all streams are synced: now compute the Qdenom  ~~~~~~~~~
    //  here : sum the block_offset first elements of dvc_Qdenom_elements
    // faire plutot une copie sur CPU et la somme dans une boucle
    printf("number of elements: %d\n", Qdenom_N_elements);
    printf("here todo   ^  \n");
   
    // TODO: ascend to godhood by using pretch CUDA instruction in assembly (Fermi architecture). The prefetch instruction is used to load the data from global memory to the L2 cache
}

}

#endif