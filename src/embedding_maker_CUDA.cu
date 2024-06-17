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



__device__ void warpReduce_float_Mdim(volatile float* momenta_update_i_T, uint32_t k, uint32_t prev_len, uint32_t stride, uint32_t block_surface){
    /* prev_len = stride;
    stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
    if((k + stride < prev_len)){
        #pragma unroll
        for(uint32_t m = 0u; m < Mld; m++){
            momenta_update_i_T[m*block_surface] += momenta_update_i_T[m*block_surface + stride];
        }
    }
    if(stride == 1u){return;} */


    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
        if((k + stride < prev_len)){
            #pragma unroll
            for(uint32_t m = 0u; m < Mld; m++){
                momenta_update_i_T[m*block_surface] += momenta_update_i_T[m*block_surface + stride];
            }
        }
    }

    c est le if qui posait probleme: sans le if c est pas ok (car pas forcement power of 2)
    
    
    /* // 32 max
    prev_len = stride;
    stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
    #pragma unroll 
    for(uint32_t m = 0u; m < Mld; m++){
        momenta_update_i_T[m*block_surface] += momenta_update_i_T[m*block_surface + stride];
    }
    if(stride == 1u){return;}
    prev_len = stride;
    stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
    // 16 max
    #pragma unroll 
    for(uint32_t m = 0u; m < Mld; m++){
        momenta_update_i_T[m*block_surface] += momenta_update_i_T[m*block_surface + stride];
    }
    if(stride == 1u){return;}
    prev_len = stride;
    stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
    // 8 max
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        momenta_update_i_T[m*block_surface] += momenta_update_i_T[m*block_surface + stride];
    }
    if(stride == 1u){return;}
    prev_len = stride;
    stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
    // 4 max
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        momenta_update_i_T[m*block_surface] += momenta_update_i_T[m*block_surface + stride];
    }
    if(stride == 1u){return;}
    prev_len = stride;
    stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
    // 2 max
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        momenta_update_i_T[m*block_surface] += momenta_update_i_T[m*block_surface + stride];
    }
    if(stride == 1u){return;}
    prev_len = stride;
    stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
    // 1 max
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        momenta_update_i_T[m*block_surface] += momenta_update_i_T[m*block_surface + stride];
    } */
}


/*
grid shape : 1-d with total number of threads >= N * Khd
block shape: (Khd, Ni)
*/
__global__ void interactions_K_HD(uint32_t N, float* dvc_Pij, float* dvc_Xld_nester,\
        uint32_t* dvc_neighsHD, float* furthest_neighdists_LD, float Qdenom_EMA,\
        float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_attraction,\
        float* dvc_momenta_repulsion){
    // ~~~~~~~~~~~~~~~~~~~ get i, k and j ~~~~~~~~~~~~~~~~~~~
    uint32_t Khd           = blockDim.x; // Khd is guaranteed to be >= 32u
    uint32_t Ni            = blockDim.y;
    uint32_t block_surface = blockDim.x * blockDim.y;
    uint32_t i0            = (block_surface * blockIdx.x) / Khd; // the value of the smallest i in the block
    uint32_t i             = i0 + threadIdx.y;
    if( i >= N){return;} // out of bounds 
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
    float* momenta_aggregator_i = &smem[(i - i0) * (2u * Mld)];
    float* Xi                   = &smem[Ni*(2u*Mld) + (i - i0) * Mld];
    float* momenta_update_i_T   = &smem[Ni*(3u*Mld) + tid];  // stride for changing m: block_surface
    float* momenta_update_j_T   = &momenta_update_i_T[block_surface * Mld]; // stride for changing m: block_surface
    if(k == 0){ // fetch Xi from DRAM  and  initialise aggregator
        #pragma unroll
        for (uint32_t m = 0u; m < Mld; m++) {
            Xi[m] = dvc_Xld_nester[i * Mld + m];
            momenta_aggregator_i[m] = 0.0f; 
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

    // ~~~~~~~~~~~~~~~~~~~ attractive forces: independant gradients ~~~~~~~~~~~~~~~~~~~
    // fill individual updates to momenta for attraction
    float common_gradient_multiplier = 2.0f * (pij - (wij / Qdenom_EMA)) * powf(wij, 1.0f / alpha_cauchy);
    #pragma unroll
    for(uint32_t m = 0u; m < Mld; m++){
        float gradient = (Xi[m] - Xj[m]) * common_gradient_multiplier;
        momenta_update_i_T[m*block_surface] = -gradient; // i movement
        momenta_update_j_T[m*block_surface] =  gradient; // j movement
    }







    // ~~~~~~~~~~~~~~~~~~~ insanity check ~~~~~~~~~~~~~~~~~~~
    float ground_truth_aggregated_momenta[Mld];
    __syncthreads();
    if(k == 0u){
        for(uint32_t m = 0; m < Mld; m++){
            ground_truth_aggregated_momenta[m] = 0.0f;
            for(uint32_t k2 = 0u; k2 < Khd; k2++){
              ground_truth_aggregated_momenta[m] += momenta_update_i_T[m*block_surface + k2];
            }
        }
        
    }
    __syncthreads();



    // ~~~~~~~~~~~~~~~~~~~ attractive forces : aggregate ~~~~~~~~~~~~~~~~~~~
    /* // ---------------------  THIS WORKS  ---------------------
    uint32_t prev_len = 2u * Khd;
    uint32_t stride   = Khd;
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
        if((k + stride < prev_len)){
            #pragma unroll
            for(uint32_t m = 0u; m < Mld; m++){
                momenta_update_i_T[m*block_surface] += momenta_update_i_T[m*block_surface + stride];
            }
        }
        __syncthreads();
    }
    if(k == 0u){
        for(uint32_t m = 0u; m < Mld; m++){
            momenta_aggregator_i[m] += momenta_update_i_T[m*block_surface];
        }
    }
    //----------------------------------------------------- */
    uint32_t prev_len = 2u * Khd;
    uint32_t stride   = Khd;
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
        if((k + stride < prev_len)){
            #pragma unroll
            for(uint32_t m = 0u; m < Mld; m++){
                momenta_update_i_T[m*block_surface] += momenta_update_i_T[m*block_surface + stride];
            }
        }
        __syncthreads();
    }
    // one warp remaining
    if(k + stride < prev_len){ hmmm.. est ce que vraimnet toujours 1 seul warp? 
        warpReduce_float_Mdim(momenta_update_i_T, k, prev_len, stride, block_surface);
    } 
    __syncthreads(); 
    if(k == 0u){
        for(uint32_t m = 0u; m < Mld; m++){
            momenta_aggregator_i[m] += momenta_update_i_T[m*block_surface];
        }
    }



   if(k == 0u){
        float m0_GT = ground_truth_aggregated_momenta[0] * 60000.0f;
        float m0 = momenta_aggregator_i[0] * 60000.0f;
        float m1_GT = ground_truth_aggregated_momenta[1] * 60000.0f;
        float m1 = momenta_aggregator_i[1] * 60000.0f;
        printf(" %f == %f  ,      %f == %f ?\n", m0, m0_GT, m1, m1_GT);
    }
    return;


/*
    __syncthreads();
    if(k == 0u){
        float m0_GT = ground_truth_aggregated_momenta[0] * 60000.0f;
        float m0 = momenta_aggregator_i[0] * 60000.0f;
        float m1_GT = ground_truth_aggregated_momenta[1] * 60000.0f;
        float m1 = momenta_aggregator_i[1] * 60000.0f;
        printf(" %f == %f  ,      %f == %f ?\n", m0, m0_GT, m1, m1_GT);
    }
    return;



    /* if(k == 0u){
        for(uint32_t m = 0u; m < Mld; m++){
            Xi[m] = 0.0f; }
    }
    __syncthreads();
    for(uint32_t m = 0; m < Mld; m++){
        atomicAdd(&Xi[m], momenta_update_i_T[m*block_surface]);
    }
    __syncthreads();
    if(k == 0u){
        float v0 = Xi[0] * 60000.0f;
        float v0bis = ground_truth_aggregated_momenta[0] * 60000.0f;
        float v1 = Xi[1] * 60000.0f;
        float v1bis = ground_truth_aggregated_momenta[1] * 60000.0f;
        printf(" %f == %f  ,      %f == %f ?\n", v0, v0bis, v1, v1bis);
    } */
    return;
/* ok j avais un probleme de block surface qui etait pas bon

mtnt je sais que ces sommes sont ok: comparer avec mon accumulator
ATTENTION : FAUT SAUVEGARDER LA SOMME AVANT D ACCUMULER SINON ON COMPTE 2 FOIS LES TRUCS !!! 

--> sauvegarder dans Xj pour k=0 */





    // ~~~~~~~~~~~~~~~~~~~ attractive forces : aggregate ~~~~~~~~~~~~~~~~~~~
    
    /* 
    while(stride > 32u){
        if((k + stride < prev_len)){
            #pragma unroll
            for(uint32_t m = 0u; m < Mld; m++){
                momenta_update_i_T[m*block_surface] += momenta_update_i_T[m*block_surface + stride];
            }
        }
        __syncthreads(); 
        if(stride == 1u){break;}
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len / 2.0f);
        __syncthreads(); 
    }
    __syncthreads(); 
    // one warp remaining
    if(k + stride < prev_len){
        warpReduce_float_Mdim(momenta_update_i_T, k, prev_len, stride, block_surface);
    } */
    __syncthreads();
    

    
    // ~~~~~~~~~~~~~~~~~~~ repulsive forces ~~~~~~~~~~~~~~~~~~~
    // si pas do_repulsion: faut quand meme mettre des 0 sinon le calcul de la somme des moments est fausse
    // bool do_repulsion = ... ;
     

    // ~~~~~~~~~~~~~~~~~~~ save Qdenom_element ~~~~~~~~~~~~~~~~~~~



    /* if(i0 == 0)
        printf("Thread %d: i = %d, k = %d   (Khd %u   Mld: %u)  j: %u   eucl %f and simi LD: %f  (wij %f)   simi HD %f\n",\
         threadIdx.x, i, k, Khd, Mld, j, eucl_sq, wij / Qdenom_EMA, wij, 10000.0f * pij); */


    /* 
    // ~~~~~~~~~~~~~~~~~~~ attractive forces ~~~~~~~~~~~~~~~~~~~
    // 1: independent attraction momenta for each thread
    float common_gradient_multiplier = 2.0f * (pij - (wij / Qdenom_EMA)) * powf(wij, 1.0f / alpha_cauchy);
    #pragma unroll
    for(uint32_t m = 0; m < Mld; m++){
        float gradient = (Xi[m] - Xj[m]) * common_gradient_multiplier;
        attrac_momentum_update_i_T[m * blockDim.x] = -gradient; // i movement
        attrac_momentum_update_j_T[m * blockDim.x] =  gradient; // j movement
    }
    // 2: efficient parallel reduduction of these momenta (computing the sum of the momenta)
    __syncthreads(); 

    // temptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemp
    // temptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemp
    // temptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemp
    // (temporary) compute the true sum of the momenta, store in Xi
    if(threadIdx.x == 0 || k == 0){
        for(uint32_t m = 0; m < Mld; m++){
            Xi[m] = 0.0f;}
    }
    __syncthreads();
    for(uint32_t m = 0; m < Mld; m++){
        atomicAdd(&Xi[m], attrac_momentum_update_i_T[m * blockDim.x]);
        if(blockIdx.x == 0 && threadIdx.x == 0){
            printf("momentum x 10000 %f\n", attrac_momentum_update_i_T[m * blockDim.x] * 10000.0f);
        }
    }
    __syncthreads();
    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("\n\nThread %d: i = %d, k = %d   (Khd %u   Mld: %u)  j: %u   eucl %f and simi: %f  (wij %f) \n", threadIdx.x, i, k, Khd, Mld, j, eucl_sq, wij / Qdenom_EMA, wij); 
        for(uint32_t m = 0; m < Mld; m++){
            printf("(GT) sum of momenta 10000.0f: %f\n", Xi[m] * 10000.0f);
        }
    }
    // temptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemp
    // temptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemp
    // temptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemptemp

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1){

    }
*/
/* nouveau: il faut garantir que CHAQUE block commence avce k = 0 et termine avec k = K-1 
ie: on veut des voisinages completes par block. ca permet de faire la réduction sur des vecteurs de taille constantes

(attention: Khd n est pas toujours une pouissance de 2, donc faut un if dans la réduction) */
    

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
     uint32_t* Kern_HD_blockshape, uint32_t* Kern_HD_gridshape, uint32_t N, uint32_t Khd, float* dvc_Pij, float* dvc_Xld_nester, uint32_t* dvc_neighsHD, float* furthest_neighdists_LD, float Qdenom_EMA, float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_attraction, float* dvc_momenta_repulsion){
    
    // ~~~~~~~~~  launch Kernel 1 (HD neighbours) ~~~~~~~~~
    // verify that block size is a multiple of 32
    uint32_t Kern_HD_block_surface = Kern_HD_blockshape[0] * Kern_HD_blockshape[1]; // N threads per block
    if((Kern_HD_block_surface % 32) != 0){
        printf("\n\nError: block size must be a multiple of 32\n");
        return;
    }
    // shared memory size
    uint32_t n_floats_in_smem = (Kern_HD_blockshape[1] * (2u * Mld)) + (Kern_HD_blockshape[1] * Mld) + (Kern_HD_block_surface * (4u * Mld));
    uint32_t sharedMemorySize = (uint32_t) (sizeof(float) * n_floats_in_smem);
    dim3 Kern_HD_grid(Kern_HD_gridshape[0], Kern_HD_gridshape[1]);
    dim3 Kern_HD_block(Kern_HD_blockshape[0], Kern_HD_blockshape[1]);
    printf("kernel 1, grid shape: (%d %d %d), block shape: (%d %d %d)\n", Kern_HD_grid.x, Kern_HD_grid.y, Kern_HD_grid.z, Kern_HD_block.x, Kern_HD_block.y, Kern_HD_block.z);
    interactions_K_HD<<<Kern_HD_grid, Kern_HD_block, sharedMemorySize, stream_HD>>>(N, dvc_Pij, dvc_Xld_nester, dvc_neighsHD, furthest_neighdists_LD, Qdenom_EMA, alpha_cauchy, dvc_Qdenom_elements, dvc_momenta_attraction, dvc_momenta_repulsion);
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