#ifndef CUDA_THINGS_H
#define CUDA_THINGS_H
#include <stdint.h>
extern "C"{

__global__ void interactions_K_HD(int N, int Khd, float* dvc_Pij, float* dvc_Xld_nester, uint32_t* dvc_neighsHD, float* furthest_neighdists_LD, float Qdenom_EMA, float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_attraction, float* dvc_momenta_repulsion){

}

void fill_raw_momenta_launch_cuda(int K_HD_block_size, int K_HD_n_blocks, int N, int Khd, float* dvc_Pij, float* dvc_Xld_nester, uint32_t* dvc_neighsHD, float* furthest_neighdists_LD, float Qdenom_EMA, float alpha_cauchy, double* dvc_Qdenom_elements, float* dvc_momenta_attraction, float* dvc_momenta_repulsion){
    // ~~~~~~~~~  Kernel 1: HD neighbours  ~~~~~~~~~
    interactions_K_HD<<<K_HD_block_size, K_HD_n_blocks>>>(N, Khd, dvc_Pij, dvc_Xld_nester, dvc_neighsHD, furthest_neighdists_LD, Qdenom_EMA, alpha_cauchy, dvc_Qdenom_elements, dvc_momenta_attraction, dvc_momenta_repulsion);
    cudaDeviceSynchronize();
}

}

#endif