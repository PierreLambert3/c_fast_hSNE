#ifndef EMBEDDING_MAKER_H
#define EMBEDDING_MAKER_H

#include <pthread.h>
#include "system.h"
#include "vector.h"
#include "probabilities.h"
#include "constants_global.h"

// one struct for CPU-based embedding maker
typedef struct {
    bool     running;
    pthread_mutex_t* mutex_Qdenom;
} EmbeddingMaker_CPU;

// one struct for GPU-based embedding maker
typedef struct {
    pthread_mutex_t* mutex_thread;
    uint32_t         rand_state;
    bool             is_running;
    uint32_t         work_type;
    uint32_t         N;
    uint32_t         Khd;
    pthread_mutex_t* mutexes_sizeN;
    float*           hparam_LDkernel_alpha; // shared with gui    
    pthread_mutex_t* mutex_hparam_LDkernel_alpha;
    float*           hparam_repulsion_multiplier; // shared with gui    
    pthread_mutex_t* mutex_hparam_repulsion_multiplier;
    float**          Xld_cpu;      // will be on CPU as a 2d-array
    uint32_t**       neighsLD_cpu;
    uint32_t**       neighsHD_cpu;
    float*           furthest_neighdists_LD_cpu;
    float            Qdenom_EMA;
    pthread_mutex_t* mutex_P; // for when writing CPU<->GPU

    // safe GPU / CPU communication: neighsHD and Psym
    GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsHD;
    GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsLD;
    GPU_CPU_float_buffer*  GPU_CPU_comms_P;
    
    // GPU kernel, number of floats un shared memory per thread
    uint32_t* Kern_HD_gridshape;  // 3d array
    uint32_t* Kern_HD_blockshape; // 3d array
    uint32_t* Kern_LD_gridshape;  // 3d array
    uint32_t* Kern_LD_blockshape; // 3d array
    uint32_t* Kern_FAR_gridshape;  // 3d array
    uint32_t* Kern_FAR_blockshape; // 3d array
    uint32_t* Kern_Qdenomsum_blockshape; // 3d array
    uint32_t* Kern_Qdenomsum_gridshape;

    // streams
    cudaStream_t stream_K_HD;
    cudaStream_t stream_K_LD;
    cudaStream_t stream_rand;
    cudaStream_t stream_Qdenomsum;

    // things on GPU
    float*          Xld_base_cuda;     // will be on GPU as a 1d-array, use Xnesterov[N] to access the 1d data
    float*          Xld_nesterov_cuda; // will be on GPU as a 1d-array, use Xnesterov[N] to access the 1d data
    float*          momenta_attraction_cuda;   // will be on GPU as a 1d-array, use momenta_attraction[N] to access the 1d data
    float*          momenta_repulsion_far_cuda;  // this will leak to neighbours 
    float*          momenta_repulsion_cuda;
    uint32_t*       neighsLD_cuda;
    uint32_t*       neighsHD_cuda;
    float*          furthest_neighdists_LD_cuda;
    float*          temporary_furthest_neighdists_LD_cuda;
    uint32_t*       random_numbers_size_NxRand_cuda;
    uint32_t        N_elements_of_Qdenom;
    double*         elements_of_Qdenom_cuda; // will be on GPU as a 1d-array
    float*         sum_Qdenom_elements_cuda;
    float*          P_cuda; 
} EmbeddingMaker_GPU;

typedef struct {
    pthread_t thread;
    EmbeddingMaker_CPU* maker_cpu;
    EmbeddingMaker_GPU* maker_gpu;
} EmbeddingMaker;


void  new_EmbeddingMaker(EmbeddingMaker* thing, uint32_t N, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P,\
    GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsHD, GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsLD, GPU_CPU_float_buffer* GPU_CPU_comms_P);
void  new_EmbeddingMaker_CPU(EmbeddingMaker_CPU* thing, uint32_t N, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P);
void  new_EmbeddingMaker_GPU(EmbeddingMaker_GPU* thing, uint32_t N, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P,\
    GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsHD, GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsLD, GPU_CPU_float_buffer* GPU_CPU_comms_P);
void  destroy_EmbeddingMaker(EmbeddingMaker* thing);
void* routine_EmbeddingMaker_CPU(void* arg);
void* routine_EmbeddingMaker_GPU(void* arg);
void  start_thread_EmbeddingMaker(EmbeddingMaker* thing);

void fill_raw_momenta_GPU(EmbeddingMaker_GPU* thing);
void momenta_leak_GPU(EmbeddingMaker_GPU* thing);
void apply_momenta_and_decay_GPU(EmbeddingMaker_GPU* thing);

void fill_raw_momenta_launch_cuda(cudaStream_t, cudaStream_t, cudaStream_t, cudaStream_t,\
 uint32_t*, uint32_t*,uint32_t*, uint32_t*,uint32_t*, uint32_t*,uint32_t*, uint32_t*,\
  uint32_t, uint32_t, float*,\
   float*, uint32_t*, uint32_t*, float*, float,\
    float, double*, float*, uint32_t,\
     float*, float*, float*, float*,\
      uint32_t*);
#endif // EMBEDDING_MAKER_H