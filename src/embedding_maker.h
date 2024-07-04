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
    uint32_t* Kern_leak_gridshape;
    uint32_t* Kern_leak_blockshape;
    uint32_t* Kern_parameter_updates_gridshape;
    uint32_t* Kern_parameter_updates_blockshape;
    

    // streams
    // computing nudges (& Qdenomsum)
    cudaStream_t stream_nudge_HD;
    cudaStream_t stream_nudge_LD;
    cudaStream_t stream_nudge_FAR;
    cudaStream_t stream_Qdenomsum;
    // leak momentum
    cudaStream_t stream_leak;
    // parameter update
    cudaStream_t stream_parameter_updates;

    // rescale embedding (get things unstuck)
    pthread_mutex_t* mutex_rescale_embedding;
    bool rescale_embedding;

    bool leak_phase; // will be toggled at each iteration

    // things on GPU  (all as 1d arrays)
    float*          cu_nudge_attrac_HD;
    float*          cu_nudge_repuls_HDLD;
    float*          cu_nudge_FAR;
    
    float*          cu_momenta_attrac;
    float*          cu_momenta_repuls_near;
    float*          cu_momenta_repuls_far___0;
    float*          cu_momenta_repuls_far___1;

    float*          cu_Xld_base;    
    float*          cu_Xld_nesterov;

    uint32_t*       cu_neighsLD;
    uint32_t*       cu_neighsHD;
    float*          cu_furthest_neighdists_LD;
    float*          cu_temporary_furthest_neighdists_LD;
    uint32_t*       cu_random_numbers_size_NxRand;
    float*          cu_P; 

    uint32_t        N_elements_of_Qdenom;
    double*         cu_elements_of_Qdenom; // will be on GPU as a 1d-array
    float*          cu_sum_Qdenom_elements;
    float* HD_pct_new_neighs;
} EmbeddingMaker_GPU;

typedef struct {
    pthread_t thread;
    EmbeddingMaker_CPU* maker_cpu;
    EmbeddingMaker_GPU* maker_gpu;
    float* HD_pct_new_neighs;
} EmbeddingMaker;


void  new_EmbeddingMaker(EmbeddingMaker* thing, uint32_t N, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P,\
    GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsHD, GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsLD, GPU_CPU_float_buffer* GPU_CPU_comms_P,\
    float* pct_new_neighs_HD);
void  new_EmbeddingMaker_CPU(EmbeddingMaker_CPU* thing, uint32_t N, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P);
void  new_EmbeddingMaker_GPU(EmbeddingMaker_GPU* thing, uint32_t N, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P,\
    GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsHD, GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsLD, GPU_CPU_float_buffer* GPU_CPU_comms_P,\
    float* pct_new_neighs_HD);
void  destroy_EmbeddingMaker(EmbeddingMaker* thing);
void* routine_EmbeddingMaker_CPU(void* arg);
void* routine_EmbeddingMaker_GPU(void* arg);
void  start_thread_EmbeddingMaker(EmbeddingMaker* thing);

void fill_nudges_GPU(EmbeddingMaker_GPU* thing);
void apply_momenta_and_decay_GPU(EmbeddingMaker_GPU* thing);

void cuda_launch___fill_nudges_and_leak(cudaStream_t, cudaStream_t, cudaStream_t, cudaStream_t, cudaStream_t,\
        uint32_t*, uint32_t*,uint32_t*, uint32_t*,uint32_t*, uint32_t*,uint32_t*, uint32_t*,uint32_t*, uint32_t*,\
        uint32_t, uint32_t, float*,\
        float*, uint32_t*, uint32_t*, float*, float,\
        float, double*, float*, float*, uint32_t,\
        float*, float*, float*, float*,\
        uint32_t*,\
        float*, float*);


void cuda_launch___apply_momenta_and_decay(cudaStream_t, uint32_t*, uint32_t*,\
        uint32_t, float*, float*,\
        float*, float*, float*,\
        float*, float*, float*,\
        float, float);

void cuda_launch___rescale_embedding(uint32_t*, uint32_t*,\
    uint32_t, float*, float*,\
    float*, float*, float*, float*, float*, float*, float*);

void cuda_launch___recompute_LD_neighdists(uint32_t*, uint32_t*,\
    uint32_t, float*, uint32_t*, float*);

#endif // EMBEDDING_MAKER_H