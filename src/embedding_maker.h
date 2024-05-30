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
    pthread_mutex_t* mutexes_sizeN;
    float*           hparam_LDkernel_alpha; // shared with gui    
    pthread_mutex_t* mutex_hparam_LDkernel_alpha;
    float**          Xld_cpu;      // will be on CPU as a 2d-array
    uint32_t**       neighsLD_cpu;
    uint32_t**       neighsHD_cpu;
    float*           furthest_neighdists_LD_cpu;
    float**          P_cpu; 
    pthread_mutex_t* mutex_P; // for when writing CPU<->GPU
    // things on GPU
    float*          Xld_base_cuda;     // will be on GPU as a 1d-array, use Xnesterov[N] to access the 1d data
    float*          Xld_nesterov_cuda; // will be on GPU as a 1d-array, use Xnesterov[N] to access the 1d data
    float*          momenta_attraction_cuda;   // will be on GPU as a 1d-array, use momenta_attraction[N] to access the 1d data
    float*          momenta_repulsion_far_cuda;  // this will leak to neighbours 
    float*          momenta_repulsion_cuda; 
    uint32_t*       neighsLD_cuda;
    uint32_t*       neighsHD_cuda;
    float*          furthest_neighdists_LD_cuda;
    float*          P_cuda; 
    float           Qdenom_cuda;
} EmbeddingMaker_GPU;

typedef struct {
    pthread_t thread;
    EmbeddingMaker_CPU* maker_cpu;
    EmbeddingMaker_GPU* maker_gpu;
} EmbeddingMaker;



void  new_EmbeddingMaker(EmbeddingMaker* thing, uint32_t N, uint32_t Mld, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t Kld, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P);
void  new_EmbeddingMaker_CPU(EmbeddingMaker_CPU* thing, uint32_t N, uint32_t Mld, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t Kld, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P);
void  new_EmbeddingMaker_GPU(EmbeddingMaker_GPU* thing, uint32_t N, uint32_t Mld, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t Kld, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P);
void  destroy_EmbeddingMaker(EmbeddingMaker* thing);
void* routine_EmbeddingMaker_CPU(void* arg);
void* routine_EmbeddingMaker_GPU(void* arg);
void  start_thread_EmbeddingMaker(EmbeddingMaker* thing);

#endif // EMBEDDING_MAKER_H