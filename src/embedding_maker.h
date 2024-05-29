#ifndef EMBEDDING_MAKER_H
#define EMBEDDING_MAKER_H

#include <pthread.h>
#include "system.h"
#include "vector.h"
#include "probabilities.h"
#include <stdbool.h>
#include <stdint.h>
#include "constants_global.h"

// one struct for CPU-based embedding maker
typedef struct {
    bool     running;
} EmbeddingMaker_CPU;

// one struct for GPU-based embedding maker
typedef struct {
    pthread_mutex_t* mutex_thread;
    uint32_t         rand_state;
    bool             running;
    uint32_t         work_type;
    uint32_t         N;
    pthread_mutex_t* mutexes_sizeN;
    float            Qdenom;
    pthread_mutex_t* mutex_Qdenom;
    float*           hparam_LDkernel_alpha; // shared with gui    
    pthread_mutex_t* mutex_hparam_LDkernel_alpha;
} EmbeddingMaker_GPU;

typedef struct {
    pthread_t thread;
    EmbeddingMaker_CPU* maker_cpu;
    EmbeddingMaker_GPU* maker_gpu;
} EmbeddingMaker;



void  new_EmbeddingMaker(EmbeddingMaker* thing, uint32_t N, uint32_t* thread_rand_seed);
void  new_EmbeddingMaker_CPU(EmbeddingMaker_CPU* thing, uint32_t N, uint32_t* thread_rand_seed);
void  new_EmbeddingMaker_GPU(EmbeddingMaker_GPU* thing, uint32_t N, uint32_t* thread_rand_seed);
void  destroy_EmbeddingMaker(EmbeddingMaker* thing);
void* routine_EmbeddingMaker_CPU(void* arg);
void* routine_EmbeddingMaker_GPU(void* arg);
void  start_thread_EmbeddingMaker(EmbeddingMaker* thing);

#endif // EMBEDDING_MAKER_H