#ifndef NEIGHHD_DISCOVERER_H
#define NEIGHHD_DISCOVERER_H

#include <pthread.h>
#include "system.h"
#include "vector.h"
#include "probabilities.h"
#include <stdbool.h>
#include <stdint.h>
#include "constants_global.h"

typedef struct {
      
} SubthreadHD_data;

typedef struct {
    // worker and subthread management
    pthread_t thread;
    bool     isRunning;
    uint32_t rand_state;
    uint32_t N_reserved_subthreads;
    uint32_t N_subthreads_target;
    uint32_t subthreads_chunck_size; // the number of elements to be processed by each subthread
    pthread_t*       subthreads;
    pthread_mutex_t* subthreads_mutexes;
    pthread_mutex_t  mutex_N_subthreads_target;
    bool* threads_waiting_for_task;

    // Algorithm and subthread data: for determining LD neighbours, Q, and Qdenom
    SubthreadHD_data* subthreadHD_data;
    uint32_t   N;
    float**    Xhd;
    uint32_t   Khd;
    uint32_t   Kld;
    uint32_t** neighsHD;
    uint32_t** neighsLD;
    float*     furthest_neighdists_HD;

    float**    Pasy;
    float*     Pasy_sumJ_Pij;
    bool*      flag_neigh_update;
    float**    Psym_GT;
    float*     radii;
    float      pct_new_neighs;
    pthread_mutex_t* mutexes_sizeN;

} NeighHDDiscoverer;

void new_NeighHDDiscoverer(NeighHDDiscoverer* thing, uint32_t _N_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads,\
    pthread_mutex_t* mutexes_sizeN, float** _Xhd_, uint32_t _Khd_, uint32_t _Kld_, uint32_t** _neighsHD_, uint32_t** _neighsLD_,\
    float* furthest_neighdists_HD, float** _Psym_GT_);
void  destroy_NeighHDDiscoverer(NeighHDDiscoverer* thing);
void* routine_NeighHDDiscoverer(void* arg);
void start_thread_NeighHDDiscoverer(NeighHDDiscoverer* thing);

#endif // NEIGHHD_DISCOVERER_H