#ifndef NEIGHHD_DISCOVERER_H
#define NEIGHHD_DISCOVERER_H

#include <pthread.h>
#include "system.h"
#include "probabilities.h"

typedef struct {
    pthread_t thread;
    uint32_t N_reserved_subthreads;
    uint32_t N_subthreads_now;
    uint32_t N_subthreads_target;
    pthread_t* subthreads;
    pthread_cond_t* subthreads_cond;
    pthread_rwlock_t* subthreads_rwlock;
    uint32_t subthreads_chunck_size; // the number of elements to be processed by each subthread
    bool     isRunning;
    uint32_t rand_state;
    uint32_t passes_since_reset;
    float    p_wakeup;
    uint32_t N;
    int*     test_variable;
} NeighHDDiscoverer;

void new_NeighHDDiscoverer(NeighHDDiscoverer* thing, uint32_t _N_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads);
void  destroy_NeighHDDiscoverer(NeighHDDiscoverer* thing);
void* routine_NeighHDDiscoverer(void* arg);
void start_thread_NeighHDDiscoverer(NeighHDDiscoverer* thing);

#endif // NEIGHHD_DISCOVERER_H