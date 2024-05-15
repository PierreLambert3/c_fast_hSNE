#ifndef NEIGHLD_DISCOVERER_H
#define NEIGHLD_DISCOVERER_H

#include <pthread.h>
#include "system.h"
#include "probabilities.h"
#include <stdbool.h>
#include <stdint.h>
#include "constants_global.h"

typedef struct {
    bool     stop_this_thread;
    uint32_t N;
    uint32_t rand_state;
    int      L;
    int      R;
    float**    Xld;
    float**    Xhd;
    uint32_t   Mld;
    uint32_t   Khd;
    uint32_t   Kld;
    uint32_t** neighsLD;
    uint32_t** neighsHD;
    float**    Q;
    float      estimated_Qdenom;
    pthread_mutex_t* mutexes_sizeN;
    pthread_mutex_t* thread_mutex;
    bool* thread_waiting_for_task;
} SubthreadData;

typedef struct {
    // worker and subthread management
    pthread_t thread;
    bool     isRunning;
    uint32_t rand_state;
    uint32_t passes_since_reset;
    float    p_wakeup;
    uint32_t N_reserved_subthreads;
    uint32_t N_subthreads_target;
    uint32_t subthreads_chunck_size; // the number of elements to be processed by each subthread
    pthread_t*       subthreads;
    pthread_mutex_t* subthreads_mutexes;
    pthread_mutex_t  mutex_N_subthreads_target;
    bool* threads_waiting_for_task;
    
    // Algorithm and subthread data: for determining LD neighbours, Q, and Qdenom
    SubthreadData* subthread_data;
    uint32_t N;
    float**    Xld;
    float**    Xhd;
    uint32_t   Mld;
    uint32_t   Khd;
    uint32_t   Kld;
    uint32_t** neighsLD;
    uint32_t** neighsHD;
    float**    Q;
    float*     Qdenom;
    pthread_mutex_t* mutex_Qdenom;  
    pthread_mutex_t* mutexes_sizeN;
} NeighLDDiscoverer;



NeighLDDiscoverer* new_NeighLDDiscoverer(uint32_t _N_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads,\
    pthread_mutex_t* _mutex_Qdenom_, pthread_mutex_t* mutexes_sizeN, float** _Xld_, float** _Xhd_, uint32_t _Mld_, uint32_t _Khd_,\
    uint32_t _Kld_, uint32_t** _neighsLD_, uint32_t** _neighsHD_, float** _Q_, float* _Qdenom_);
void  destroy_NeighLDDiscoverer(NeighLDDiscoverer* thing);
void* routine_NeighLDDiscoverer(void* arg);
void  start_thread_NeighLDDiscoverer(NeighLDDiscoverer* thing);

void* subroutine_NeighLDDiscoverer(void* arg);

#endif // NEIGHHD_DISCOVERER_H