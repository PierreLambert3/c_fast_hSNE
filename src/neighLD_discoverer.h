#ifndef NEIGHLD_DISCOVERER_H
#define NEIGHLD_DISCOVERER_H

#include <pthread.h>
#include "system.h"
#include "vector.h"
#include "probabilities.h"
#include <stdbool.h>
#include <stdint.h>
#include "constants_global.h"

typedef struct {
    bool     stop_this_thread;
    uint32_t N;
    uint32_t rand_state;
    uint32_t      L;
    uint32_t      R;
    uint32_t    N_new_neighs;
    float**    Xld;
    uint32_t   Mld;
    uint32_t   Khd;
    uint32_t   Kld;
    uint32_t** neighsLD;
    uint32_t** neighsHD;
    float*     furthest_neighdists_LD;
    // float kernel_LD_alpha;
    bool* thread_waiting_for_task;
    pthread_mutex_t* mutexes_sizeN;
    pthread_mutex_t* thread_mutex;
    uint32_t*    random_indices_exploration;    
    uint32_t*    random_indices_exploitation_HD;    
    uint32_t*    random_indices_exploitation_LD;    
} SubthreadData;

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
    bool* threads_waiting_for_task;

    // coordinating LD/HD compute (want goos balance between the two)
    float*           other_space_pct;
    pthread_mutex_t* mutex_LDHD_balance;
    
    
    // Algorithm and subthread data: for determining LD neighbours
    SubthreadData* subthread_data;
    uint32_t   N;
    float**    Xld;
    uint32_t   Mld;
    uint32_t   Khd;
    uint32_t   Kld;
    uint32_t** neighsLD;
    uint32_t** neighsHD;
    float*     furthest_neighdists_LD;
    float      pct_new_neighs;
    pthread_mutex_t* mutexes_sizeN;
} NeighLDDiscoverer;


void new_NeighLDDiscoverer(NeighLDDiscoverer* thing, uint32_t _N_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads,\
    pthread_mutex_t* mutexes_sizeN, float** _Xld_, float** _Xhd_, uint32_t _Mld_, uint32_t _Khd_,\
    uint32_t _Kld_, uint32_t** _neighsLD_, uint32_t** _neighsHD_, float* furthest_neighdists_LD,\
    float* _ptr_kernel_LD_alpha_, pthread_mutex_t* _mutex_kernel_LD_alpha_, pthread_mutex_t* mutex_LDHD_balance, float* other_space_pct);

void  destroy_NeighLDDiscoverer(NeighLDDiscoverer* thing);
void* routine_NeighLDDiscoverer(void* arg);
void  start_thread_NeighLDDiscoverer(NeighLDDiscoverer* thing);

void* subroutine_NeighLDDiscoverer(void* arg);

// refines the estimated set of LD neighbours for points between L and R
// return the estimation of the total sum of q_ij based on the q_ij that were computed during the neigbhour refinenement process
void refine_LD_neighbours(SubthreadData* thing);
// attempts to add j into j's neighbours,
// at the end, recheck with the lock if the neighbour is still to be added, and add using the lock if so
bool attempt_to_add_LD_neighbour(uint32_t i, uint32_t j, float euclsq_ij, SubthreadData* thing); 

#endif // NEIGHHD_DISCOVERER_H