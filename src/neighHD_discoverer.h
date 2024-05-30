#ifndef NEIGHHD_DISCOVERER_H
#define NEIGHHD_DISCOVERER_H

#include <pthread.h>
#include "system.h"
#include "vector.h"
#include "probabilities.h"
#include "constants_global.h"

typedef struct {
    bool     stop_this_thread;
    uint32_t  task_number;
    uint32_t N;
    uint32_t rand_state;
    uint32_t      L;
    uint32_t      R;
    uint32_t    N_new_neighs;
    float**    Xhd;
    uint32_t   Mhd;
    uint32_t   Khd;
    uint32_t   Kld;
    uint32_t** neighsHD;
    uint32_t** neighsLD;
    float*     furthest_neighdists_HD;
    float**    dists_neighHD;
    float**    Pasym;
    float*     Pasym_sumJ_Pij;
    bool*      flag_neigh_update;
    float**    Psym;
    float*     radii;
    float*      target_perplexity;
    pthread_mutex_t* mutex_target_perplexity;
    
    bool* thread_waiting_for_task;
    pthread_mutex_t* mutexes_sizeN;
    pthread_mutex_t* thread_mutex;
    uint32_t*    random_indices_exploration;    
    uint32_t*    random_indices_exploitation_HD;    
    uint32_t*    random_indices_exploitation_LD;
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
    bool* threads_waiting_for_task;

    // coordinating LD/HD compute (want goos balance between the two)
    pthread_mutex_t* mutex_LDHD_balance;
    float*           other_space_pct;

    // Algorithm and subthread data: for determining LD neighbours, Q, and Qdenom
    SubthreadHD_data* subthreadHD_data;
    uint32_t   N;
    uint32_t   Mhd;
    float**    Xhd;
    uint32_t   Khd;
    uint32_t   Kld;
    uint32_t** neighsHD;
    uint32_t** neighsLD;
    float*     furthest_neighdists_HD;
    float*      target_perplexity;
    pthread_mutex_t* mutex_target_perplexity;

    float**    dists_neighHD;
    float**    Pasym;
    float*     Pasym_sumJ_Pij;
    bool*      flag_neigh_update;
    float**    Psym;
    float*     radii;
    float      pct_new_neighs;
    pthread_mutex_t* mutexes_sizeN;

} NeighHDDiscoverer;

void new_NeighHDDiscoverer(NeighHDDiscoverer* thing, uint32_t _N_, uint32_t _Mhd_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads,\
    pthread_mutex_t* mutexes_sizeN, float** _Xhd_, uint32_t _Khd_, uint32_t _Kld_, uint32_t** _neighsHD_, uint32_t** _neighsLD_,\
    float* furthest_neighdists_HD, float** _Psym_GT_,\
    float* perplexity, pthread_mutex_t* mutex_perplexity, pthread_mutex_t* mutex_LDHD_balance, float* other_space_pct);
void  destroy_NeighHDDiscoverer(NeighHDDiscoverer* thing);
void* routine_NeighHDDiscoverer(void* arg);
void start_thread_NeighHDDiscoverer(NeighHDDiscoverer* thing);
void wait_full_path_finished(NeighHDDiscoverer* thing);

bool attempt_to_add_HD_neighbour(uint32_t i, uint32_t j, float euclsq_ij, SubthreadHD_data* thing);

void* subroutine_NeighLDDiscoverer(void* arg);
void refine_HD_neighbours(SubthreadHD_data* thing);
void update_radii(SubthreadHD_data* thing);
void recompute_Pasym(SubthreadHD_data* thing);
void recompute_Psym(SubthreadHD_data* thing);

float obs_H(SubthreadHD_data* thing, uint32_t i, float radius);

#endif // NEIGHHD_DISCOVERER_H