#include "neighHD_discoverer.h"

void new_NeighHDDiscoverer(NeighHDDiscoverer* thing, uint32_t _N_, uint32_t _M_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads,\
        pthread_mutex_t* mutexes_sizeN, float** _Xhd_, uint32_t _Khd_, uint32_t** _neighsHD_, uint32_t** _neighsLD_,\
        float* furthest_neighdists_HD, float** _Psym_,\
    float* perplexity, pthread_mutex_t* mutex_perplexity, pthread_mutex_t*  mutex_LDHD_balance, float* other_space_pct){
    // worker and subthread management
    thing->isRunning = false;
    thing->rand_state = ++thread_rand_seed[0];
    thing->N_reserved_subthreads = max_nb_of_subthreads;
    thing->N_subthreads_target   = max_nb_of_subthreads;
    thing->subthreads_chunck_size = 1 + (uint32_t)floorf(SUBTHREADS_CHUNK_SIZE_PCT * (float)_N_);
    printf("subthreads chunck size %d\n", thing->subthreads_chunck_size);
    thing->subthreads = (pthread_t*)malloc(sizeof(pthread_t) * max_nb_of_subthreads);
    thing->subthreads_mutexes = mutexes_allocate_and_init(max_nb_of_subthreads);
    thing->threads_waiting_for_task = malloc_bool(max_nb_of_subthreads, true);

    // coordinating LD/HD compute (want goos balance between the two)
    thing->mutex_LDHD_balance       = mutex_LDHD_balance;
    thing->other_space_pct          = other_space_pct;

    // safe GPU / CPU communication: neighsHD and Psym
    thing->GPU_CPU_comms_neighsHD = malloc_GPU_CPU_uint32_buffer(_N_* _Khd_);
    thing->GPU_CPU_comms_Psym     = malloc_GPU_CPU_float_buffer(_N_ * _Khd_);

    // initialise algorithm data on this thread
    thing->N = _N_;
    thing->Mhd = _M_;
    thing->Xhd = _Xhd_;
    thing->Khd = _Khd_;
    thing->neighsHD = _neighsHD_;
    thing->neighsLD = _neighsLD_;
    thing->furthest_neighdists_HD = furthest_neighdists_HD;
    thing->Pasym = malloc_float_matrix(_N_, _Khd_, 1.0f);
    thing->dists_neighHD = malloc_float_matrix(_N_, _Khd_, 1.0f);
    thing->Pasym_sumJ_Pij = malloc_float(_N_, 1.0f);
    thing->Psym = _Psym_;
    thing->radii = malloc_float(_N_, 1.0f);
    thing->target_perplexity = perplexity;
    thing->mutex_target_perplexity = mutex_perplexity;

    // initialise Pasym, dists_neighHD, Pasym_sumJ_Pij
    for(uint32_t i = 0u; i < _N_; i++){
        float sumJ_Pij = 0.0f;
        for(uint32_t k = 0u; k < _Khd_; k++){
            uint32_t j = _neighsHD_[i][k];
            float eucl = f_euclidean_sq(_Xhd_[i], _Xhd_[j], _M_);
            thing->dists_neighHD[i][k] = eucl;
            float radius = thing->radii[i];
            float pij = expf(-eucl / radius) + FLOAT_EPS;
            thing->Pasym[i][k] = pij;
            sumJ_Pij += pij;
        }
        thing->Pasym_sumJ_Pij[i] = sumJ_Pij;
    }
    // values for Psym_GT
    for(uint32_t i = 0u; i < _N_; i++){
        for(uint32_t k = 0u; k < _Khd_; k++){
            uint32_t j = _neighsHD_[i][k];
            float eucl = thing->dists_neighHD[i][k];
            float pij  = thing->Pasym[i][k] / thing->Pasym_sumJ_Pij[i];
            float pji  = (expf(-eucl / thing->radii[j]) + FLOAT_EPS) / thing->Pasym_sumJ_Pij[j];
            if(pji > 1.0f){pji = 1.0f;}
            if(pij > 1.0f){dying_breath("ok this is wierd");}
            float p_symmetrised = (pij + pji) / (2.0f * (float)thing->N);
            thing->Psym[i][k] = p_symmetrised;
        }
    }

    thing->flag_neigh_update = malloc_bool(_N_, true);
    thing->pct_new_neighs = 1.0f;
    thing->mutexes_sizeN = mutexes_sizeN;
    thing->subthreadHD_data = (SubthreadHD_data*)malloc(sizeof(SubthreadHD_data) * max_nb_of_subthreads);
    for(uint32_t i = 0u; i < max_nb_of_subthreads; i++){
        SubthreadHD_data* subthread_data = &thing->subthreadHD_data[i];
        subthread_data->stop_this_thread = false;
        subthread_data->task_number = 0;
        subthread_data->N = _N_;
        subthread_data->rand_state = ++thread_rand_seed[0];
        printf("(subthread) %d rand state\n", subthread_data->rand_state);
        subthread_data->L = 0u;
        subthread_data->R = 0u;
        subthread_data->N_new_neighs = 0u;
        subthread_data->Xhd = _Xhd_;
        subthread_data->Mhd = _M_;
        subthread_data->Khd = _Khd_;
        subthread_data->neighsHD = _neighsHD_;
        subthread_data->neighsLD = _neighsLD_;
        subthread_data->furthest_neighdists_HD = furthest_neighdists_HD;
        subthread_data->Pasym = thing->Pasym;
        subthread_data->Pasym_sumJ_Pij = thing->Pasym_sumJ_Pij;
        subthread_data->flag_neigh_update = thing->flag_neigh_update;
        subthread_data->Psym = thing->Psym;
        subthread_data->dists_neighHD = thing->dists_neighHD;
        subthread_data->radii = thing->radii;
        subthread_data->target_perplexity = thing->target_perplexity;
        subthread_data->mutex_target_perplexity = thing->mutex_target_perplexity;
        subthread_data->thread_waiting_for_task = &thing->threads_waiting_for_task[i];
        subthread_data->mutexes_sizeN = mutexes_sizeN;
        subthread_data->thread_mutex = &thing->subthreads_mutexes[i];
        subthread_data->random_indices_exploration = malloc_uint32_t(NEIGH_FAR_EXPLORATION_N_SAMPLES, 0u);
        subthread_data->random_indices_exploitation_LD = malloc_uint32_t(NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES, 0u);
        subthread_data->random_indices_exploitation_HD = malloc_uint32_t(NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES, 0u);
    }
}

void destroy_NeighHDDiscoverer(NeighHDDiscoverer* thing) {
    for(uint32_t i = 0u; i < thing->N_reserved_subthreads; i++){
        pthread_mutex_destroy(&thing->subthreads_mutexes[i]);
    }
    free(thing->subthreads);
    free(thing->subthreads_mutexes);
    free(thing->threads_waiting_for_task);
    free(thing->Pasym_sumJ_Pij);
    free(thing->flag_neigh_update);
    free(thing->radii);
    free_matrix((void**)thing->Pasym, thing->N);
    free(thing->subthreadHD_data);
    free(thing);
}

static void wait_full_path_finished(NeighHDDiscoverer* thing){
    // check each subthread and sleep 1pct of a second untill all are waiting for a task
    bool all_idle = false;
    while(!all_idle){
        all_idle = true;
        for(uint32_t i = 0u; i < thing->N_reserved_subthreads; i++){
            pthread_mutex_lock(&thing->subthreads_mutexes[i]);
            if(!thing->threads_waiting_for_task[i]){
                all_idle = false;
            }
            pthread_mutex_unlock(&thing->subthreads_mutexes[i]);
        }
        if(all_idle){
            sleep_ms(10); // 10 ms (1% of a second)
        }
    }
}

// sync targets: neighsHD and Psym
void NeighHDDiscoverer_perhaps_sync_with_GPU(NeighHDDiscoverer* thing){
    if(!USE_GPU){
        return;}
    // for neigh_HD
    GPU_CPU_sync* sync_neighsHD = &thing->GPU_CPU_comms_neighsHD->sync;
    if(is_requesting_now(sync_neighsHD) && !is_ready_now(sync_neighsHD)){
        // wait for the subthreads to finish
        wait_full_path_finished(thing);
        // copy the neighsHD to the buffer, safely
        pthread_mutex_lock(thing->GPU_CPU_comms_neighsHD->sync.mutex_buffer);
        memcpy(thing->GPU_CPU_comms_neighsHD->buffer, as_uint32_1d(thing->neighsHD, thing->N, thing->Khd), thing->N*thing->Khd*sizeof(uint32_t));
        pthread_mutex_unlock(thing->GPU_CPU_comms_neighsHD->sync.mutex_buffer);
        // notify the GPU that the data is ready
        notify_ready(sync_neighsHD);
    }
    
    // for Psym
    GPU_CPU_sync* sync_Psym = &thing->GPU_CPU_comms_Psym->sync;
    if(is_requesting_now(sync_Psym) && !is_ready_now(sync_Psym)){
        // wait for the subthreads to finish
        wait_full_path_finished(thing);
        // copy the Psym to the buffer, safely
        pthread_mutex_lock(thing->GPU_CPU_comms_Psym->sync.mutex_buffer);
        memcpy(thing->GPU_CPU_comms_Psym->buffer, as_float_1d(thing->Psym, thing->N, thing->Khd), thing->N*thing->Khd*sizeof(float));
        pthread_mutex_unlock(thing->GPU_CPU_comms_Psym->sync.mutex_buffer);
        // notify the GPU that the data is ready
        notify_ready(sync_Psym);
    }
}

bool attempt_to_add_HD_neighbour(uint32_t i, uint32_t j, float euclsq_ij, SubthreadHD_data* thing){
    // in HD, the HD distances never change: no need to recompute everything
    // 1: go through distances in HD and find the 2 furthest neighbours
    float furthest_d_i = -1.0f;
    float second_furthest_d_i = -1.0f;
    uint32_t furthest_k = 0u;
    pthread_mutex_lock(&thing->mutexes_sizeN[i]);
    for(uint32_t k = 0u; k < thing->Khd; k++){
        if(thing->neighsHD[i][k] == j){
            pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
            return false;// j is already a neighbour of i
        } 
        float dist = thing->dists_neighHD[i][k];
        if(dist > furthest_d_i){
            second_furthest_d_i = furthest_d_i;
            furthest_d_i = dist;
            furthest_k = k;}
        else if(dist > second_furthest_d_i){
           second_furthest_d_i = dist;}
    }
    // re-check that j is candidate for being a neighbour of i
    if(euclsq_ij < furthest_d_i){
        thing->neighsHD[i][furthest_k] = j;
        thing->dists_neighHD[i][furthest_k] = euclsq_ij;
        thing->furthest_neighdists_HD[i] = euclsq_ij > second_furthest_d_i ? euclsq_ij : second_furthest_d_i;
        thing->flag_neigh_update[i] = true;
        pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
        return true;
    }
    else{
        thing->furthest_neighdists_HD[i] = furthest_d_i;
        pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
        return false;
    }
}

void refine_HD_neighbours(SubthreadHD_data* thing){
    // -----------------  generate random uint32_T for exploration and exploitation -----------------
    // between 0 and N
    for(uint32_t i = 0u; i < NEIGH_FAR_EXPLORATION_N_SAMPLES; i++){
        thing->random_indices_exploration[i] = rand_uint32_between(&thing->rand_state, 0u, thing->N);}
    // between 0 and Kld
    for(uint32_t i = 0u; i < NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES; i++){
        thing->random_indices_exploitation_LD[i] = rand_uint32_between(&thing->rand_state, 0u, Kld);}
    // between 0 and Khd
    for(uint32_t i = 0u; i < NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES; i++){
        thing->random_indices_exploitation_HD[i] = rand_uint32_between(&thing->rand_state, 0u, thing->Khd);}
    

    // -----------------  for each point: -----------------
    // 1/  find new neighbours 
    // 2/  if i gets new neigh j :  flag of i is set (if j gets i: flag of j is set)
    // 3/  if flag of i:  i recomputes Pasym
    uint32_t n_new_neighs  = 0u;
    // temp variables filled when mutex are acquired
    float    euclsq_ij    = 1.0f;
    float    furthest_d_i = 1.0f;
    float    furthest_d_j = 1.0f;
    for(uint32_t i = thing->L; i < thing->R; i++){
        bool new_neigh = false;
        // -------------------  TODO  ---------------------------
        // clever algorithm with 3 or more point: i, j, and r. compute dists dir and drj.
        // We should be able to tell if j and i are also candidate based on the 2 dists and the radius of i and j
        // extension to higher number points: better efficiency (i.e. more candidates for fewer Euclidean() call?)
        // ------------------------------------------------------

        // 1: exploration: random point j in [0, N[
        uint32_t j = thing->random_indices_exploration[i%NEIGH_FAR_EXPLORATION_N_SAMPLES];
        if(i != j){
            uint32_t i_1 = i < j ? i : j;
            uint32_t i_2 = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
            euclsq_ij    = f_euclidean_sq(thing->Xhd[i], thing->Xhd[j], thing->Mhd);
            furthest_d_i = thing->furthest_neighdists_HD[i];
            furthest_d_j = thing->furthest_neighdists_HD[j];
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
            if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                if(attempt_to_add_HD_neighbour(i, j, euclsq_ij, thing)){
                    new_neigh = true;
                    // propagate the new neighbour to HD_NEIGH_PROPAGATION_N other neighbours
                    for(uint32_t k = 0u; k < thing->Khd; k++){
                        uint32_t i_candidate = thing->neighsHD[i][k];
                        if(i_candidate != j){
                            uint32_t i_1 = i_candidate < j ? i_candidate : j;
                            uint32_t i_2 = i_candidate < j ? j : i_candidate;
                            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
                            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]);
                            float euclsq_ij_candidate = f_euclidean_sq(thing->Xhd[i_candidate], thing->Xhd[j], thing->Mhd);
                            float furthest_d_i_candidate = thing->furthest_neighdists_HD[i_candidate];
                            float furthest_d_j_candidate = thing->furthest_neighdists_HD[j];
                            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
                            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]);
                            if(euclsq_ij_candidate < furthest_d_i_candidate){
                                if(attempt_to_add_HD_neighbour(i_candidate, j, euclsq_ij_candidate, thing)){
                                    new_neigh = true;}
                            }
                            if(euclsq_ij_candidate < furthest_d_j_candidate){
                                if(attempt_to_add_HD_neighbour(j, i_candidate, euclsq_ij_candidate, thing)){
                                    new_neigh = true;}
                            }
                        }
                    }
                }
            }
            if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                if(attempt_to_add_HD_neighbour(j, i, euclsq_ij, thing)){
                    new_neigh = true;}
            }
        }
        // 2: exploitation: neighbour of neighbour
        // 2.1 : no bias 
        uint32_t k1 = thing->random_indices_exploitation_HD[(i+0)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        uint32_t k2 = thing->random_indices_exploitation_HD[(i+1)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        if(k2 == k1){
            k2 = (k2 + 1) % thing->Khd;}
        j = thing->neighsHD[thing->neighsHD[i][k1]][k2];
        if(i != j){
            uint32_t i_1 = i < j ? i : j;
            uint32_t i_2 = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
            euclsq_ij    = f_euclidean_sq(thing->Xhd[i], thing->Xhd[j], thing->Mhd);
            furthest_d_i = thing->furthest_neighdists_HD[i];
            furthest_d_j = thing->furthest_neighdists_HD[j];
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
            if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                if(attempt_to_add_HD_neighbour(i, j, euclsq_ij, thing)){
                    new_neigh = true;
                    // propagate the new neighbour to 5 other neighbours
                    for(uint32_t k = 0u; k < thing->Khd; k++){
                        uint32_t i_candidate = thing->neighsHD[i][k];
                        if(i_candidate != j){
                            uint32_t i_1 = i_candidate < j ? i_candidate : j;
                            uint32_t i_2 = i_candidate < j ? j : i_candidate;
                            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
                            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]);
                            float euclsq_ij_candidate = f_euclidean_sq(thing->Xhd[i_candidate], thing->Xhd[j], thing->Mhd);
                            float furthest_d_i_candidate = thing->furthest_neighdists_HD[i_candidate];
                            float furthest_d_j_candidate = thing->furthest_neighdists_HD[j];
                            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
                            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]);
                            if(euclsq_ij_candidate < furthest_d_i_candidate){
                                if(attempt_to_add_HD_neighbour(i_candidate, j, euclsq_ij_candidate, thing)){
                                    new_neigh = true;}
                            }
                            if(euclsq_ij_candidate < furthest_d_j_candidate){
                                if(attempt_to_add_HD_neighbour(j, i_candidate, euclsq_ij_candidate, thing)){
                                    new_neigh = true;}
                            }
                        }
                    }
                }
            }
            if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                if(attempt_to_add_HD_neighbour(j, i, euclsq_ij, thing)){
                    new_neigh = true;}
            }
        }
        // 2.2 : bias towards small k values
        uint32_t tmpK1 = thing->random_indices_exploitation_HD[(i+3u)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        uint32_t tmpK2 = thing->random_indices_exploitation_HD[(i+4u)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        uint32_t tmpK3 = thing->random_indices_exploitation_HD[(i+5u)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        k1 = tmpK1 < tmpK2 ? tmpK1 : tmpK2;
        k1 = k1 < tmpK3 ? k1 : tmpK3;
        uint32_t tmpK4 = thing->random_indices_exploitation_HD[(i+6u)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        uint32_t tmpK5 = thing->random_indices_exploitation_HD[(i+7u)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        uint32_t tmpK6 = thing->random_indices_exploitation_HD[(i+8u)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        k2 = tmpK4 < tmpK5 ? tmpK4 : tmpK5;
        k2 = k2 < tmpK6 ? k2 : tmpK6;
        if(k2 == k1){
            k2 = (k2 + 1) % thing->Khd;}
        j = thing->neighsHD[thing->neighsHD[i][k1]][k2];
        if(i != j){
            uint32_t i_1 = i < j ? i : j;
            uint32_t i_2 = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
            euclsq_ij    = f_euclidean_sq(thing->Xhd[i], thing->Xhd[j], thing->Mhd);
            furthest_d_i = thing->furthest_neighdists_HD[i];
            furthest_d_j = thing->furthest_neighdists_HD[j];
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
            if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                if(attempt_to_add_HD_neighbour(i, j, euclsq_ij, thing)){
                    new_neigh = true;
                    // propagate the new neighbour to 5 other neighbours
                    for(uint32_t k = 0u; k < thing->Khd; k++){
                        uint32_t i_candidate = thing->neighsHD[i][k];
                        if(i_candidate != j){
                            uint32_t i_1 = i_candidate < j ? i_candidate : j;
                            uint32_t i_2 = i_candidate < j ? j : i_candidate;
                            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
                            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]);
                            float euclsq_ij_candidate = f_euclidean_sq(thing->Xhd[i_candidate], thing->Xhd[j], thing->Mhd);
                            float furthest_d_i_candidate = thing->furthest_neighdists_HD[i_candidate];
                            float furthest_d_j_candidate = thing->furthest_neighdists_HD[j];
                            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
                            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]);
                            if(euclsq_ij_candidate < furthest_d_i_candidate){
                                if(attempt_to_add_HD_neighbour(i_candidate, j, euclsq_ij_candidate, thing)){
                                    new_neigh = true;}
                            }
                            if(euclsq_ij_candidate < furthest_d_j_candidate){
                                if(attempt_to_add_HD_neighbour(j, i_candidate, euclsq_ij_candidate, thing)){
                                    new_neigh = true;}
                            }
                        }
                    } 
                }
            }
            if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                if(attempt_to_add_HD_neighbour(j, i, euclsq_ij, thing)){
                    new_neigh = true;}
            }
        }
        if(k1 > 0u){
            k1 = k1 - 1u;
            if(k2 == k1){
            k2 = (k2 + 1) % thing->Khd;}
            j = thing->neighsHD[thing->neighsHD[i][k1]][k2];
            if(i != j){
                uint32_t i_1 = i < j ? i : j;
                uint32_t i_2 = i < j ? j : i;
                pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
                pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
                euclsq_ij    = f_euclidean_sq(thing->Xhd[i], thing->Xhd[j], thing->Mhd);
                furthest_d_i = thing->furthest_neighdists_HD[i];
                furthest_d_j = thing->furthest_neighdists_HD[j];
                pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
                pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
                if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                    if(attempt_to_add_HD_neighbour(i, j, euclsq_ij, thing)){
                        new_neigh = true;}
                }
                if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                    if(attempt_to_add_HD_neighbour(j, i, euclsq_ij, thing)){
                        new_neigh = true;}
                }
            }
        }
        uint32_t k3 = thing->random_indices_exploitation_HD[(i+9u)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        uint32_t k4 = thing->random_indices_exploitation_HD[(i+10u)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        uint32_t k5 = thing->random_indices_exploitation_HD[(i+11u)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        uint32_t k6 = thing->random_indices_exploitation_HD[(i+12u)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        uint32_t k7 = thing->random_indices_exploitation_HD[(i+13u)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        // find the smallest two values of k among k3, k4, k5, k6, k7
        uint32_t kmin1 = k3;
        uint32_t kmin2 = k4;
        if (k5 < kmin1) {
            kmin2 = kmin1;
            kmin1 = k5;
        } else if (k5 < kmin2) {
            kmin2 = k5;
        }
        if (k6 < kmin1) {
            kmin2 = kmin1;
            kmin1 = k6;
        } else if (k6 < kmin2) {
            kmin2 = k6;
        }
        if (k7 < kmin1) {
            kmin2 = kmin1;
            kmin1 = k7;
        } else if (k7 < kmin2) {
            kmin2 = k7;
        }
        j = thing->neighsHD[thing->neighsHD[i][kmin1]][kmin2];
        if(i != j){
            uint32_t i_1 = i < j ? i : j;
            uint32_t i_2 = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
            euclsq_ij    = f_euclidean_sq(thing->Xhd[i], thing->Xhd[j], thing->Mhd);
            furthest_d_i = thing->furthest_neighdists_HD[i];
            furthest_d_j = thing->furthest_neighdists_HD[j];
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
            if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                if(attempt_to_add_HD_neighbour(i, j, euclsq_ij, thing)){
                    new_neigh = true;
                    // propagate the new neighbour to HD_NEIGH_PROPAGATION_N other neighbours
                    for(uint32_t k = 0u; k < thing->Khd; k++){
                        uint32_t i_candidate = thing->neighsHD[i][k];
                        if(i_candidate != j){
                            uint32_t i_1 = i_candidate < j ? i_candidate : j;
                            uint32_t i_2 = i_candidate < j ? j : i_candidate;
                            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
                            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]);
                            float euclsq_ij_candidate = f_euclidean_sq(thing->Xhd[i_candidate], thing->Xhd[j], thing->Mhd);
                            float furthest_d_i_candidate = thing->furthest_neighdists_HD[i_candidate];
                            float furthest_d_j_candidate = thing->furthest_neighdists_HD[j];
                            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
                            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]);
                            if(euclsq_ij_candidate < furthest_d_i_candidate){
                                if(attempt_to_add_HD_neighbour(i_candidate, j, euclsq_ij_candidate, thing)){
                                    new_neigh = true;}
                            }
                            if(euclsq_ij_candidate < furthest_d_j_candidate){
                                if(attempt_to_add_HD_neighbour(j, i_candidate, euclsq_ij_candidate, thing)){
                                    new_neigh = true;}
                            }
                        }
                    }
                }
            }
            if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                if(attempt_to_add_HD_neighbour(j, i, euclsq_ij, thing)){
                    new_neigh = true;}
            }
        }
        j = thing->neighsHD[thing->neighsHD[i][kmin2]][kmin1];
        if(i != j){
            uint32_t i_1 = i < j ? i : j;
            uint32_t i_2 = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
            euclsq_ij    = f_euclidean_sq(thing->Xhd[i], thing->Xhd[j], thing->Mhd);
            furthest_d_i = thing->furthest_neighdists_HD[i];
            furthest_d_j = thing->furthest_neighdists_HD[j];
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
            if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                if(attempt_to_add_HD_neighbour(i, j, euclsq_ij, thing)){
                    new_neigh = true;
                    // propagate the new neighbour to 5 other neighbours
                    for(uint32_t k = 0u; k < thing->Khd; k++){
                        uint32_t i_candidate = thing->neighsHD[i][k];
                        if(i_candidate != j){
                            uint32_t i_1 = i_candidate < j ? i_candidate : j;
                            uint32_t i_2 = i_candidate < j ? j : i_candidate;
                            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
                            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]);
                            float euclsq_ij_candidate = f_euclidean_sq(thing->Xhd[i_candidate], thing->Xhd[j], thing->Mhd);
                            float furthest_d_i_candidate = thing->furthest_neighdists_HD[i_candidate];
                            float furthest_d_j_candidate = thing->furthest_neighdists_HD[j];
                            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
                            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]);
                            if(euclsq_ij_candidate < furthest_d_i_candidate){
                                if(attempt_to_add_HD_neighbour(i_candidate, j, euclsq_ij_candidate, thing)){
                                    new_neigh = true;}
                            }
                            if(euclsq_ij_candidate < furthest_d_j_candidate){
                                if(attempt_to_add_HD_neighbour(j, i_candidate, euclsq_ij_candidate, thing)){
                                    new_neigh = true;}
                            }
                        }
                    }
                }
            }
            if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                if(attempt_to_add_HD_neighbour(j, i, euclsq_ij, thing)){
                    new_neigh = true;}
            }
        }

        // 4: exploitation: neighbour from neighbour list of the other space
        uint32_t kld = thing->random_indices_exploitation_LD[i%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        j = thing->neighsLD[i][kld];
        if(i!=j){
            uint32_t i_1 = i < j ? i : j;
            uint32_t i_2 = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
            euclsq_ij    = f_euclidean_sq(thing->Xhd[i], thing->Xhd[j], thing->Mhd);
            furthest_d_i = thing->furthest_neighdists_HD[i];
            furthest_d_j = thing->furthest_neighdists_HD[j];
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
            if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                if(attempt_to_add_HD_neighbour(i, j, euclsq_ij, thing)){
                    new_neigh = true;

                    for(uint32_t k = 0u; k < thing->Khd; k++){
                        uint32_t i_candidate = thing->neighsHD[i][k];
                        if(i_candidate != j){
                            uint32_t i_1 = i_candidate < j ? i_candidate : j;
                            uint32_t i_2 = i_candidate < j ? j : i_candidate;
                            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
                            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]);
                            float euclsq_ij_candidate = f_euclidean_sq(thing->Xhd[i_candidate], thing->Xhd[j], thing->Mhd);
                            float furthest_d_i_candidate = thing->furthest_neighdists_HD[i_candidate];
                            float furthest_d_j_candidate = thing->furthest_neighdists_HD[j];
                            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
                            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]);
                            if(euclsq_ij_candidate < furthest_d_i_candidate){
                                if(attempt_to_add_HD_neighbour(i_candidate, j, euclsq_ij_candidate, thing)){
                                    new_neigh = true;}
                            }
                            if(euclsq_ij_candidate < furthest_d_j_candidate){
                                if(attempt_to_add_HD_neighbour(j, i_candidate, euclsq_ij_candidate, thing)){
                                    new_neigh = true;}
                            }
                        }
                    }
                }
            }
            if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                if(attempt_to_add_HD_neighbour(j, i, euclsq_ij, thing)){
                    new_neigh = true;}
            }
        }

        // 5: sorting : refine the neighbour ordering stochatically and quickly
        if(new_neigh || ((i%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES) < 1)){
            for(uint32_t rep = 0; rep < 30; rep++){
                uint32_t k_1 = thing->random_indices_exploitation_HD[i%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
                uint32_t k_2 = thing->random_indices_exploitation_HD[(i+3)%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
                if(k_1 == k_2){k_2 = (k_2 + 1u) % thing->Khd;} // make sure they are different
                if((thing->dists_neighHD[i][k_1] < thing->dists_neighHD[i][k_2]) && k_1 > k_2){ // need to swap these two
                    pthread_mutex_lock(&thing->mutexes_sizeN[i]);
                    uint32_t j1 = thing->neighsHD[i][k_1];
                    float    d1 = thing->dists_neighHD[i][k_1];
                    uint32_t j2 = thing->neighsHD[i][k_2];
                    float    d2 = thing->dists_neighHD[i][k_2];
                    thing->neighsHD[i][k_1] = j2;
                    thing->dists_neighHD[i][k_1] = d2;
                    thing->neighsHD[i][k_2] = j1;
                    thing->dists_neighHD[i][k_2] = d1;
                    pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
                }
            }
        }
        /* 
        -faire en lock-free pour les reading (expensive)
        verifier que zero sleep*/
        if(new_neigh){n_new_neighs++;}
    }

    // save the number of new neighbours and notify the thread that it is waiting for a task
    pthread_mutex_lock(thing->thread_mutex);
    thing->N_new_neighs = n_new_neighs;
    thing->thread_waiting_for_task[0] = true;
    pthread_mutex_unlock(thing->thread_mutex);
    return;
}

/* float log_based(float base, float x){
    return logf(x) / logf(base);
} */

void update_radii(SubthreadHD_data* thing){
    // float base_of_log = 2.0f;
    float target_perplexity = thing->target_perplexity[0];
    float PP_tol = 0.01f * target_perplexity;
    if(PP_tol < 0.05){
        PP_tol = 0.05;}
    float R_entropy = logf(target_perplexity + PP_tol);
    float L_entropy = logf(target_perplexity - PP_tol);
    /* float R_entropy = log_based(base_of_log, target_perplexity + PP_tol);
    float L_entropy = log_based(base_of_log, target_perplexity - PP_tol); */
    float desired_entropy = (R_entropy + L_entropy) / 2.0f;
    
    float temp_pijs[thing->Khd];
    float mmtm_alpha = 0.9f;
    for(uint32_t i = thing->L; i < thing->R; i++){
        // 1: filter out points that did not have a change in their neighbours
        if(!thing->flag_neigh_update[i]){
            continue;}
        thing->flag_neigh_update[i] = false;

        float beta_min = 0.0f;
        float beta_max = 99999999999.0f;
        float beta     = 0.0001f;

        // initialise max_beta and min_beta by growing beta
        bool growing_beta = true;
        uint32_t iter2 = 0u;
        uint32_t iter1 = 0u;
        while(growing_beta){
            iter1++;
            float sumPi = 0.0f;
            for(uint32_t k = 0u; k < thing->Khd; k++){
                uint32_t j = thing->neighsHD[i][k];
                float eucl = thing->dists_neighHD[i][k];
                temp_pijs[k] = expf(-eucl*beta);
                sumPi += temp_pijs[k];
            }
            sumPi = sumPi > 0.0f ? sumPi : FLOAT_EPS;
            float sum_P_x_dist = 0.0f;
            for(uint32_t k = 0u; k < thing->Khd; k++){
                sum_P_x_dist += (temp_pijs[k] / sumPi) * thing->dists_neighHD[i][k];
            }
            float entropy = logf(sumPi) + beta*sum_P_x_dist;
            // float entropy = log_based(base_of_log, sumPi) + beta*sum_P_x_dist;
            float entropy_diff = entropy - desired_entropy;
            if(entropy_diff > 0.){
                beta_min = beta;
                beta     = 5. * beta;
            } else{
                growing_beta = false;
                beta_max = beta;
                beta     = (beta_max + beta_min) / 2.;
            }
        }
        // binary search for the beta that gives the desired entropy
        bool converged = false;
        while(!converged && iter2 < 200u){
            iter2++;
            // compute entropy 
            float sumPi = 0.0f;
            for(uint32_t k = 0u; k < thing->Khd; k++){
                uint32_t j = thing->neighsHD[i][k];
                float eucl = thing->dists_neighHD[i][k];
                temp_pijs[k] = expf(-eucl*beta);
                sumPi += temp_pijs[k];
            }
            sumPi = sumPi > 0.0f ? sumPi : FLOAT_EPS;
            float sum_P_x_dist = 0.0f;
            for(uint32_t k = 0u; k < thing->Khd; k++){
                sum_P_x_dist += (temp_pijs[k] / sumPi) * thing->dists_neighHD[i][k];
            }
            float entropy = logf(sumPi) + beta*sum_P_x_dist;
            // float entropy = log_based(base_of_log, sumPi) + beta*sum_P_x_dist;
            converged = entropy > L_entropy && entropy < R_entropy;
            if(!converged){
                float entropy_diff = entropy - desired_entropy;
                if(entropy_diff > 0.){
                    beta_min = beta;
                    beta     = (beta + beta_max) / 2.;
                } else{
                    beta_max = beta;
                    beta     = (beta + beta_min) / 2.;
                }
            }
        }

        

        // update the radius
        float new_radius = 1.0f / (beta + FLOAT_EPS);
        pthread_mutex_lock(&thing->mutexes_sizeN[i]);
        thing->radii[i] = new_radius;
        pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
        //verify perplexity 
        // float H  = obs_H(thing, i, thing->radii[i]);
        // float PP = expf(H);
        // printf("PP: %f  target: %f\n", PP, target_perplexity);



        /* if(iter2 >= 200u){
            float H  = obs_H(thing, i, thing->radii[i]);
            float PP = expf(H);

            float saved_radius_min = 1.0f / (saved_beta_min + FLOAT_EPS);
            float saved_radius_max = 1.0f / (saved_beta_max + FLOAT_EPS);

            float H_min  = obs_H(thing, i, saved_radius_min);
            float PP_min = expf(H_min);
            float H_max  = obs_H(thing, i, saved_radius_max);
            float PP_max = expf(H_max);

            printf("PP: %f  target: %f\n", PP, target_perplexity);
            printf("PP_min: %f  PP_max: %f\n", PP_min, PP_max);
            die();
        } */
    }

    pthread_mutex_lock(thing->thread_mutex);
    thing->thread_waiting_for_task[0] = true;
    pthread_mutex_unlock(thing->thread_mutex);
}

inline float obs_H(SubthreadHD_data* thing, uint32_t i, float radius){
    float temp_pijs[thing->Khd];
    float beta = 1.0f / (FLOAT_EPS + radius);
    float sumPi = 0.0f;
    for(uint32_t k = 0u; k < thing->Khd; k++){
        uint32_t j = thing->neighsHD[i][k];
        float eucl = thing->dists_neighHD[i][k];
        temp_pijs[k] = expf(-eucl*beta);
        sumPi += temp_pijs[k];
    }
    sumPi = sumPi > 0.0f ? sumPi : FLOAT_EPS;
    float sum_P_x_dist = 0.0f;
    for(uint32_t k = 0u; k < thing->Khd; k++){
        sum_P_x_dist += (temp_pijs[k] / sumPi) * thing->dists_neighHD[i][k];
    }
    return logf(sumPi) + beta*sum_P_x_dist;
}

// no need ot lock the mutexes_sizeN[i] because we wait for all radii threads to be finished bfore this
void recompute_Pasym(SubthreadHD_data* thing){
    for(uint32_t i = thing->L; i < thing->R; i++){
        float sumJ_Pij = 0.0f;
        // pthread_mutex_lock(&thing->mutexes_sizeN[i]);
        for(uint32_t k = 0u; k < thing->Khd; k++){
            float pij = expf(-thing->dists_neighHD[i][k] / thing->radii[i]);
            thing->Pasym[i][k] = pij;
            sumJ_Pij          += pij;
        }
        thing->Pasym_sumJ_Pij[i] = sumJ_Pij > 0.0f ? sumJ_Pij : FLOAT_EPS;
        // pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
    }
    pthread_mutex_lock(thing->thread_mutex);
    thing->thread_waiting_for_task[0] = true;
    pthread_mutex_unlock(thing->thread_mutex);
}

// no need ot lock the mutexes_sizeN[i] because we wait for Pasym to be fully computed before this
void recompute_Psym(SubthreadHD_data* thing){
    float multiplier = 1.0f / (2.0f * (float)thing->N);
    for(uint32_t i = thing->L; i < thing->R; i++){
        // pthread_mutex_lock(&thing->mutexes_sizeN[i]);
        for(uint32_t k = 0u; k < thing->Khd; k++){
            uint32_t j = thing->neighsHD[i][k];
            float pij  = thing->Pasym[i][k] / thing->Pasym_sumJ_Pij[i];
            float pji  = expf(-thing->dists_neighHD[i][k] / thing->radii[j]) / thing->Pasym_sumJ_Pij[j];
            if(pji > 1.0f){ pji = 1.0f;} // this is possible, because neighs aren t perfect (yet) and the j denom might be way too small
            thing->Psym[i][k] = (pij + pji) * multiplier;

            if(thing->Psym[i][k] > 1.0f || thing->Psym[i][k] < -0.0f){
                dying_breath("--------  recompute_Psym: Pasym > 1.0f");
            } 

        }
        // pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
    }
    pthread_mutex_lock(thing->thread_mutex);
    thing->thread_waiting_for_task[0] = true;
    pthread_mutex_unlock(thing->thread_mutex);
}

void* subroutine_NeighHDDiscoverer(void* arg){
    SubthreadHD_data* thing = (SubthreadHD_data*)arg;
    while(!thing->stop_this_thread){
        // wait for a task
        pthread_mutex_lock(thing->thread_mutex);
        if(thing->thread_waiting_for_task[0]){
            pthread_mutex_unlock(thing->thread_mutex);
            usleep(10000); // 1% of a second
        }
        else{
            thing->N_new_neighs = 0u;
            uint32_t  task_number = thing->task_number;
            pthread_mutex_unlock(thing->thread_mutex);
            // do the task: refine HD neighbours, update radii and Pasym and Psym
            if(task_number == 0){
                refine_HD_neighbours(thing);
            }
            else if(task_number == 1u){
                update_radii(thing);
            }
            else if(task_number == 2u){
                recompute_Pasym(thing);
            }
            else if(task_number == 3u){
                recompute_Psym(thing);
            }
            else{
                dying_breath("subroutine_NeighHDDiscoverer: unknown task number");
            }
        }
    }
    return NULL;
}



void* routine_NeighHDDiscoverer(void* arg) {
    NeighHDDiscoverer* thing = (NeighHDDiscoverer*)arg;
    thing->isRunning = true;
    // launch subthreads
    for(uint32_t i = 0u; i < thing->N_reserved_subthreads; i++){
        if(pthread_create(&thing->subthreads[i], NULL, subroutine_NeighHDDiscoverer, &thing->subthreadHD_data[i]) != 0){
            dying_breath("pthread_create routine_NeighHDDiscoverer_subthread failed");}
    }
    // work dispatcher loop
    uint32_t cursor          = 0u;
    uint32_t N_neigh_finding = 10u;
    uint32_t task_counter    = 0u;
    while(thing->isRunning){
        // get the current value of N_subthreads_target, for use locally
        // 50:50 : 0.5*max:0.5*max     10:90 : 0.1*max:0.9*max
        pthread_mutex_lock(thing->mutex_LDHD_balance);
        float other_pct = thing->other_space_pct[0] * 0.1;
        float this_pct  = (thing->pct_new_neighs + HD_PCT_BIAS);
        float total     = FLOAT_EPS + other_pct + this_pct;
        float ressource_allocation_ratio = this_pct / total;
        uint32_t now_N_subthreads_target = (uint32_t)(ressource_allocation_ratio * (float)thing->N_reserved_subthreads);
        if(now_N_subthreads_target == 0u){
            now_N_subthreads_target = 1u;}
        pthread_mutex_unlock(thing->mutex_LDHD_balance);
        // printf("HD now_N_subthreads_target: %u\n", now_N_subthreads_target);
        for(uint32_t i = 0u; i < now_N_subthreads_target; i++){
            // if the subthread is waiting for a task: give a new task
            pthread_mutex_lock(&thing->subthreads_mutexes[i]);
            if(thing->threads_waiting_for_task[i]){
                SubthreadHD_data* subthread_data = &thing->subthreadHD_data[i];
                uint32_t task_number =  (task_counter < N_neigh_finding) ? 0u : 1u + (task_counter - N_neigh_finding);
                // 1.1: update estimated pct of new neighs
                if((subthread_data->task_number == 0) && subthread_data->R - subthread_data->L == thing->subthreads_chunck_size){
                    uint32_t N_new_neighs = subthread_data->N_new_neighs;
                    float pctage = N_new_neighs > thing->subthreads_chunck_size ? 1.0f : (float)N_new_neighs / (float)thing->subthreads_chunck_size;
                    thing->pct_new_neighs = thing->pct_new_neighs * 0.98f + 0.02f * pctage;
                }
                // 2: assign new task to the thread
                subthread_data->L = cursor;
                subthread_data->R = cursor + thing->subthreads_chunck_size > thing->N ? thing->N : cursor + thing->subthreads_chunck_size;
                subthread_data->task_number = task_number;
                thing->threads_waiting_for_task[i] = false;
                pthread_mutex_unlock(&thing->subthreads_mutexes[i]);
                // 3: update the cursor in N for the next subthread
                cursor += thing->subthreads_chunck_size;
                if(cursor >= thing->N){
                    cursor = 0u;
                    // update the counter that determines the job type
                    task_counter++;
                    if(task_counter >= N_neigh_finding + 3u){
                        task_counter = 0u;
                        NeighHDDiscoverer_perhaps_sync_with_GPU(thing); // just finished computing Psym: might want to sync with GPU
                    }

                    task_number = (task_counter < N_neigh_finding) ? 0u : 1u + (task_counter - N_neigh_finding);
                    if(task_number > 0u){ // if: will start working on radius or on Psym
                        wait_full_path_finished(thing);}
                }
            }
            else{
                pthread_mutex_unlock(&thing->subthreads_mutexes[i]);
            }
        }
        usleep(10000); // 1% of a second, prevent the thread from taking too much CPU time
    }
    dying_breath("routine_NeighHDDiscoverer ended");
    return NULL;
}

void start_thread_NeighHDDiscoverer(NeighHDDiscoverer* thing) {
    if(pthread_create(&thing->thread, NULL, routine_NeighHDDiscoverer, thing) != 0){
        dying_breath("pthread_create routine_NeighHDDiscoverer failed");}
}
