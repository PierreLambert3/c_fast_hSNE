#include "neighLD_discoverer.h"



void new_NeighLDDiscoverer(NeighLDDiscoverer* thing, uint32_t _N_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads,\
    pthread_mutex_t* _mutex_Qdenom_, pthread_mutex_t* mutexes_sizeN, float** _Xld_, float** _Xhd_, uint32_t _Mld_, uint32_t _Khd_,\
    uint32_t _Kld_, uint32_t** _neighsLD_, uint32_t** _neighsHD_, float* furthest_neighdists_LD, float** _Q_, float* _Qdenom_,\
    float* _ptr_kernel_LD_alpha_, pthread_mutex_t* _mutex_kernel_LD_alpha_){

    // work-management data
    thing->isRunning = false;
    thing->rand_state = (uint32_t)time(NULL) +  ++thread_rand_seed[0];
    thing->N_reserved_subthreads  = max_nb_of_subthreads;
    thing->N_subthreads_target    = max_nb_of_subthreads;
    thing->subthreads_chunck_size = 1 + floorf(SUBTHREADS_CHUNK_SIZE_PCT * (float)_N_);
    thing->subthreads             = (pthread_t*)malloc(sizeof(pthread_t) * max_nb_of_subthreads);
    thing->threads_waiting_for_task = malloc_bool(max_nb_of_subthreads, true);
    thing->subthread_data           = (SubthreadData*)malloc(sizeof(SubthreadData) * max_nb_of_subthreads);
    thing->subthreads_mutexes       = mutexes_allocate_and_init(max_nb_of_subthreads);
    pthread_mutex_init(&thing->mutex_N_subthreads_target, NULL);

    // initialise algorithm data on this thread
    thing->N = _N_;
    thing->Xld = _Xld_;
    thing->Xhd = _Xhd_;
    thing->Mld = _Mld_;
    thing->Khd = _Khd_;
    thing->Kld = _Kld_;
    thing->neighsLD = _neighsLD_;
    thing->neighsHD = _neighsHD_;
    thing->furthest_neighdists_LD = furthest_neighdists_LD;
    thing->pct_new_neighs = 1.0f;
    thing->Q = _Q_;
    thing->ptr_Qdenom = _Qdenom_;
    thing->ptr_kernel_LD_alpha = _ptr_kernel_LD_alpha_;
    thing->mutex_Qdenom  = _mutex_Qdenom_;
    thing->mutexes_sizeN = mutexes_sizeN;
    thing->mutex_kernel_LD_alpha = _mutex_kernel_LD_alpha_;

    // initialize subthread internals
    for(uint32_t i = 0; i < max_nb_of_subthreads; i++){
        SubthreadData* subthread_data = &thing->subthread_data[i];
        subthread_data->stop_this_thread = false;
        subthread_data->N = _N_;
        subthread_data->rand_state = ++thread_rand_seed[0];
        printf("(subthread) %d rand state\n", subthread_data->rand_state);
        subthread_data->L = 0;
        subthread_data->R = 0;
        subthread_data->Xld = _Xld_;
        subthread_data->Xhd = _Xhd_;
        subthread_data->Mld = _Mld_;
        subthread_data->Khd = _Khd_;
        subthread_data->Kld = _Kld_;
        subthread_data->neighsLD = _neighsLD_;
        subthread_data->neighsHD = _neighsHD_;
        subthread_data->furthest_neighdists_LD = furthest_neighdists_LD;
        subthread_data->Q = _Q_;
        subthread_data->kernel_LD_alpha = _ptr_kernel_LD_alpha_[0];
        // lock with mutex_Qdenom
        pthread_mutex_lock(thing->mutex_Qdenom);
        subthread_data->estimated_Qdenom = thing->ptr_Qdenom[0];
        pthread_mutex_unlock(thing->mutex_Qdenom);
        subthread_data->mutexes_sizeN = mutexes_sizeN;
        subthread_data->thread_mutex = &thing->subthreads_mutexes[i];
        subthread_data->thread_waiting_for_task = &thing->threads_waiting_for_task[i];
        subthread_data->random_indices_exploration  = malloc_uint32_t(NEIGH_FAR_EXPLORATION_N_SAMPLES, 0);
        subthread_data->random_indices_exploitation_LD = malloc_uint32_t(NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES, 0);
        subthread_data->random_indices_exploitation_HD = malloc_uint32_t(NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES, 0);
    }
}

void destroy_NeighLDDiscoverer(NeighLDDiscoverer* thing){
    for(uint32_t i = 0; i < thing->N_reserved_subthreads; i++){
        pthread_mutex_destroy(&thing->subthreads_mutexes[i]);
        free(thing->subthread_data[i].random_indices_exploration);
        free(thing->subthread_data[i].random_indices_exploitation_LD);
        free(thing->subthread_data[i].random_indices_exploitation_HD);
    }
    free(thing->subthreads);
    free(thing->subthreads_mutexes);
    free(thing->subthread_data);
    free(thing);
}


bool attempt_to_add_neighbour(uint32_t i, uint32_t j, float euclsq_ij, SubthreadData* thing){
    // 1: can trust dists in LD : recompute all dists to find the 2 furthest neighbours
    float furthest_d_i = -1.0f;
    float second_furthest_d_i = -1.0f;
    uint32_t furthest_k = 0;
    // !!!!!!!!!!!  unsafe here: j position is not locked !!!!!!!!!!!!!!!!!
    pthread_mutex_lock(&thing->mutexes_sizeN[i]); //ici la merde
    for(uint32_t k = 0; k < thing->Kld; k++){
        if(thing->neighsLD[i][k] == j){
            pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
            return false;// j is already a neighbour of i
        } 
        float dist = f_euclidean_sq(thing->Xld[i], thing->Xld[thing->neighsLD[i][k]], thing->Mld);
        if(dist > furthest_d_i){
            second_furthest_d_i = furthest_d_i;
            furthest_d_i = dist;
            furthest_k = k;}
        else if(dist > second_furthest_d_i){
           second_furthest_d_i = dist;}
    }
    if(euclsq_ij < furthest_d_i){
        thing->neighsLD[i][furthest_k] = j;}
    float new_furthest_d_i = euclsq_ij > second_furthest_d_i ? euclsq_ij : second_furthest_d_i;
    thing->furthest_neighdists_LD[i] = new_furthest_d_i;
    pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
    return true;
}

void refine_LD_neighbours(SubthreadData* thing){
    // printf("it is important to update the furhtest dist to LD neighs in the tSNE optimisation, when computing them\n");
    // -----------------  generate random uint32_T for exploration and exploitation -----------------
    // between 0 and N
    for(uint32_t i = 0; i < NEIGH_FAR_EXPLORATION_N_SAMPLES; i++){
        thing->random_indices_exploration[i] = rand_uint32_between(&thing->rand_state, 0, thing->N);}
    // between 0 and Kld
    for(uint32_t i = 0; i < NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES; i++){
        thing->random_indices_exploitation_LD[i] = rand_uint32_between(&thing->rand_state, 0, thing->Kld);}
    // between 0 and Khd
    for(uint32_t i = 0; i < NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES; i++){
        thing->random_indices_exploitation_HD[i] = rand_uint32_between(&thing->rand_state, 0, thing->Khd);}
    // -----------------  for each point: -----------------
    // -----------------  find new neighbours -----------------
    // -----------------  remember that the furthest_LD dists are not up to date -----------------
    float kernel_LD_alpha = thing->kernel_LD_alpha;
    // variables for the denominator estimation
    double   dbl_acc_denom = 0.;
    uint32_t n_votes       = 0;
    uint32_t n_new_neighs  = 0;
    // temp variables filled when mutex are acquired
    float    euclsq_ij    = 1.0f;
    float    furthest_d_i = 1.0f;
    float    furthest_d_j = 1.0f;
    for(uint32_t i = thing->L; i < thing->R; i++){
        bool new_neigh = false;
        // -------------------  TODO  ---------------------------
        // clever algorithm with 3 or more point: i, j, and r. compute dists dir and drj.
        // We should be able to tell if j and i are also candidate based on the 2 dists ands the radius of i and j
        // ------------------------------------------------------

        // 1: exploration: random point j in [0, N[
        uint32_t j = thing->random_indices_exploration[i%NEIGH_FAR_EXPLORATION_N_SAMPLES];
        if(i != j){ 
            uint32_t i_1 = i < j ? i : j;
            uint32_t i_2 = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
            euclsq_ij    = f_euclidean_sq(thing->Xld[i], thing->Xld[j], thing->Mld);
            furthest_d_i = thing->furthest_neighdists_LD[i];
            furthest_d_j = thing->furthest_neighdists_LD[j];
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
            if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                if(attempt_to_add_neighbour(i, j, euclsq_ij, thing)){
                    new_neigh = true;}
            }
            if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                if(attempt_to_add_neighbour(j, i, euclsq_ij, thing)){
                    new_neigh = true;}
            }
            // update the denominator estimation
            dbl_acc_denom += (double) kernel_LD(euclsq_ij, kernel_LD_alpha);
            n_votes++;
        }
        
        // 2: exploitation: neighbour of neighbour
        // 2.1 : no bias
        uint32_t k1 = thing->random_indices_exploitation_LD[(i+0)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        uint32_t k2 = thing->random_indices_exploitation_LD[(i+1)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        if(k2 == k1){
            k2 = (k2 + 1) % thing->Kld;}
        j = thing->neighsLD[thing->neighsLD[i][k1]][k2];
        if(i != j){
            uint32_t i_1 = i < j ? i : j;
            uint32_t i_2 = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
            euclsq_ij    = f_euclidean_sq(thing->Xld[i], thing->Xld[j], thing->Mld);
            furthest_d_i = thing->furthest_neighdists_LD[i];
            furthest_d_j = thing->furthest_neighdists_LD[j];
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
            if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                if(attempt_to_add_neighbour(i, j, euclsq_ij, thing)){
                    new_neigh = true;}
            }
            if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                if(attempt_to_add_neighbour(j, i, euclsq_ij, thing)){
                    new_neigh = true;}
            }
            // update the denominator estimation
            dbl_acc_denom += (double) kernel_LD(euclsq_ij, kernel_LD_alpha);
            n_votes++;
        }

        // 2.2 : bias towards small k values
        uint32_t tmpK1 = thing->random_indices_exploitation_LD[(i+3)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        uint32_t tmpK2 = thing->random_indices_exploitation_LD[(i+4)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        uint32_t tmpK3 = thing->random_indices_exploitation_LD[(i+5)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        k1 = tmpK1 < tmpK2 ? tmpK1 : tmpK2;
        k1 = k1 < tmpK3 ? k1 : tmpK3;
        uint32_t tmpK4 = thing->random_indices_exploitation_LD[(i+5)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        uint32_t tmpK5 = thing->random_indices_exploitation_LD[(i+6)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        uint32_t tmpK6 = thing->random_indices_exploitation_LD[(i+7)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        k2 = tmpK4 < tmpK5 ? tmpK4 : tmpK5;
        k2 = k2 < tmpK6 ? k2 : tmpK6;
        if(k2 == k1){
            k2 = (k2 + 1) % thing->Kld;}
        j = thing->neighsLD[thing->neighsLD[i][k1]][k2];
        if(i != j){
            uint32_t i_1 = i < j ? i : j;
            uint32_t i_2 = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
            euclsq_ij    = f_euclidean_sq(thing->Xld[i], thing->Xld[j], thing->Mld);
            furthest_d_i = thing->furthest_neighdists_LD[i];
            furthest_d_j = thing->furthest_neighdists_LD[j];
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
            if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                if(attempt_to_add_neighbour(i, j, euclsq_ij, thing)){
                    new_neigh = true;}
            }
            if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                if(attempt_to_add_neighbour(j, i, euclsq_ij, thing)){
                    new_neigh = true;}
            }
            // update the denominator estimation
            dbl_acc_denom += (double) kernel_LD(euclsq_ij, kernel_LD_alpha);
            n_votes++;
        }  
        if(k1 > 0){
            k1 = k1 - 1;
            if(k2 == k1){
                k2 = (k2 + 1) % thing->Kld;}
            j = thing->neighsLD[thing->neighsLD[i][k1]][k2];
            if(i != j){
                uint32_t i_1 = i < j ? i : j;
                uint32_t i_2 = i < j ? j : i;
                pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
                pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
                euclsq_ij    = f_euclidean_sq(thing->Xld[i], thing->Xld[j], thing->Mld);
                furthest_d_i = thing->furthest_neighdists_LD[i];
                furthest_d_j = thing->furthest_neighdists_LD[j];
                pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
                pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
                if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                    if(attempt_to_add_neighbour(i, j, euclsq_ij, thing)){
                        new_neigh = true;}
                }
                if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                    if(attempt_to_add_neighbour(j, i, euclsq_ij, thing)){
                        new_neigh = true;}
                }
                // update the denominator estimation
                dbl_acc_denom += (double) kernel_LD(euclsq_ij, kernel_LD_alpha);
                n_votes++;
            }  
        }
        uint32_t k3 = thing->random_indices_exploitation_LD[(i+7)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        uint32_t k4 = thing->random_indices_exploitation_LD[(i+8)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        uint32_t k5 = thing->random_indices_exploitation_LD[(i+9)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        uint32_t k6 = thing->random_indices_exploitation_LD[(i+10)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
        uint32_t k7 = thing->random_indices_exploitation_LD[(i+11)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
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
        j = thing->neighsLD[thing->neighsLD[i][kmin1]][kmin2];
        if(i!=j){
            uint32_t i_1 = i < j ? i : j;
            uint32_t i_2 = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
            euclsq_ij    = f_euclidean_sq(thing->Xld[i], thing->Xld[j], thing->Mld);
            furthest_d_i = thing->furthest_neighdists_LD[i];
            furthest_d_j = thing->furthest_neighdists_LD[j];
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
            if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                if(attempt_to_add_neighbour(i, j, euclsq_ij, thing)){
                    new_neigh = true;}
            }
            if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                if(attempt_to_add_neighbour(j, i, euclsq_ij, thing)){
                    new_neigh = true;}
            }
            // update the denominator estimation
            dbl_acc_denom += (double) kernel_LD(euclsq_ij, kernel_LD_alpha);
            n_votes++;
        }
        j = thing->neighsLD[thing->neighsLD[i][kmin2]][kmin1];
        if(i!=j && kmin1 != kmin2){
            uint32_t i_1 = i < j ? i : j;
            uint32_t i_2 = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
            euclsq_ij    = f_euclidean_sq(thing->Xld[i], thing->Xld[j], thing->Mld);
            furthest_d_i = thing->furthest_neighdists_LD[i];
            furthest_d_j = thing->furthest_neighdists_LD[j];
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
            if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                if(attempt_to_add_neighbour(i, j, euclsq_ij, thing)){
                    new_neigh = true;}
            }
            if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                if(attempt_to_add_neighbour(j, i, euclsq_ij, thing)){
                    new_neigh = true;}
            }
            // update the denominator estimation
            dbl_acc_denom += (double) kernel_LD(euclsq_ij, kernel_LD_alpha);
            n_votes++;
        }


        // 4: exploitation: neighbour from neighbour list of the other space
        uint32_t khd = thing->random_indices_exploitation_HD[i%NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES];
        j = thing->neighsHD[i][khd];
        if(i!=j){
            uint32_t i_1 = i < j ? i : j;
            uint32_t i_2 = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]); 
            euclsq_ij    = f_euclidean_sq(thing->Xld[i], thing->Xld[j], thing->Mld);
            furthest_d_i = thing->furthest_neighdists_LD[i];
            furthest_d_j = thing->furthest_neighdists_LD[j];
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]); 
            if(euclsq_ij < furthest_d_i){ // if j should be a new neighbour to i
                if(attempt_to_add_neighbour(i, j, euclsq_ij, thing)){
                    new_neigh = true;}
            }
            if(euclsq_ij < furthest_d_j){ // if i should be a new neighbour to j
                if(attempt_to_add_neighbour(j, i, euclsq_ij, thing)){
                    new_neigh = true;}
            }
            // update the denominator estimation
            dbl_acc_denom += (double) kernel_LD(euclsq_ij, kernel_LD_alpha);
            n_votes++;
        }

        // 5: sorting : refine the neighbour ordering stochatically and quickly
        if(new_neigh || ((i%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES) < 1)){
         // first refinement
            uint32_t k_1 = thing->random_indices_exploitation_LD[i%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
            uint32_t k_2 = thing->random_indices_exploitation_LD[(i+3)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
            if(k_1 == k_2){k_2 = (k_2 + 1) % thing->Kld;} // make sure they are different
            float d_1 = f_euclidean_sq(thing->Xld[i], thing->Xld[thing->neighsLD[i][k_1]], thing->Mld);
            float d_2 = f_euclidean_sq(thing->Xld[i], thing->Xld[thing->neighsLD[i][k_2]], thing->Mld);
            if(d_1 < d_2 && k_1 > k_2){ // need to swap these two
                uint32_t j1 = thing->neighsLD[i][k_1];
                uint32_t j2 = thing->neighsLD[i][k_2];
                pthread_mutex_lock(&thing->mutexes_sizeN[i]);
                thing->neighsLD[i][k_1] = j2;
                thing->neighsLD[i][k_2] = j1;
                pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
            }
            // second refinement
            k_1 = thing->random_indices_exploitation_LD[(i+4)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
            k_2 = thing->random_indices_exploitation_LD[(i+5)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
            if(k_1 == k_2){k_2 = (k_2 + 1) % thing->Kld;} // make sure they are different
            d_1 = f_euclidean_sq(thing->Xld[i], thing->Xld[thing->neighsLD[i][k_1]], thing->Mld);
            d_2 = f_euclidean_sq(thing->Xld[i], thing->Xld[thing->neighsLD[i][k_2]], thing->Mld);
            if(d_1 < d_2 && k_1 > k_2){ // need to swap these two
                uint32_t j1 = thing->neighsLD[i][k_1];
                uint32_t j2 = thing->neighsLD[i][k_2];
                pthread_mutex_lock(&thing->mutexes_sizeN[i]);
                thing->neighsLD[i][k_1] = j2;
                thing->neighsLD[i][k_2] = j1;
                pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
            }
            // third refinement
            k_1 = thing->random_indices_exploitation_LD[(i+6)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
            k_2 = thing->random_indices_exploitation_LD[(i+7)%NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES];
            if(k_1 == k_2){k_2 = (k_2 + 1) % thing->Kld;} // make sure they are different
            d_1 = f_euclidean_sq(thing->Xld[i], thing->Xld[thing->neighsLD[i][k_1]], thing->Mld);
            d_2 = f_euclidean_sq(thing->Xld[i], thing->Xld[thing->neighsLD[i][k_2]], thing->Mld);
            if(d_1 < d_2 && k_1 > k_2){ // need to swap these two
                uint32_t j1 = thing->neighsLD[i][k_1];
                uint32_t j2 = thing->neighsLD[i][k_2];
                pthread_mutex_lock(&thing->mutexes_sizeN[i]);
                thing->neighsLD[i][k_1] = j2;
                thing->neighsLD[i][k_2] = j1;
                pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
            }
            
        }
        /* 
        -faire en lock-free pour les reading (expensive)
        verifier que zero sleep*/

        if(new_neigh){n_new_neighs++;}
    }

    // save the Qdenom estimation, the number of new neighbours, and notify the main thread for a new job
    pthread_mutex_lock(thing->thread_mutex);
    thing->N_new_neighs = n_new_neighs;
    thing->estimated_Qdenom = (float) (dbl_acc_denom * ( ((double) (thing->N*thing->N - thing->N)) / (double) n_votes));
    thing->thread_waiting_for_task[0] = true;
    pthread_mutex_unlock(thing->thread_mutex);
    return ;
}


// the subthreads execute this function
void* subroutine_NeighLDDiscoverer(void* arg){
    SubthreadData* thing = (SubthreadData*)arg;
    while(!thing->stop_this_thread){
        pthread_mutex_lock(thing->thread_mutex);
        // subthread is waiting for a task to be assigned: unlock thread mutex and wait 2% of a second
        if(thing->thread_waiting_for_task[0]){ 
            pthread_mutex_unlock(thing->thread_mutex);
            usleep(10000);  // sleep for 1% of a second
        }
        // a task has been assigned to the subthread
        else{ 
            thing->estimated_Qdenom = 0.0f;
            thing->N_new_neighs     = 0;
            pthread_mutex_unlock(thing->thread_mutex);

            // refine neighbours in LD, and estimate the Q denominator
            refine_LD_neighbours(thing);
        }
    }
    // (pour le deadlock juste copier i sur la stack avec lock puis demander le lock pour les j apres)
    return NULL;
}

void* routine_NeighLDDiscoverer(void* arg){
    NeighLDDiscoverer* thing = (NeighLDDiscoverer*)arg;
    thing->isRunning = true;
    // launch subthreads
    for(uint32_t i = 0; i < thing->N_reserved_subthreads; i++){
        if(pthread_create(&thing->subthreads[i], NULL, subroutine_NeighLDDiscoverer, &thing->subthread_data[i]) != 0){
            dying_breath("pthread_create subroutine_NeighLDDiscoverer failed");}
    }
    uint32_t cursor = 0; // the cursor for the start of the next chunk
    while (thing->isRunning) {
        // get the current value of Qdenom, for use locally
        pthread_mutex_lock(thing->mutex_Qdenom);
        float    now_Qdenom = thing->ptr_Qdenom[0];
        pthread_mutex_unlock(thing->mutex_Qdenom);
        // get the current value of N_subthreads_target, for use locally
        pthread_mutex_lock(&thing->mutex_N_subthreads_target);
        uint32_t now_N_subthreads_target = thing->N_subthreads_target;
        pthread_mutex_unlock(&thing->mutex_N_subthreads_target);
        // get the current value of kernel_LD_alpha, for use locally
        pthread_mutex_lock(thing->mutex_kernel_LD_alpha);
        float    now_kernel_LD_alpha = thing->ptr_kernel_LD_alpha[0];
        pthread_mutex_unlock(thing->mutex_kernel_LD_alpha);


        // debug
        // tester ici v 
        // printf("diest furthes for 0 %f\n", thing->furthest_neighdists_LD[0]);

        for(uint32_t i = 0; i < now_N_subthreads_target; i++){
            // if the subthread is waiting for a task: give a new task
            float subthread_estimation_of_denom = -1.0f; 
            pthread_mutex_lock(&thing->subthreads_mutexes[i]);
            if(thing->threads_waiting_for_task[i]){ // subthread is waiting for a task
                // 1.1: temporarily save previous estimated denominator value
                subthread_estimation_of_denom = thing->subthread_data[i].estimated_Qdenom;
                // 1.2: update estimated pct of new neighs
                uint32_t N_new_neighs = thing->subthread_data[i].N_new_neighs;
                float pctage = N_new_neighs > thing->subthreads_chunck_size ? 1.0f : (float)N_new_neighs / (float)thing->subthreads_chunck_size;
                thing->pct_new_neighs = thing->pct_new_neighs * 0.98f + 0.02f * pctage;
                // 2: Assign a new task to the thread
                thing->subthread_data[i].L = cursor;
                thing->subthread_data[i].R = cursor + thing->subthreads_chunck_size > thing->N ? thing->N : cursor + thing->subthreads_chunck_size;
                thing->subthread_data[i].kernel_LD_alpha = now_kernel_LD_alpha;
                // 3: update the cursor in N for the next subthread
                // printf("assingning task to subthread %d, cursor: %d\n", i, cursor);
                cursor += thing->subthreads_chunck_size;
                if(cursor >= thing->N){
                    cursor = 0;
                    N_new_neighs = 0;
                    // print the mean furthest dists for all points in N
                    float mean_dist = 0.0f;
                    for(uint32_t i = 0; i < thing->N; i++){
                        mean_dist += thing->furthest_neighdists_LD[i];
                    }
                    mean_dist /= (float)thing->N;
                    printf("mean furthest dists for all points in N: %f\n", mean_dist);
                }
                thing->threads_waiting_for_task[i] = false;
            }
            pthread_mutex_unlock(&thing->subthreads_mutexes[i]);
            // if a subthread has been detected as finished, update the global Qdenom value
            if(subthread_estimation_of_denom > 0.0f){
                // float contribution = (1.0 - ALPHA_QDENOM) * subthread_estimation_of_denom;
                pthread_mutex_lock(thing->mutex_Qdenom);
                // bug previously: the alpha and complement didnt sum to 1: forgot to ad "f" to EPSILON defintion
                thing->ptr_Qdenom[0] = thing->ptr_Qdenom[0]*ALPHA_QDENOM  + (1.0f - ALPHA_QDENOM) * subthread_estimation_of_denom;
                pthread_mutex_unlock(thing->mutex_Qdenom);
            }
        } 
        usleep(10000); // 1% of a second, prevent the thread from taking too much CPU time
    }
    dying_breath("routine_NeighLDDiscoverer ended");
    return NULL;
}

void start_thread_NeighLDDiscoverer(NeighLDDiscoverer* thing){
    if(pthread_create(&thing->thread, NULL, routine_NeighLDDiscoverer, thing) != 0){
        dying_breath("pthread_create routine_NeighLDDiscoverer failed");}
}

