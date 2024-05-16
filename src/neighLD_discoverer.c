#include "neighLD_discoverer.h"

void new_NeighLDDiscoverer(NeighLDDiscoverer* thing, uint32_t _N_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads,\
        pthread_mutex_t* _mutex_Qdenom_, pthread_mutex_t* mutexes_sizeN, float** _Xld_, float** _Xhd_, uint32_t _Mld_, uint32_t _Khd_,\
        uint32_t _Kld_, uint32_t** _neighsLD_, uint32_t** _neighsHD_,\
        float* furthest_neighdists_LD, float** _Q_, float* _Qdenom_){

    // work-management data
    thing->isRunning = false;
    thing->rand_state = (uint32_t)time(NULL) +  ++thread_rand_seed[0];
    thing->N_reserved_subthreads  = max_nb_of_subthreads;
    thing->N_subthreads_target    = max_nb_of_subthreads;
    thing->subthreads_chunck_size = 1 + floorf(SUBTHREADS_CHUNK_SIZE_PCT * (float)_N_);
    thing->subthreads             = (pthread_t*)malloc(sizeof(pthread_t) * max_nb_of_subthreads);
    thing->threads_waiting_for_task = bool_array_initval(max_nb_of_subthreads, true);
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
    thing->Q = _Q_;
    thing->ptr_Qdenom = _Qdenom_;
    thing->mutex_Qdenom  = _mutex_Qdenom_;
    thing->mutexes_sizeN = mutexes_sizeN;

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
        // lock with mutex_Qdenom
        pthread_mutex_lock(thing->mutex_Qdenom);
        subthread_data->estimated_Qdenom = thing->ptr_Qdenom[0];
        pthread_mutex_unlock(thing->mutex_Qdenom);
        subthread_data->mutexes_sizeN = mutexes_sizeN;
        subthread_data->thread_mutex = &thing->subthreads_mutexes[i];
        subthread_data->thread_waiting_for_task = &thing->threads_waiting_for_task[i];
        subthread_data->random_indices_exploration  = uint32_t_array_initval(NEIGH_FAR_EXPLORATION_N_SAMPLES, 0);
        subthread_data->random_indices_exploitation_LD = uint32_t_array_initval(NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES, 0);
        subthread_data->random_indices_exploitation_HD = uint32_t_array_initval(NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES, 0);
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

double  refine_LD_neighbours(SubthreadData* thing){
    // first, generate random uint32_T for exploration and exploitation
    // between 0 and N
    for(uint32_t i = 0; i < NEIGH_FAR_EXPLORATION_N_SAMPLES; i++){
        thing->random_indices_exploration[i] = rand_uint32_between(&thing->rand_state, 0, thing->N);
    }
    // between 0 and Kld
    for(uint32_t i = 0; i < NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES; i++){
        thing->random_indices_exploitation_LD[i] = rand_uint32_between(&thing->rand_state, 0, thing->Kld);
    }
    // between 0 and Khd
    for(uint32_t i = 0; i < NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES; i++){
        thing->random_indices_exploitation_HD[i] = rand_uint32_between(&thing->rand_state, 0, thing->Khd);
    }
    // variables for the denominator estimation
    double   dbl_acc_denom = 0.;
    uint32_t n_votes       = 0;
    // temp variables filled when mutex are acquired
    float    euclsq_ij    = 0.0f;
    float    furthest_d_i = 1.0f;
    float    furthest_d_j = 1.0f;
    for(uint32_t i = thing->L; i < thing->R; i++){
        // -------------------  TODO  ---------------------------
        // clever algorithm with 3 or more point: i, j, and r. compute dists dir and drj.
        // We should be able to tell if j and i are also candidate based on the 2 dists ands the radius of i and j
        // ------------------------------------------------------

        

        // 2: exploration: random point j in [0, N[
        uint32_t j = thing->random_indices_exploration[i%NEIGH_FAR_EXPLORATION_N_SAMPLES];
        uint32_t i_1    = i < j ? i : j;
        uint32_t i_2    = i < j ? j : i;
        pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
        pthread_mutex_lock(&thing->mutexes_sizeN[i_2]);
        euclsq_ij      = f_euclidean_sq(thing->Xld[i], thing->Xld[j], thing->Mld);
        furthest_d_i = thing->furthest_neighdists_LD[i];
        furthest_d_j = thing->furthest_neighdists_LD[j];
        pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
        pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]);
        if(euclsq_ij < furthest_d_i)){
            attempt_to_insert_neighbour(i, j, euclsq_ij, thing->neighsLD, thing->furthest_neighdists_LD, thing->Kld);
        }
        id(euclsq_ij < furthest_d_j){
            attempt_to_insert_neighbour(j, i, euclsq_ij, thing->neighsLD, thing->furthest_neighdists_LD, thing->Kld);
        }

        todo dans attempt:
            considerer qu on doit l insérer 
            a la tout fin: prendre le lock e re-check le furthest et enfin insert 
            cf git quand meme et checker i different de j dans la fonction pour etre safe

        /* template for a neighbour candidate point j towards i : 
            uint32_t j      = <something>;
            float   eucl_ij = 0.0f;
            uint32_t i_1    = i < j ? i : j;
            uint32_t i_2    = i < j ? j : i;
            pthread_mutex_lock(&thing->mutexes_sizeN[i_1]);
            pthread_mutex_lock(&thing->mutexes_sizeN[i_2]);
            eucl_ij = f_euclidean(thing->Xld[i], thing->Xld[j], thing->Mld);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_2]);
            pthread_mutex_unlock(&thing->mutexes_sizeN[i_1]);
        */
        
    }
    die();
    return (dbl_acc_denom * ( ((double) (thing->N*thing->N - thing->N)) / (double) n_votes));
}


// the subthreads execute this function
void* subroutine_NeighLDDiscoverer(void* arg){
    SubthreadData* thing = (SubthreadData*)arg;
    while(!thing->stop_this_thread){
        pthread_mutex_lock(thing->thread_mutex);
        // subthread is waiting for a task to be assigned: unlock thread mutex and wait 2% of a second
        if(thing->thread_waiting_for_task[0]){ 
            pthread_mutex_unlock(thing->thread_mutex);
            // sleep for 1% of a second
            usleep(10000); 
        }
        // a task has been assigned to the subthread
        else{ 
            thing->estimated_Qdenom = 0.0f;
            pthread_mutex_unlock(thing->thread_mutex);
            // refine neighbours in LD, and estimate the Q denominator
            double double_denom_estimation = refine_LD_neighbours(thing);
            // save the estimation, and notify the main thread for a new job
            pthread_mutex_lock(thing->thread_mutex);
            thing->estimated_Qdenom = (float) double_denom_estimation;
            thing->thread_waiting_for_task[0] = true;
            pthread_mutex_unlock(thing->thread_mutex);

            /* // simulate computations
            for(uint32_t rep = 0; rep < 100; rep++){
                double random_double = (double) rand_float_between(&data->rand_state, 0.0f, 1.0f);
                dbl_acc_denom += random_double;
                n_votes++;
            } */
            //
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

        for(uint32_t i = 0; i < now_N_subthreads_target; i++){
            // if the subthread is waiting for a task: give a new task
            float subthread_estimation_of_denom = -1.0f; 
            pthread_mutex_lock(&thing->subthreads_mutexes[i]);
            if(thing->threads_waiting_for_task[i]){ // subthread is waiting for a task
                // 1: temporarily save previous estimated denominator value
                subthread_estimation_of_denom = thing->subthread_data[i].estimated_Qdenom;
                // 2: Assign a new task to the thread
                thing->subthread_data[i].L = cursor;
                thing->subthread_data[i].R = cursor + thing->subthreads_chunck_size > thing->N ? thing->N : cursor + thing->subthreads_chunck_size;
                // 3: update the cursor in N for the next subthread
                cursor += thing->subthreads_chunck_size;
                if(cursor >= thing->N){
                    cursor = 0;}

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

        // artificially stop the thread, for now
        float random_number = rand_float_between(&thing->rand_state, 0.0f, 1.0f);
        if(random_number < 0.002f){
            thing->isRunning = false;
            for(uint32_t i = 0; i < now_N_subthreads_target; i++){
                thing->subthread_data[i].stop_this_thread = true;}
        }
    }


    return NULL;
}

void start_thread_NeighLDDiscoverer(NeighLDDiscoverer* thing){
    if(pthread_create(&thing->thread, NULL, routine_NeighLDDiscoverer, thing) != 0){
        dying_breath("pthread_create routine_NeighLDDiscoverer failed");}
}

