#include "neighHD_discoverer.h"

void new_NeighHDDiscoverer(NeighHDDiscoverer* thing, uint32_t _N_, uint32_t _M_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads,\
        pthread_mutex_t* mutexes_sizeN, float** _Xhd_, uint32_t _Khd_, uint32_t _Kld_, uint32_t** _neighsHD_, uint32_t** _neighsLD_,\
        float* furthest_neighdists_HD, float** _Psym_GT_){
    // worker and subthread management
    thing->isRunning = false;
    thing->rand_state = (uint32_t)time(NULL) + ++thread_rand_seed[0];
    thing->N_reserved_subthreads = max_nb_of_subthreads;
    thing->N_subthreads_target   = max_nb_of_subthreads;
    thing->subthreads_chunck_size = 1 + (uint32_t)floorf(SUBTHREADS_CHUNK_SIZE_PCT * (float)_N_);
    printf("subthreads chunck size %d\n", thing->subthreads_chunck_size);
    thing->subthreads = (pthread_t*)malloc(sizeof(pthread_t) * max_nb_of_subthreads);
    thing->subthreads_mutexes = mutexes_allocate_and_init(max_nb_of_subthreads);
    thing->threads_waiting_for_task = malloc_bool(max_nb_of_subthreads, true);
    pthread_mutex_init(&thing->mutex_N_subthreads_target, NULL);

    // initialise algorithm data on this thread
    thing->N = _N_;
    thing->Mhd = _M_;
    thing->Xhd = _Xhd_;
    thing->Khd = _Khd_;
    thing->Kld = _Kld_;
    thing->neighsHD = _neighsHD_;
    thing->neighsLD = _neighsLD_;
    thing->furthest_neighdists_HD = furthest_neighdists_HD;
    thing->Pasym = malloc_float_matrix(_N_, _Khd_, 1.0f);
    thing->Pasym_sumJ_Pij = malloc_float(_N_, 1.0f);
    thing->flag_neigh_update = malloc_bool(_N_, false);
    thing->Psym = _Psym_GT_;
    thing->radii = malloc_float(_N_, 1.0f);
    thing->pct_new_neighs = 1.0f;
    thing->mutexes_sizeN = mutexes_sizeN;
    thing->subthreadHD_data = (SubthreadHD_data*)malloc(sizeof(SubthreadHD_data) * max_nb_of_subthreads);
    for(uint32_t i = 0; i < max_nb_of_subthreads; i++){
        SubthreadHD_data* subthread_data = &thing->subthreadHD_data[i];
        subthread_data->stop_this_thread = false;
        subthread_data->N = _N_;
        subthread_data->rand_state = ++thread_rand_seed[0];
        subthread_data->L = 0;
        subthread_data->R = 0;
        subthread_data->N_new_neighs = 0;
        subthread_data->Xhd = _Xhd_;
        subthread_data->Mhd = _M_;
        subthread_data->Khd = _Khd_;
        subthread_data->Kld = _Kld_;
        subthread_data->neighsHD = _neighsHD_;
        subthread_data->neighsLD = _neighsLD_;
        subthread_data->furthest_neighdists_HD = furthest_neighdists_HD;
        subthread_data->Pasym = thing->Pasym;
        subthread_data->Pasym_sumJ_Pij = thing->Pasym_sumJ_Pij;
        subthread_data->flag_neigh_update = thing->flag_neigh_update;
        subthread_data->Psym = thing->Psym;
        subthread_data->radii = thing->radii;
        subthread_data->thread_waiting_for_task = &thing->threads_waiting_for_task[i];
        subthread_data->mutexes_sizeN = mutexes_sizeN;
        subthread_data->thread_mutex = &thing->subthreads_mutexes[i];
        subthread_data->random_indices_exploration = malloc_uint32_t(NEIGH_FAR_EXPLORATION_N_SAMPLES, 0);
        subthread_data->random_indices_exploitation_LD = malloc_uint32_t(NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES, 0);
        subthread_data->random_indices_exploitation_HD = malloc_uint32_t(NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES, 0);
    }
}

void destroy_NeighHDDiscoverer(NeighHDDiscoverer* thing) {
    for(uint32_t i = 0; i < thing->N_reserved_subthreads; i++){
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

void refine_HD_interactions(SubthreadHD_data* thing){
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
    // 1/  find new neighbours 
    // 2/  if i gets new neigh j :  flag of i is set (if j gets i: flag of j is set)
    // 3/  if flag of i:  i recomputes Pasym
    uint32_t n_new_neighs  = 0;

    // save the number of new neighbours and notify the thread that it is waiting for a task
    pthread_mutex_lock(thing->thread_mutex);
    thing->N_new_neighs = n_new_neighs;
    thing->thread_waiting_for_task[0] = true;
    pthread_mutex_unlock(thing->thread_mutex);
    return;
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
            thing->N_new_neighs = 0;
            pthread_mutex_unlock(thing->thread_mutex);
            // do the task: refine HD neighbours, update radii and Pasym and Psym
            refine_HD_interactions(thing);
        }
    }
    return NULL;
}


void* routine_NeighHDDiscoverer(void* arg) {
    NeighHDDiscoverer* thing = (NeighHDDiscoverer*)arg;
    thing->isRunning = true;
    // launch subthreads
    for(uint32_t i = 0; i < thing->N_reserved_subthreads; i++){
        if(pthread_create(&thing->subthreads[i], NULL, subroutine_NeighHDDiscoverer, &thing->subthreadHD_data[i]) != 0){
            dying_breath("pthread_create routine_NeighHDDiscoverer_subthread failed");}
    }
    // work dispatcher loop
    uint32_t cursor = 0;
    bool     working_on_Psym = false
    while(thing->isRunning){

        

        // get the current value of N_subthreads_target, for use locally
        pthread_mutex_lock(&thing->mutex_N_subthreads_target);
        uint32_t now_N_subthreads_target = thing->N_subthreads_target;
        pthread_mutex_unlock(&thing->mutex_N_subthreads_target);
        for(uint32_t i = 0; i < now_N_subthreads_target; i++){
            // if the subthread is waiting for a task: give a new task
            pthread_mutex_lock(&thing->subthreads_mutexes[i]);
            if(thing->threads_waiting_for_task[i]){
                SubthreadHD_data* subthread_data = &thing->subthreadHD_data[i];
                // 1.1: update estimated pct of new neighs
                if(subthread_data->R - subthread_data->L == thing->subthreads_chunck_size){
                    uint32_t N_new_neighs = subthread_data->N_new_neighs;
                    float pctage = N_new_neighs > thing->subthreads_chunck_size ? 1.0f : (float)N_new_neighs / (float)thing->subthreads_chunck_size;
                    thing->pct_new_neighs = thing->pct_new_neighs * 0.98f + 0.02f * pctage;
                }
                // 2: assign new task to the thread
                subthread_data->L = cursor;
                subthread_data->R = cursor + thing->subthreads_chunck_size > thing->N ? thing->N : cursor + thing->subthreads_chunck_size;
                thing->threads_waiting_for_task[i] = false;
                // 3: update the cursor in N for the next subthread
                printf("assingning task to subthread %d, cursor: %d\n", i, cursor);
                cursor += thing->subthreads_chunck_size;
                if(cursor >= thing->N){
                    cursor = 0;


                    every 4 passes, update Psym_GT (set working_on_Psym to true)
                    obligé car changer un neigh sur i va influcencer ses asym 
                    et donc aussi ses Pij avec tous les j, meme les acience neighbours
                    donc obligé de tut recompute de temps en temps et de garder des calculs simples (sur i et ses nouveau voisins) pour la plupart des iterations


                    // print the mean furthest dists for all points in N
                    float mean_dist = 0.0f;
                    for(uint32_t i = 0; i < thing->N; i++){
                        mean_dist += thing->furthest_neighdists_HD[i];
                    }
                    mean_dist /= (float)thing->N;
                    printf("mean furthest dists for all points in N: %f  (HD)\n", mean_dist);
                }
            }
            pthread_mutex_unlock(&thing->subthreads_mutexes[i]);
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
