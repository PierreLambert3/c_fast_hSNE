#include "neighHD_discoverer.h"

void new_NeighHDDiscoverer(NeighHDDiscoverer* thing, uint32_t _N_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads,\
        pthread_mutex_t* mutexes_sizeN, float** _Xhd_, uint32_t _Khd_, uint32_t _Kld_, uint32_t** _neighsHD_, uint32_t** _neighsLD_,\
        float* furthest_neighdists_HD, float** _Psym_GT_){
    // worker and subthread management
    thing->isRunning = false;
    thing->rand_state = (uint32_t)time(NULL) + ++thread_rand_seed[0];
    thing->N_reserved_subthreads = max_nb_of_subthreads;
    thing->N_subthreads_target   = max_nb_of_subthreads;
    thing->subthreads_chunck_size = 1 + (uint32_t)floorf(SUBTHREADS_CHUNK_SIZE_PCT * (float)_N_);
    printf("subthreads chunck size %d\n", thing->subthreads_chunck_size);
    dying_breath("neighHD_discoverer");
    thing->subthreads = (pthread_t*)malloc(sizeof(pthread_t) * max_nb_of_subthreads);
    thing->subthreads_mutexes = mutexes_allocate_and_init(max_nb_of_subthreads);
    thing->threads_waiting_for_task = malloc_bool(max_nb_of_subthreads, true);
    pthread_mutex_init(&thing->mutex_N_subthreads_target, NULL);

    // initialise algorithm data on this thread
    subthreadHD_data = (SubthreadHD_data*)malloc(sizeof(SubthreadHD_data) * max_nb_of_subthreads);
    thing->N = _N_;

    les P_asym sont malloc ici (seulement les Psym sont shared)

    Pasy          = malloc_float_matrix(N, Khd, 1.0f);
    Pasy_sumJ_Pij = malloc_float(N, 1.0f);
    Psym_needs_update = malloc_bool(N, true);

    // initialize subthread internals
    for(uint32_t i = 0; i < max_nb_of_subthreads; i++){

    }
    
    
    /* thing->N_reserved_subthreads = max_nb_of_subthreads;
    thing->N_subthreads_now      = 0;
    thing->N_subthreads_target   = 1;
    // initialize subthreads
    thing->subthreads = (pthread_t*)malloc(sizeof(pthread_t) * max_nb_of_subthreads);
    thing->subthreads_cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t) * max_nb_of_subthreads);
    thing->subthreads_rwlock = (pthread_rwlock_t*)malloc(sizeof(pthread_rwlock_t) * max_nb_of_subthreads);
    for(uint32_t i = 0; i < max_nb_of_subthreads; i++){
        pthread_cond_init(&thing->subthreads_cond[i], NULL);
        pthread_rwlock_init(&thing->subthreads_rwlock[i], NULL);
    }
    thing->subthreads_chunck_size = 1000;
    thing->isRunning = false;
    thing->rand_state = (uint32_t)time(NULL) + thread_rand_seed[0]++;
    thing->passes_since_reset = 0;
    thing->p_wakeup = 1.0f;
    thing->N = _N_;
    thing->test_variable = malloc_uint32_t(10, 0);
    printf("%d rand state\n", thing->rand_state); */
}

void destroy_NeighHDDiscoverer(NeighHDDiscoverer* thing) {
    free_array(thing->test_variable);
    free(thing);
}

void* routine_NeighHDDiscoverer(void* arg) {
    NeighHDDiscoverer* thing = (NeighHDDiscoverer*)arg;
    thing->isRunning = true;
    while (thing->isRunning) {
        //check random value
        uint32_t random_value = rand_uint32_between(&thing->rand_state, 0, 10);
        printf("%d                   neigh\n", random_value);
        sleep(1);
        if(random_value < 1) {
            thing->isRunning = false;}
    }
    return NULL;
}

void start_thread_NeighHDDiscoverer(NeighHDDiscoverer* thing) {
    if(pthread_create(&thing->thread, NULL, routine_NeighHDDiscoverer, thing) != 0){
        dying_breath("pthread_create routine_NeighHDDiscoverer failed");}
}
