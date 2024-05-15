#include "embedding_maker.h"

EmbeddingMaker* new_EmbeddingMaker(uint32_t _N_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads){
    EmbeddingMaker* thing = (EmbeddingMaker*)malloc(sizeof(EmbeddingMaker));
    thing->N_reserved_subthreads = max_nb_of_subthreads;
    thing->N_subthreads_now      = 0;
    thing->N_subthreads_target   = max_nb_of_subthreads;
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
    thing->rand_state = (uint32_t)time(NULL) +  thread_rand_seed[0]++;
    thing->passes_since_reset = 0;
    thing->p_wakeup = 1.0f;
    thing->N = _N_;
    printf("%d rand state\n", thing->rand_state);
    return thing;
}

/*
IMPORTANT:
when computing the Qij part of gradients: define them such that the Qij_denom
is dividing the sum after the loops (since it's a value shaed for all points)
*/

void destroy_EmbeddingMaker(EmbeddingMaker* thing){
    free(thing);
}

void* routine_EmbeddingMaker(void* arg){
    EmbeddingMaker* thing = (EmbeddingMaker*)arg;
    thing->isRunning = true;
    while (thing->isRunning) {
        //check random value
        uint32_t random_value = rand_uint32_between(&thing->rand_state, 0, 10);
        printf("%d                                              embedding\n", random_value);
        sleep(1);
        if(random_value < 1) {
            thing->isRunning = false;}
    }
    return NULL;
}

void start_thread_EmbeddingMaker(EmbeddingMaker* thing){
    if(pthread_create(&thing->thread, NULL, routine_EmbeddingMaker, thing) != 0){
        dying_breath("pthread_create routine_EmbeddingMaker failed");}
}