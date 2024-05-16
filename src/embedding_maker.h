#ifndef EMBEDDING_MAKER_H
#define EMBEDDING_MAKER_H

#include <pthread.h>
#include "system.h"
#include "probabilities.h"

typedef struct {
    pthread_t thread;
    uint32_t N_reserved_subthreads;
    uint32_t N_subthreads_now;
    uint32_t N_subthreads_target;
    pthread_t* subthreads;
    bool     isRunning;
    uint32_t rand_state;
    uint32_t passes_since_reset;
    float    p_wakeup;
    uint32_t N;
} EmbeddingMaker;

void  new_EmbeddingMaker(EmbeddingMaker* thing, uint32_t _N_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads);
void  destroy_EmbeddingMaker(EmbeddingMaker* thing);
void* routine_EmbeddingMaker(void* arg);
void  start_thread_EmbeddingMaker(EmbeddingMaker* thing);

#endif // EMBEDDING_MAKER_H