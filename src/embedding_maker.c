#include "embedding_maker.h"


void new_EmbeddingMaker_CPU(EmbeddingMaker_CPU* thing, uint32_t N, uint32_t* thread_rand_seed){
    dying_breath("CPU-based embedding maker not implemented yet");
}

void new_EmbeddingMaker_GPU(EmbeddingMaker_GPU* thing, uint32_t N, uint32_t* thread_rand_seed){
    thing->rand_state = *thread_rand_seed;
    printf("OK good\n");
}

void new_EmbeddingMaker(EmbeddingMaker* thing, uint32_t N, uint32_t* thread_rand_seed){

    for(uint32_t i = 0; i < 1000; i++){
        printf("d abord: faire que ls neigh_disco ont le GPU dependant de leur part de pourcentage decouvert (100 et 100 : 0.5et0.5, 100 et 0 : 1 et 0, 0 et 100 : 0 et 1)\n");
    }

    thing->maker_cpu = NULL;
    thing->maker_gpu = NULL;
    if(USE_GPU){
        thing->maker_gpu = (EmbeddingMaker_GPU*) malloc(sizeof(EmbeddingMaker_GPU));
        new_EmbeddingMaker_GPU(thing->maker_gpu, N, thread_rand_seed);
    } else {
        thing->maker_cpu = (EmbeddingMaker_CPU*) malloc(sizeof(EmbeddingMaker_CPU));
        new_EmbeddingMaker_CPU(thing->maker_cpu, N, thread_rand_seed);
    }
}

/*
IMPORTANT:
when computing the Qij part of gradients: define them such that the Qij_denom
is dividing the sum after the loops (since it's a value shaed for all points)
*/

/*
sous-poudrer le tout avec des gradients de MDS
*/

// thing->estimated_Qdenom = (float) (dbl_acc_denom * ( ((double) (thing->N*thing->N - thing->N)) / (double) n_votes));
// thing->ptr_Qdenom[0] = thing->ptr_Qdenom[0]*ALPHA_QDENOM  + (1.0f - ALPHA_QDENOM) * subthread_estimation_of_denom;

// printf("it is important to update the furhtest dist to LD neighs in the tSNE optimisation, when computing them\n");
// printf("it is important to update the furhtest dist to LD neighs in the tSNE optimisation, when computing them\n");
// printf("it is important to update the furhtest dist to LD neighs in the tSNE optimisation, when computing them\n");
// printf("it is important to update the furhtest dist to LD neighs in the tSNE optimisation, when computing them\n");

void destroy_EmbeddingMaker(EmbeddingMaker* thing){
    free(thing);
}

void* routine_EmbeddingMaker_CPU(void* arg){
    dying_breath("CPU-based embedding maker not implemented yet");
    // printf("it is important to update the furhtest dist to LD neighs in the tSNE optimisation, when computing them\n");
    return NULL; 
}

void* routine_EmbeddingMaker_GPU(void* arg){
    // printf("it is important to update the furhtest dist to LD neighs in the tSNE optimisation, when computing them\n");
    return NULL; 
}


void start_thread_EmbeddingMaker(EmbeddingMaker* thing){
    if(USE_GPU){
        if(pthread_create(&thing->thread, NULL, routine_EmbeddingMaker_GPU, thing) != 0){
            dying_breath("pthread_create routine_EmbeddingMaker failed");}
    }
    else {
        dying_breath("CPU-based embedding maker not implemented yet");
        if(pthread_create(&thing->thread, NULL, routine_EmbeddingMaker_CPU, thing) != 0){
            dying_breath("pthread_create routine_EmbeddingMaker failed");}
    }
    printf("TODO : understand CUDA streams! \n");
}