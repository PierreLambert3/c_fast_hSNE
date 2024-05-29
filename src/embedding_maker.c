#include "embedding_maker.h"


void new_EmbeddingMaker_CPU(EmbeddingMaker_CPU* thing, uint32_t N, uint32_t Mld, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t Kld, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD){
    dying_breath("CPU-based embedding maker not implemented yet");
}

void new_EmbeddingMaker_GPU(EmbeddingMaker_GPU* thing, uint32_t N, uint32_t Mld, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t Kld, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD){
    thing->mutex_thread = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
    if(pthread_mutex_init(thing->mutex_thread, NULL) != 0){
        dying_breath("pthread_mutex_init mutex_thread failed");}
    thing->rand_state = thread_rand_seed[0];
    thing->running = false;
    thing->work_type = 0;
    thing->N = N;
    thing->mutexes_sizeN = mutexes_sizeN;
    thing->Qdenom = 0.0f;
    thing->hparam_LDkernel_alpha       = malloc_float(1, 1.0f);
    thing->mutex_hparam_LDkernel_alpha = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
    if(pthread_mutex_init(thing->mutex_hparam_LDkernel_alpha, NULL) != 0){
        dying_breath("pthread_mutex_init mutex_hparam_LDkernel_alpha failed");}
    thing->Xld_true = Xld;
    thing->Xld_nesterov = malloc_float_matrix(N, Mld, 0.0f);
    memcpy(thing->Xld_nesterov[N], Xld[N], N*Mld*sizeof(float));

    faire 2 fonctions utilitaires : 
        un memcpy pour flaot matrix qui fait la ligne ci dessus ,
        un contiguous_floats(float** matrice )  --> out est une float*   (c est matrix[N])

    for(uint32_t n = 0; n < N; n++){
        for(uint32_t m = 0; m < Mld; m++){
            float abs_error = fabs(Xld[n][m] - thing->Xld_nesterov[n][m]);
            printf("err   %f\n", abs_error);
        }
    }
    die();
    
}

// depending on the (user-determined) use of GPU vs CPU, this initialises the appropriate struct
void new_EmbeddingMaker(EmbeddingMaker* thing, uint32_t N, uint32_t Mld, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t Kld, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD){
    thing->maker_cpu = NULL;
    thing->maker_gpu = NULL;
    if(USE_GPU){
        thing->maker_gpu = (EmbeddingMaker_GPU*) malloc(sizeof(EmbeddingMaker_GPU));
        new_EmbeddingMaker_GPU(thing->maker_gpu, N, Mld, thread_rand_seed, mutexes_sizeN,\
            Xld, Khd, Kld, neighsLD, neighsHD, furthest_neighdists_LD);
    } else {
        thing->maker_cpu = (EmbeddingMaker_CPU*) malloc(sizeof(EmbeddingMaker_CPU));
        new_EmbeddingMaker_CPU(thing->maker_cpu, N, Mld, thread_rand_seed, mutexes_sizeN,\
            Xld, Khd, Kld, neighsLD, neighsHD, furthest_neighdists_LD);
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
    if(thing->maker_cpu != NULL){
        free(thing->maker_cpu);
    }
    if(thing->maker_gpu != NULL){
        free(thing->maker_gpu->hparam_LDkernel_alpha);
        free(thing->maker_gpu);
    }
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