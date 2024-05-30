#include "embedding_maker.h"


void new_EmbeddingMaker_CPU(EmbeddingMaker_CPU* thing, uint32_t N, uint32_t Mld, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t Kld, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P){
    dying_breath("CPU-based embedding maker not implemented yet");
}

void new_EmbeddingMaker_GPU(EmbeddingMaker_GPU* thing, uint32_t N, uint32_t Mld, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
        float** Xld, uint32_t Khd, uint32_t Kld, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
        float** P, pthread_mutex_t* mutex_P){
    thing->mutex_thread = mutex_allocate_and_init();
    thing->rand_state = ++thread_rand_seed[0];
    thing->is_running = false;
    thing->work_type = 0;
    thing->N = N;
    thing->mutexes_sizeN = mutexes_sizeN;
    thing->hparam_LDkernel_alpha       = malloc_float(1, 1.0f);
    thing->mutex_hparam_LDkernel_alpha = mutex_allocate_and_init();
    thing->Xld_cpu = Xld;
    thing->neighsLD_cpu = neighsLD;
    thing->neighsHD_cpu = neighsHD;
    thing->furthest_neighdists_LD_cpu = furthest_neighdists_LD;
    thing->P_cpu = P;
    thing->mutex_P = mutex_P;
    // things on GPU
    thing->Xld_base_cuda = malloc_float(N*Mld, 0.0f);
    memcpy(thing->Xld_base_cuda, as_float_1d(thing->Xld_cpu, N, Mld), N*Mld*sizeof(float));
    thing->Xld_nesterov_cuda = malloc_float(N*Mld, 0.0f);
    memcpy(thing->Xld_nesterov_cuda, as_float_1d(thing->Xld_cpu, N, Mld), N*Mld*sizeof(float));
    thing->momenta_attraction_cuda = malloc_float(N*Mld, 0.0f);
    thing->momenta_repulsion_far_cuda = malloc_float(N*Mld, 0.0f);
    thing->momenta_repulsion_cuda = malloc_float(N*Mld, 0.0f);
    thing->neighsLD_cuda = malloc_uint32_t(N*Kld, 0u);
    memcpy(thing->neighsLD_cuda, as_uint32_1d(thing->neighsLD_cpu, N, Kld), N*Kld*sizeof(uint32_t));
    thing->neighsHD_cuda = malloc_uint32_t(N*Khd, 0u);
    memcpy(thing->neighsHD_cuda, as_uint32_1d(thing->neighsHD_cpu, N, Khd), N*Khd*sizeof(uint32_t));
    thing->furthest_neighdists_LD_cuda = malloc_float(N, 0.0f);
    memcpy(thing->furthest_neighdists_LD_cuda, furthest_neighdists_LD, N*sizeof(float));
    thing->P_cuda = malloc_float(N*Khd, 0.0f);
    memcpy(thing->P_cuda, as_float_1d(thing->P_cpu, N, Khd), N*Khd*sizeof(float));
    thing->Qdenom_cuda = 1.0f;
}


/*
- neighsLD_cpu -> neighsLD_cuda
- neighsHD_cpu -> neighsHD_cuda
- P_cpu -> P_cuda
*/
void safely_sync_with_CPU(EmbeddingMaker_GPU* thing){
    prendre les mutex les uns apres les autres puis un gros memcpy puis unlock les uns apres lesautres?
    for(uint32_t i=0; i<thing->N; i++){
        pthread_mutex_lock(&thing->mutexes_sizeN[i]);
        
        continuer ici

        pthread_mutex_unlock(&thing->mutexes_sizeN[i]);
    }
}


solution truovées avec buffer et deux flags, cf le papier.
solution truovées avec buffer et deux flags, cf le papier.
solution truovées avec buffer et deux flags, cf le papier.
solution truovées avec buffer et deux flags, cf le papier.
/***
 *    _________     _______  _        _______                 _______           ______   _______ 
 *    \__   __/    (  ____ \( (    /|(  ____ \               (  ____ \|\     /|(  __  \ (  ___  )
 *       ) (       | (    \/|  \  ( || (    \/ _             | (    \/| )   ( || (  \  )| (   ) |
 *       | | _____ | (_____ |   \ | || (__    (_)            | |      | |   | || |   ) || (___) |
 *       | |(_____)(_____  )| (\ \) ||  __)                  | |      | |   | || |   | ||  ___  |
 *       | |             ) || | \   || (       _             | |      | |   | || |   ) || (   ) |
 *       | |       /\____) || )  \  || (____/\(_)            | (____/\| (___) || (__/  )| )   ( |
 *       )_(       \_______)|/    )_)(_______/               (_______/(_______)(______/ |/     \|
 *                                                                                               
 */

/*
This function performs the gradient-descent part of t-SNE, using the neighbour sets (LD and HD) and the P matrix that are continuously updated by other threads in parallel.
This thread does its heavy-filting on the GPU using CUDA. The other threads don't use CUDA: this thread peridically writes and reads CPU-based 
variables to ensure communication between all threads.

Description of the periodic exchanges with other threads:
- XLD_CPU is copied from GPU to CPU at each iteration, in an UNSAFE manner.
- furthest_neighdists_LD_cuda is updated here at each iteration, and copied to furthest_neighdists_LD_cpu at each iteration.
   The exchange is done in an UNSAFE manner, for speed.
- neighsLD_cuda is updated here every 0.5 seconds, by copying neighsLD_cpu to neighsLD_cuda. 
   The exchange is done in a SAFE manner using mutexes_sizeN
- neighsHD_cuda is updated here every 0.5 seconds, by copying neighsHD_cpu to neighsHD_cuda. 
   The exchange is done in a SAFE manner using mutexes_sizeN
- P on the GPU is updated from the CPU every 0.5seconds, in a SAFE manner.
*/
void* routine_EmbeddingMaker_GPU(void* arg){
    EmbeddingMaker_GPU* thing = (EmbeddingMaker_GPU*) arg;
    clock_t start_time, current_time;
    start_time = clock();
    while(thing->is_running){
        // gradient descent: fill momenta_attraction, momenta_repulsion_far, momenta_repulsion
        // ...

        // momentum leak: momenta_repulsion_far gets smoothed across neighbours (with conservation of vector norm)
        // ...

        // apply momenta to Xld, regenerate Xld_nesterov, decay momenta
        // ...

        // in an UNSAFE manner, update Xld_cpu and furthest_neighdists_LD_cpu
        // ...

        // every 0.5 seconds, update neighsLD_cuda and neighsHD_cuda in a SAFE manner
        current_time = clock();
        double time_elapsed = ((double) (current_time - start_time)) / CLOCKS_PER_SEC;
        if(time_elapsed > GUI_CPU_SYNC_PERIOD){
            safely_sync_with_CPU(thing);
            start_time = clock();
        }
        
        // printf("it is important to update the furhtest dist to LD neighs in the tSNE optimisation, when computing them\n");
    }
    return NULL; 
}

// depending on the (user-determined) use of GPU vs CPU, this initialises the appropriate struct
void new_EmbeddingMaker(EmbeddingMaker* thing, uint32_t N, uint32_t Mld, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t Kld, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P){
    thing->maker_cpu = NULL;
    thing->maker_gpu = NULL;
    if(USE_GPU){
        thing->maker_gpu = (EmbeddingMaker_GPU*) malloc(sizeof(EmbeddingMaker_GPU));
        new_EmbeddingMaker_GPU(thing->maker_gpu, N, Mld, thread_rand_seed, mutexes_sizeN,\
            Xld, Khd, Kld, neighsLD, neighsHD, furthest_neighdists_LD, P, mutex_P);
    } else {
        thing->maker_cpu = (EmbeddingMaker_CPU*) malloc(sizeof(EmbeddingMaker_CPU));
        new_EmbeddingMaker_CPU(thing->maker_cpu, N, Mld, thread_rand_seed, mutexes_sizeN,\
            Xld, Khd, Kld, neighsLD, neighsHD, furthest_neighdists_LD, P, mutex_P);
    }
}

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