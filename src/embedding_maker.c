#include "embedding_maker.h"


void new_EmbeddingMaker_CPU(EmbeddingMaker_CPU* thing, uint32_t N, uint32_t Mld, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t Kld, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P){
    dying_breath("CPU-based embedding maker not implemented yet");
}

void new_EmbeddingMaker_GPU(EmbeddingMaker_GPU* thing, uint32_t N, uint32_t Mld, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
        float** Xld, uint32_t Khd, uint32_t Kld, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
        float** P, pthread_mutex_t* mutex_P,\
    GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsHD, GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsLD, GPU_CPU_float_buffer* GPU_CPU_comms_P){
    thing->mutex_thread = mutex_allocate_and_init();
    thing->rand_state = ++thread_rand_seed[0];
    thing->is_running = false;
    thing->work_type = 0;
    thing->N = N;
    thing->Mld = Mld;
    thing->Kld = Kld;
    thing->Khd = Khd;
    thing->mutexes_sizeN = mutexes_sizeN;
    thing->hparam_LDkernel_alpha       = malloc_float(1, 1.0f);
    thing->mutex_hparam_LDkernel_alpha = mutex_allocate_and_init();
    thing->Xld_cpu = Xld;
    thing->neighsLD_cpu = neighsLD;
    thing->neighsHD_cpu = neighsHD;
    thing->furthest_neighdists_LD_cpu = furthest_neighdists_LD;
    thing->P_cpu = P;
    thing->mutex_P = mutex_P;

    // initialise CUDA adn get device properties
    struct cudaDeviceProp prop = initialise_cuda();
    print_cuda_device_info(prop);

    // safe GPU / CPU communication: neighsHD and Psym
    thing->GPU_CPU_comms_neighsHD = GPU_CPU_comms_neighsHD;
    thing->GPU_CPU_comms_neighsLD = GPU_CPU_comms_neighsLD;
    thing->GPU_CPU_comms_P        = GPU_CPU_comms_P;

    // things on GPU
    cudaError_t cuda_error;
    malloc_1d_float_cuda(&thing->Xld_base_cuda, N*Mld);
    malloc_1d_float_cuda(&thing->Xld_nesterov_cuda, N*Mld);
    malloc_1d_float_cuda(&thing->momenta_attraction_cuda, N*Mld);
    malloc_1d_float_cuda(&thing->momenta_repulsion_far_cuda, N*Mld);
    malloc_1d_float_cuda(&thing->momenta_repulsion_cuda, N*Mld);
    malloc_1d_uint32_cuda(&thing->neighsLD_cuda, N*Kld);
    malloc_1d_uint32_cuda(&thing->neighsHD_cuda, N*Khd);
    malloc_1d_float_cuda(&thing->furthest_neighdists_LD_cuda, N);
    malloc_1d_float_cuda(&thing->P_cuda, N*Khd);
    malloc_1d_float_cuda(&thing->Qdenom_cuda, 1);


    memcpy_CPU_to_CUDA_float(thing->Xld_base_cuda, as_float_1d(Xld, N, Mld), N*Mld);
    memcpy_CPU_to_CUDA_float(thing->Xld_nesterov_cuda, as_float_1d(Xld, N, Mld), N*Mld);
    memcpy_CPU_to_CUDA_float(thing->momenta_attraction_cuda, as_float_1d(Xld, N, Mld), N*Mld);
    memcpy_CPU_to_CUDA_float(thing->momenta_repulsion_far_cuda, as_float_1d(Xld, N, Mld), N*Mld);
    memcpy_CPU_to_CUDA_float(thing->momenta_repulsion_cuda, as_float_1d(Xld, N, Mld), N*Mld);
    memcpy_CPU_to_CUDA_uint32(thing->neighsLD_cuda, as_uint32_1d(neighsLD, N, Kld), N*Kld);
    memcpy_CPU_to_CUDA_uint32(thing->neighsHD_cuda, as_uint32_1d(neighsHD, N, Khd), N*Khd);
    memcpy_CPU_to_CUDA_float(thing->furthest_neighdists_LD_cuda, furthest_neighdists_LD, N);
    memcpy_CPU_to_CUDA_float(thing->P_cuda, as_float_1d(P, N, Khd), N*Khd);
    float one = 1.0f;
    memcpy_CPU_to_CUDA_float(thing->Qdenom_cuda, &one, 1u);
}


/* "
__shared__ float v[PDIST* BLOCKDIMX];
cudaStream_t streams[PDIST];
for (int k = 0; k < PDIST; ++k) {
    cudaStreamCreate(&streams[k]);
    cudaMemcpyAsync(&v[k], &arr[threadIdx.x + k * BLOCKDIMX], 8, cudaMemcpyDeviceToDevice, streams[k]);
}

for (int i = threadIdx.x, ctr = 0; i < imax; i += BLOCKDIMX, ctr++) {
    int ctr_mod = ctr % PDIST;
    cudaStreamSynchronize(streams[ctr_mod]);
    float locvar = v[ctr_mod];
    if (i < imax - PDIST * BLOCKDIMX) {
        cudaMemcpyAsync(&v[ctr_mod], &arr[i + PDIST * BLOCKDIMX], 8, cudaMemcpyDeviceToDevice, streams[ctr_mod]);
    }
    // More instructions using locvar, for example, transcendentals
}
" */


// depending on the (user-determined) use of GPU vs CPU, this initialises the appropriate struct
void new_EmbeddingMaker(EmbeddingMaker* thing, uint32_t N, uint32_t Mld, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t Kld, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P,\
    GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsHD, GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsLD
    , GPU_CPU_float_buffer* GPU_CPU_comms_P){
    thing->maker_cpu = NULL;
    thing->maker_gpu = NULL;
    if(USE_GPU){
        thing->maker_gpu = (EmbeddingMaker_GPU*) malloc(sizeof(EmbeddingMaker_GPU));
        new_EmbeddingMaker_GPU(thing->maker_gpu, N, Mld, thread_rand_seed, mutexes_sizeN,\
            Xld, Khd, Kld, neighsLD, neighsHD, furthest_neighdists_LD, P, mutex_P,\
            GPU_CPU_comms_neighsHD, GPU_CPU_comms_neighsLD, GPU_CPU_comms_P);
    } else {
        thing->maker_cpu = (EmbeddingMaker_CPU*) malloc(sizeof(EmbeddingMaker_CPU));
        new_EmbeddingMaker_CPU(thing->maker_cpu, N, Mld, thread_rand_seed, mutexes_sizeN,\
            Xld, Khd, Kld, neighsLD, neighsHD, furthest_neighdists_LD, P, mutex_P);
    }
}

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


// this function sends the Xld and furthest_neighdists_LD towards the CPU, in an UNSAFE manner
static void send_Xld_and_furthest_neighdists_LD_to_CPU(EmbeddingMaker_GPU* thing){
    /* // add random noise to Xld
    for(uint32_t i = 0; i < thing->N; i++){
        for(uint32_t j = 0; j < thing->Mld; j++){
            thing->Xld_cpu[i][j] = thing->Xld_cpu[i][j] + 0.01f * (float) rand_r(&thing->rand_state) / (float) RAND_MAX;
        }
    } */
    // Xld: GPU to CPU
    memcpy_CUDA_to_CPU_float(as_float_1d(thing->Xld_cpu, thing->N, thing->Mld), thing->Xld_base_cuda, thing->N*thing->Mld);
    // furthest_neighdists_LD: GPU to CPU
    memcpy_CUDA_to_CPU_float(thing->furthest_neighdists_LD_cpu, thing->furthest_neighdists_LD_cuda, thing->N);
}

// this function receives the neighs and P from the CPU, in a SAFE manner
//  Read if ready, then request sync
static void receive_neighs_and_P_from_CPU(EmbeddingMaker_GPU* thing){
    // 1) neighsLD: CPU to GPU.
    GPU_CPU_sync* sync_neigh_LD = &thing->GPU_CPU_comms_neighsLD->sync;

    bool is_ready = is_ready_now(sync_neigh_LD);
    bool is_requesting = is_requesting_now(sync_neigh_LD);
    printf("\n\n(GPU)   is_ready: %d, is_requesting: %d\n\n", is_ready, is_requesting);

    if(is_ready_now(sync_neigh_LD)){
        pthread_mutex_lock(sync_neigh_LD->mutex_buffer);
        memcpy_CPU_to_CUDA_uint32(thing->neighsLD_cuda, thing->GPU_CPU_comms_neighsLD->buffer, thing->N*thing->Kld);
        // cudaMemcpy(thing->neighsLD_cuda, thing->GPU_CPU_comms_neighsLD->buffer, thing->N*thing->Kld*sizeof(uint32_t), cudaMemcpyHostToDevice);
        pthread_mutex_unlock(sync_neigh_LD->mutex_buffer);
        set_ready(sync_neigh_LD, false);
        printf("\n\n---------------  neighsLD received from CPU\n\n");
    }
    if(!is_requesting_now(sync_neigh_LD)){
        notify_request(sync_neigh_LD);
        printf("\n\n-----------------  neighsLD requested by GPU\n\n");
    } // request for the next sync




    // 2) neighsHD: CPU to GPU.
    /* GPU_CPU_sync* sync_neigh_HD = &thing->GPU_CPU_comms_neighsHD->sync;
    if(is_ready_now(sync_neigh_HD)){
        pthread_mutex_lock(sync_neigh_HD->mutex_buffer);
        memcpy_CPU_to_CUDA_uint32(thing->neighsHD_cuda, thing->GPU_CPU_comms_neighsHD->buffer, thing->N*thing->Khd);
        // cudaMemcpy(thing->neighsHD_cuda, thing->GPU_CPU_comms_neighsHD->buffer, thing->N*thing->Khd*sizeof(uint32_t), cudaMemcpyHostToDevice);
        pthread_mutex_unlock(sync_neigh_HD->mutex_buffer);
        set_ready(sync_neigh_HD, false);
    }
    if(!is_requesting_now(sync_neigh_HD)){
        notify_request(sync_neigh_HD);}

    // 3) P: CPU to GPU.
    GPU_CPU_sync* sync_P = &thing->GPU_CPU_comms_P->sync;
    if(is_ready_now(sync_P)){
        pthread_mutex_lock(sync_P->mutex_buffer);
        memcpy_CPU_to_CUDA_float(thing->P_cuda, thing->GPU_CPU_comms_P->buffer, thing->N*thing->Khd);
        // cudaMemcpy(thing->P_cuda, thing->GPU_CPU_comms_P->buffer, thing->N*thing->Khd*sizeof(float), cudaMemcpyHostToDevice);
        pthread_mutex_unlock(sync_P->mutex_buffer);
        set_ready(sync_P, false);
    }
    if(!is_requesting_now(sync_P)){
        notify_request(sync_P);} */
}


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
    thing->is_running = true;

    double start_time, current_time; // convert to seconds
    start_time = time_seconds();
    while(thing->is_running){
        // ~~~~~~~~~~ gradient descent ~~~~~~~~~~
        // gradient descent: fill momenta_attraction, momenta_repulsion_far, momenta_repulsion
        // ...

        // momentum leak: momenta_repulsion_far gets smoothed across neighbours (with conservation of vector norm)
        // ...

        // apply momenta to Xld, regenerate Xld_nesterov, decay momenta
        // ...


todo: je pense que synchro n est pas ok : trouver ou ca bloque
trouver pk % change ne change pas, meme en bloquant ca devrait changer....

        // ~~~~~~~~~~ sync with CPU workers ~~~~~~~~~~
        // 1) "UNSAFE" syncs (only 1 writer so it's okay)
        send_Xld_and_furthest_neighdists_LD_to_CPU(thing);
        // 2) SAFE syncs, periodically
        current_time = time_seconds();
        double time_elapsed = (current_time - start_time);

        printf("time_elapsed : %lf    start : %lf    current_time %lf\n", time_elapsed, start_time, current_time);    
        if(time_elapsed > GUI_CPU_SYNC_PERIOD){
            receive_neighs_and_P_from_CPU(thing);
            start_time = time_seconds();
        }
        // pour le moment, ^ en commentaires car je fais des cudaMemcpy mais Cda n est pas encore initialisé donc segfault
        // printf("it is important to update the furhtest dist to LD neighs in the tSNE optimisation, when computing them\n");
    }


    /* clock_t start_time, current_time; // convert to seconds
    start_time = clock() / CLOCKS_PER_SEC;
    while(thing->is_running){
        // ~~~~~~~~~~ gradient descent ~~~~~~~~~~
        // gradient descent: fill momenta_attraction, momenta_repulsion_far, momenta_repulsion
        // ...

        // momentum leak: momenta_repulsion_far gets smoothed across neighbours (with conservation of vector norm)
        // ...

        // apply momenta to Xld, regenerate Xld_nesterov, decay momenta
        // ...

        // ~~~~~~~~~~ sync with CPU workers ~~~~~~~~~~
        // 1) "UNSAFE" syncs (only 1 writer so it's okay)
        send_Xld_and_furthest_neighdists_LD_to_CPU(thing);
        // 2) SAFE syncs, periodically
        current_time = clock() / CLOCKS_PER_SEC;
        double time_elapsed = (double) (current_time - start_time);
        printf("time_elapsed : %lf    start : %lf    current_time %lf\n", time_elapsed, (double) start_time, (double) current_time);    
        if(time_elapsed > GUI_CPU_SYNC_PERIOD){
            receive_neighs_and_P_from_CPU(thing);
            start_time = clock();
        }
        // pour le moment, ^ en commentaires car je fais des cudaMemcpy mais Cda n est pas encore initialisé donc segfault
        // printf("it is important to update the furhtest dist to LD neighs in the tSNE optimisation, when computing them\n");
    } */
    return NULL; 
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
    dying_breath("destroy_EmbeddingMaker not implemented yet");
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
        if(pthread_create(&thing->thread, NULL, routine_EmbeddingMaker_GPU, thing->maker_gpu) != 0){
            dying_breath("pthread_create routine_EmbeddingMaker failed");}
    }
    else {
        dying_breath("CPU-based embedding maker not implemented yet");
    }
    printf("TODO : understand CUDA streams! \n");
}