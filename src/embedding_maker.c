#include "embedding_maker.h"

void new_EmbeddingMaker_CPU(EmbeddingMaker_CPU* thing, uint32_t N, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P){
    dying_breath("CPU-based embedding maker not implemented yet");
}

void new_EmbeddingMaker_GPU(EmbeddingMaker_GPU* thing, uint32_t N, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
        float** Xld, uint32_t Khd, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
        float** P, pthread_mutex_t* mutex_P,\
    GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsHD, GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsLD, GPU_CPU_float_buffer* GPU_CPU_comms_P){
    thing->mutex_thread = mutex_allocate_and_init();
    thing->rand_state = ++thread_rand_seed[0];
    thing->is_running = false;
    thing->work_type = 0;
    thing->N = N;
    thing->Khd = Khd;
    thing->mutexes_sizeN = mutexes_sizeN;
    thing->hparam_LDkernel_alpha       = malloc_float(1, 1.0f);
    thing->mutex_hparam_LDkernel_alpha = mutex_allocate_and_init();
    thing->hparam_repulsion_multiplier = malloc_float(1, 1.0f);
    thing->mutex_hparam_repulsion_multiplier = mutex_allocate_and_init();
    thing->Xld_cpu = Xld;
    thing->neighsLD_cpu = neighsLD;
    thing->neighsHD_cpu = neighsHD;
    thing->furthest_neighdists_LD_cpu = furthest_neighdists_LD;
    thing->mutex_P = mutex_P;

    // initialise the Qdenom value with an estoimator
    double Qdenom_accumulator = 0.0;
    uint32_t n_samples = 10000u;
    for(uint32_t n_votes = 0; n_votes < n_samples; n_votes++){
        uint32_t i = rand_uint32_between(&thing->rand_state, 0u, N);
        uint32_t j = rand_uint32_between(&thing->rand_state, 0u, N);
        Qdenom_accumulator += 1.0f / powf(1.0f + f_euclidean_sq_in_embedding(Xld[i], Xld[j])/thing->hparam_LDkernel_alpha[0], thing->hparam_LDkernel_alpha[0]);
    }
    uint32_t matrix_area = N * (N-1);
    double   scaling_factor = (double) (matrix_area) / (double) n_samples;
    thing->Qdenom_EMA = (float) (Qdenom_accumulator * scaling_factor);

    // initialise CUDA and get device properties
    struct cudaDeviceProp prop = initialise_cuda();
    print_cuda_device_info(prop);

    // if compute_capability < 3.5, die
    if(prop.major < 3 || (prop.major == 3 && prop.minor < 5)){
        dying_breath("compute capability of the GPU is < 3.5: please use the CPU-based version instead");}

    // safe GPU / CPU communication: neighsHD and Psym
    thing->GPU_CPU_comms_neighsHD = GPU_CPU_comms_neighsHD;
    thing->GPU_CPU_comms_neighsLD = GPU_CPU_comms_neighsLD;
    thing->GPU_CPU_comms_P        = GPU_CPU_comms_P;

    // streams    
    if(cudaStreamCreate(&thing->stream_K_HD) != cudaSuccess){
        dying_breath("cudaStreamCreate error");}
    if(cudaStreamCreate(&thing->stream_K_LD) != cudaSuccess){
        dying_breath("cudaStreamCreate error");}
    if(cudaStreamCreate(&thing->stream_rand) != cudaSuccess){
        dying_breath("cudaStreamCreate error");}
    
    // things on GPU
    malloc_1d_float_cuda(&thing->Xld_base_cuda, N*Mld);
    malloc_1d_float_cuda(&thing->Xld_nesterov_cuda, N*Mld);
    malloc_1d_float_cuda(&thing->momenta_attraction_cuda, N*Mld);
    malloc_1d_float_cuda(&thing->momenta_repulsion_far_cuda, N*Mld);
    malloc_1d_float_cuda(&thing->momenta_repulsion_cuda, N*Mld);
    malloc_1d_uint32_cuda(&thing->neighsLD_cuda, N*Kld);
    malloc_1d_uint32_cuda(&thing->neighsHD_cuda, N*Khd);
    malloc_1d_float_cuda(&thing->all_neighdists_LD_cuda, N*Kld); //nouveau
    malloc_1d_float_cuda(&thing->furthest_neighdists_LD_cuda, N);
    malloc_1d_uint32_cuda(&thing->N_elements_of_Qdenom_cuda, 1);
    malloc_1d_uint32_cuda(&thing->random_numbers_size_NxRand_cuda, N*NB_RANDOM_POINTS_FAR_REPULSION);
    uint32_t N_elements_of_Qdenom = N * (Khd + Kld + NB_RANDOM_POINTS_FAR_REPULSION);
    malloc_1d_double_cuda(&thing->elements_of_Qdenom_cuda, N_elements_of_Qdenom);
    malloc_1d_float_cuda(&thing->P_cuda, N*Khd);
    malloc_1d_float_cuda(&thing->now_Qdenom_cuda, 1);

    // copy values from the arrays on the CPU
    memcpy_CPU_to_CUDA_float(thing->Xld_base_cuda, as_float_1d(Xld, N, Mld), N*Mld);
    memcpy_CPU_to_CUDA_float(thing->Xld_nesterov_cuda, as_float_1d(Xld, N, Mld), N*Mld);
    memcpy_CPU_to_CUDA_uint32(thing->neighsLD_cuda, as_uint32_1d(neighsLD, N, Kld), N*Kld);
    memcpy_CPU_to_CUDA_uint32(thing->neighsHD_cuda, as_uint32_1d(neighsHD, N, Khd), N*Khd);
    memcpy_CPU_to_CUDA_float(thing->furthest_neighdists_LD_cuda, furthest_neighdists_LD, N);
    memcpy_CPU_to_CUDA_float(thing->P_cuda, as_float_1d(P, N, Khd), N*Khd);
    memcpy_CPU_to_CUDA_uint32(thing->N_elements_of_Qdenom_cuda, &N_elements_of_Qdenom, 1);
    // init to 0.0f all momenta
    cudaError_t err1 = cudaMemset(thing->momenta_attraction_cuda, 0, N*Mld*sizeof(float));
    cudaError_t err2 = cudaMemset(thing->momenta_repulsion_far_cuda, 0, N*Mld*sizeof(float));
    cudaError_t err3 = cudaMemset(thing->momenta_repulsion_cuda, 0, N*Mld*sizeof(float));
    cudaError_t err4 = cudaMemset(thing->all_neighdists_LD_cuda, 0, N*Kld*sizeof(float));
    if(err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess){
        dying_breath("cudamemset error");
    }
    // fill thing->random_numbers_size_NxRand_cuda with random numbers inside [0, N[
    uint32_t* random_numbers_size_NxRand = malloc_uint32_t(N*NB_RANDOM_POINTS_FAR_REPULSION, 0u);
    for(uint32_t i = 0; i < N*NB_RANDOM_POINTS_FAR_REPULSION; i++){
        random_numbers_size_NxRand[i] = rand_uint32_between(&thing->rand_state, 0u, N);
    }
    memcpy_CPU_to_CUDA_uint32(thing->random_numbers_size_NxRand_cuda, random_numbers_size_NxRand, N*NB_RANDOM_POINTS_FAR_REPULSION);

    // init value for the denominator : 1.0f
    float one = 1.0f;
    memcpy_CPU_to_CUDA_float(thing->now_Qdenom_cuda, &one, 1u);

/***
 *     _   __                     _       _                           
 *    | | / /                    | |     | |                          
 *    | |/ /  ___ _ __ _ __   ___| |  ___| |__   __ _ _ __   ___  ___ 
 *    |    \ / _ \ '__| '_ \ / _ \ | / __| '_ \ / _` | '_ \ / _ \/ __|
 *    | |\  \  __/ |  | | | |  __/ | \__ \ | | | (_| | |_) |  __/\__ \
 *    \_| \_/\___|_|  |_| |_|\___|_| |___/_| |_|\__,_| .__/ \___||___/
 *                                                   | |              
 *                                                   |_|              
 */
    uint32_t smem_max_N_floats_per_block      = prop.sharedMemPerBlock/sizeof(float);
    uint32_t registers_max_N_floats_per_block = prop.regsPerBlock;
    // target block size (1-dimensional)   (also check that it's indeed a multiple of 32)
    uint32_t target_block_size = prop.maxThreadsDim[0] / 2;
    if(target_block_size % 32 != 0){dying_breath("target block size is not a multiple of 32\n");}

    // ~~~~~~~~~  Kernel 1: HD neighbours  ~~~~~~~~~
    uint32_t max_nb_different_i = 2u + (target_block_size) / Khd;
    // determine block size and number of blocks
    uint32_t KernHD_block_size = target_block_size;
    bool size_is_ok = false;
    while(!size_is_ok){
        printf("here the register size is wrong\n");
        uint32_t block_register_n_32bits = KernHD_block_size * (1u + 1u + 1u + 2u + (2u * Mld));
        uint32_t block_smem_n_32bits = (max_nb_different_i * (2u * Mld)) + (max_nb_different_i * Mld) + (KernHD_block_size * (4u * Mld));
        bool smem_ok = block_smem_n_32bits     < smem_max_N_floats_per_block;
        bool reg_ok  = 2u * block_register_n_32bits < registers_max_N_floats_per_block;
        size_is_ok = (smem_ok && reg_ok);
        if(!size_is_ok){
            KernHD_block_size -= 32u;}
    }
    // number of threads in total
    uint32_t n_threads_total = thing->N * thing->Khd;
    if(KernHD_block_size % 32u != 0u){dying_breath("block size is not a multiple of 32 (thats really wierd, where did you get your GPU?)\n");}
    thing->Kern_HD_n_blocks   = (n_threads_total + KernHD_block_size - 1u) / KernHD_block_size;
    thing->Kern_HD_block_size = KernHD_block_size;
    if((int)thing->Kern_HD_n_blocks > prop.maxGridSize[0]){dying_breath("too many blocks for the GPU, please use the CPU or refactor the code to use 2d grids\n");}
}

// 1: gradient descent: fill momenta_attraction, momenta_repulsion_far, momenta_repulsion
// 2: this also recomputes the furthest_neighdists_LD
void fill_raw_momenta_GPU(EmbeddingMaker_GPU* thing){
    // get the alpha hyperparameter, for the simplified Cauchy kernel
    pthread_mutex_lock(thing->mutex_hparam_LDkernel_alpha);
    float cauchy_alpha = thing->hparam_LDkernel_alpha[0];
    pthread_mutex_unlock(thing->mutex_hparam_LDkernel_alpha);

    fill_raw_momenta_launch_cuda(thing->stream_K_HD, thing->stream_K_LD, thing->stream_rand, (int)thing->Kern_HD_block_size, (int)thing->Kern_HD_n_blocks, thing->N, thing->Khd, thing->P_cuda, thing->Xld_nesterov_cuda, thing->neighsHD_cuda, thing->furthest_neighdists_LD_cuda, thing->Qdenom_EMA, cauchy_alpha, thing->elements_of_Qdenom_cuda, thing->momenta_attraction_cuda, thing->momenta_repulsion_far_cuda);
    die();
}

// momentum leak: momenta_repulsion_far gets smoothed across neighbours (with conservation of vector norm)
// momenta_repulsion_far_cuda gets leaked entirely to momenta_repulsion_cuda
void momenta_leak_GPU(EmbeddingMaker_GPU* thing){

}




// apply momenta to Xld, regenerate Xld_nesterov, decay momenta
void apply_momenta_and_decay_GPU(EmbeddingMaker_GPU* thing){
    // get the repulsion multiplier hyperparameter
    /* pthread_mutex_lock(thing->mutex_hparam_repulsion_multiplier);
    float repulsion_multiplier = thing->hparam_repulsion_multiplier[0];
    pthread_mutex_unlock(thing->mutex_hparam_repulsion_multiplier); */
}


// depending on the (user-determined) use of GPU vs CPU, this initialises the appropriate struct
void new_EmbeddingMaker(EmbeddingMaker* thing, uint32_t N, uint32_t* thread_rand_seed, pthread_mutex_t* mutexes_sizeN,\
    float** Xld, uint32_t Khd, uint32_t** neighsLD, uint32_t** neighsHD, float* furthest_neighdists_LD,\
    float** P, pthread_mutex_t* mutex_P,\
    GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsHD, GPU_CPU_uint32_buffer* GPU_CPU_comms_neighsLD
    , GPU_CPU_float_buffer* GPU_CPU_comms_P){
    thing->maker_cpu = NULL;
    thing->maker_gpu = NULL;
    if(USE_GPU){
        thing->maker_gpu = (EmbeddingMaker_GPU*) malloc(sizeof(EmbeddingMaker_GPU));
        new_EmbeddingMaker_GPU(thing->maker_gpu, N, thread_rand_seed, mutexes_sizeN,\
            Xld, Khd, neighsLD, neighsHD, furthest_neighdists_LD, P, mutex_P,\
            GPU_CPU_comms_neighsHD, GPU_CPU_comms_neighsLD, GPU_CPU_comms_P);
    } else {
        thing->maker_cpu = (EmbeddingMaker_CPU*) malloc(sizeof(EmbeddingMaker_CPU));
        new_EmbeddingMaker_CPU(thing->maker_cpu, N, thread_rand_seed, mutexes_sizeN,\
            Xld, Khd, neighsLD, neighsHD, furthest_neighdists_LD, P, mutex_P);
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
    // Xld: GPU to CPU
    memcpy_CUDA_to_CPU_float(as_float_1d(thing->Xld_cpu, thing->N, Mld), thing->Xld_base_cuda, thing->N*Mld);
    // furthest_neighdists_LD: GPU to CPU
    memcpy_CUDA_to_CPU_float(thing->furthest_neighdists_LD_cpu, thing->furthest_neighdists_LD_cuda, thing->N);
}


// this function receives the neighs and P from the CPU, in a SAFE manner
//  Read if ready, then request sync
static void receive_neighs_and_P_from_CPU(EmbeddingMaker_GPU* thing){
    // 1) neighsLD: CPU to GPU.
    GPU_CPU_sync* sync_neigh_LD = &thing->GPU_CPU_comms_neighsLD->sync;

    if(is_ready_now(sync_neigh_LD)){
        pthread_mutex_lock(sync_neigh_LD->mutex_buffer);
        memcpy_CPU_to_CUDA_uint32(thing->neighsLD_cuda, thing->GPU_CPU_comms_neighsLD->buffer, thing->N*Kld);
        // cudaMemcpy(thing->neighsLD_cuda, thing->GPU_CPU_comms_neighsLD->buffer, thing->N*thing->Kld*sizeof(uint32_t), cudaMemcpyHostToDevice);
        pthread_mutex_unlock(sync_neigh_LD->mutex_buffer);
        set_ready(sync_neigh_LD, false);
    }
    if(!is_requesting_now(sync_neigh_LD)){
        notify_request(sync_neigh_LD);
    } // request for the next sync

    // 2) neighsHD: CPU to GPU.
    GPU_CPU_sync* sync_neigh_HD = &thing->GPU_CPU_comms_neighsHD->sync;
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
        notify_request(sync_P);}
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
    double start_time = time_seconds();
    while(thing->is_running){
        // ~~~~~~~~~~ gradient descent ~~~~~~~~~~
        // gradient descent: fill momenta_attraction, momenta_repulsion_far, momenta_repulsion
        fill_raw_momenta_GPU(thing);

        // momentum leak: momenta_repulsion_far gets smoothed across neighbours (with conservation of vector norm)
        momenta_leak_GPU(thing);

        // apply momenta to Xld, regenerate Xld_nesterov, decay momenta
        apply_momenta_and_decay_GPU(thing);

        // ~~~~~~~~~~ sync with CPU workers ~~~~~~~~~~
        // 1) "UNSAFE" syncs (only 1 writer so it's okay)
        send_Xld_and_furthest_neighdists_LD_to_CPU(thing);
        // 2) SAFE syncs, periodically
        double time_elapsed = (time_seconds() - start_time);
        if(time_elapsed > GUI_CPU_SYNC_PERIOD){
            receive_neighs_and_P_from_CPU(thing);
            start_time = time_seconds();
        }
        // printf("it is important to update the furhtest dist to LD neighs in the tSNE optimisation, when computing them\n");
    }
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