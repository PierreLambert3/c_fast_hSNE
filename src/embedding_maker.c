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

    // if Khd is not a multiple of 32, die
    if(Khd % 32 != 0){
        dying_breath("Khd is not a multiple of 32");}

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
    if(cudaStreamCreate(&thing->stream_nudge_HD) != cudaSuccess){
        dying_breath("cudaStreamCreate error");}
    if(cudaStreamCreate(&thing->stream_nudge_LD) != cudaSuccess){
        dying_breath("cudaStreamCreate error");}
    if(cudaStreamCreate(&thing->stream_nudge_FAR) != cudaSuccess){
        dying_breath("cudaStreamCreate error");}
    if(cudaStreamCreate(&thing->stream_Qdenomsum) != cudaSuccess){
        dying_breath("cudaStreamCreate error");}
    if(cudaStreamCreate(&thing->stream_leak) != cudaSuccess){
        dying_breath("cudaStreamCreate error");}
    if(cudaStreamCreate(&thing->stream_parameter_updates) != cudaSuccess){
        dying_breath("cudaStreamCreate error");}
    
    thing->leak_phase = 0;
    
    // things on GPU
    // nudges
    malloc_1d_float_cuda(&thing->cu_nudge_attrac_HD, N*Mld);
    malloc_1d_float_cuda(&thing->cu_nudge_repuls_HDLD, N*Mld);
    malloc_1d_float_cuda(&thing->cu_nudge_FAR, N*Mld);
    // momenta
    malloc_1d_float_cuda(&thing->cu_momenta_attrac, N*Mld);
    malloc_1d_float_cuda(&thing->cu_momenta_repuls_near, N*Mld);
    malloc_1d_float_cuda(&thing->cu_momenta_repuls_far___0, N*Mld);
    malloc_1d_float_cuda(&thing->cu_momenta_repuls_far___1, N*Mld);
    // init to 0.0f all nudges and momenta
    bool success       = (cudaMemset(thing->cu_nudge_attrac_HD, 0, N*Mld*sizeof(float)) == cudaSuccess);
    success = success && (cudaMemset(thing->cu_nudge_repuls_HDLD, 0, N*Mld*sizeof(float)) == cudaSuccess);
    success = success && (cudaMemset(thing->cu_nudge_FAR, 0, N*Mld*sizeof(float)) == cudaSuccess);
    success = success && (cudaMemset(thing->cu_momenta_attrac, 0, N*Mld*sizeof(float)) == cudaSuccess);
    success = success && (cudaMemset(thing->cu_momenta_repuls_near, 0, N*Mld*sizeof(float)) == cudaSuccess);
    success = success && (cudaMemset(thing->cu_momenta_repuls_far___0, 0, N*Mld*sizeof(float)) == cudaSuccess);
    success = success && (cudaMemset(thing->cu_momenta_repuls_far___1, 0, N*Mld*sizeof(float)) == cudaSuccess);
    if(!success){
        dying_breath("cudamemset error");}

    malloc_1d_float_cuda(&thing->cu_Xld_base, N*Mld);
    malloc_1d_float_cuda(&thing->cu_Xld_nesterov, N*Mld);

    malloc_1d_uint32_cuda(&thing->cu_neighsLD, N*Kld);
    malloc_1d_uint32_cuda(&thing->cu_neighsHD, N*Khd);
    malloc_1d_float_cuda(&thing->cu_furthest_neighdists_LD, N);
    malloc_1d_float_cuda(&thing->cu_temporary_furthest_neighdists_LD, N);
    malloc_1d_uint32_cuda(&thing->cu_random_numbers_size_NxRand, N*NB_RANDOM_POINTS_FAR_REPULSION);
    malloc_1d_float_cuda(&thing->cu_P, N*Khd); 

    // copy values from the arrays on the CPU
    memcpy_CPU_to_CUDA_float(thing->cu_Xld_base, as_float_1d(Xld, N, Mld), N*Mld);
    memcpy_CPU_to_CUDA_float(thing->cu_Xld_nesterov, as_float_1d(Xld, N, Mld), N*Mld);

    memcpy_CPU_to_CUDA_uint32(thing->cu_neighsLD, as_uint32_1d(neighsLD, N, Kld), N*Kld);
    memcpy_CPU_to_CUDA_uint32(thing->cu_neighsHD, as_uint32_1d(neighsHD, N, Khd), N*Khd);
    memcpy_CPU_to_CUDA_float(thing->cu_furthest_neighdists_LD, furthest_neighdists_LD, N);
    memcpy_CPU_to_CUDA_float(thing->cu_P, as_float_1d(P, N, Khd), N*Khd);
    
    // fill thing->random_numbers_size_NxRand_cuda with random numbers inside [0, N[
    uint32_t* random_numbers_size_NxRand = malloc_uint32_t(N*NB_RANDOM_POINTS_FAR_REPULSION, 0u);
    for(uint32_t i = 0; i < N*NB_RANDOM_POINTS_FAR_REPULSION; i++){
        random_numbers_size_NxRand[i] = (uint32_t)rand();} // not using my shitty homemade random: else rand of i at t is the same as rand of (i+1) at (t+1)
    memcpy_CPU_to_CUDA_uint32(thing->cu_random_numbers_size_NxRand, random_numbers_size_NxRand, N*NB_RANDOM_POINTS_FAR_REPULSION);
    free_array(random_numbers_size_NxRand);

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
    if(Khd > (uint32_t)prop.maxThreadsDim[0]){
        dying_breath("Khd is too large for the GPU");}
    uint32_t smem_max_N_floats = prop.sharedMemPerBlock/sizeof(float);
    uint32_t reg_max_N_floats  = prop.regsPerBlock;

    float smem_pct_target_occupancy = 0.667f;
    float reg_pct_target_occupancy  = 0.725f;
    uint32_t target_n_floats_smem = (uint32_t) floorf(smem_pct_target_occupancy * (float) smem_max_N_floats);
    uint32_t target_n_floats_regs = (uint32_t) floorf(reg_pct_target_occupancy * (float)reg_max_N_floats);
    
    // ~~~~~~~~~  Kernel 1: HD neighbours, determining block size and grid shape  ~~~~~~~~~
    // grid 1d ; block 2d : (Khd, Ni)
    // smem_N_floats          : Ni * Mld   +  (Ni*Khd)*(2u*Mld)
    // smem memory constraint : smem_N_floats < smem_pct_target_occupancy * smem_max_N_floats
    // registers_N_floats          : (Ni*Khd) * (kern1_estimated_regs + Mld)
    // registers memory constraint : registers_N_floats < 0.75 * reg_max_N_floats
    // threads per block contraint : prop.maxThreadsPerBlock
    thing->Kern_HD_gridshape  = malloc_uint32_t(3, 1u);
    thing->Kern_HD_blockshape = malloc_uint32_t(3, 1u);
    uint32_t kern1_estimated_regs = 24u; // find a good value for this
    uint32_t kern1_Ni = 1u;
    while (true) {
        uint32_t next_smem_N_floats = (kern1_Ni + 1) * (Mld) + ((kern1_Ni + 1) * Khd) * (2u * Mld);
        uint32_t next_reg_N_floats  = ((kern1_Ni + 1) * Khd) * (kern1_estimated_regs + Mld);
        bool next_blocksize_ok = (kern1_Ni + 1) < (uint32_t) prop.maxThreadsDim[1]; // for 2nd dimension (first dim is fixed to Khd)
        bool next_reg_ok  = next_reg_N_floats  <= target_n_floats_regs;
        bool next_smem_ok = next_smem_N_floats <= target_n_floats_smem;
        bool maxthreads_ok = (kern1_Ni + 1) * Khd <= (uint32_t) prop.maxThreadsPerBlock;
        if(!next_smem_ok || !next_reg_ok || !next_blocksize_ok || !maxthreads_ok){
            break;}
        kern1_Ni++;
    }
    uint32_t smem_N_floats = kern1_Ni * Mld + (kern1_Ni * Khd) * (2u * Mld);
    uint32_t reg_N_floats  = (kern1_Ni * Khd) * (kern1_estimated_regs + Mld);
    bool smem_ok = smem_N_floats <= target_n_floats_smem;
    bool blocksize_ok = (kern1_Ni) <= (uint32_t) prop.maxThreadsDim[1];
    bool nthreads_ok = (kern1_Ni) * Khd <= (uint32_t) prop.maxThreadsPerBlock;
    if(!smem_ok || !blocksize_ok || !nthreads_ok){
        dying_breath("could not find a suitable block size for the kernel 1");}

    printf("\nblock shapes for kernel 1: (%u, %u)\n", Khd, kern1_Ni);
    printf("memory usage and maxima for kernel 1: smem_N_floats, %u target_n_floats_smem, %u  reg_N_floats, %u reg_max_N_floats %u  \n", smem_N_floats, target_n_floats_smem, reg_N_floats, reg_max_N_floats);
    printf("number of threads per block: %u\n", Khd* kern1_Ni);
    thing->Kern_HD_blockshape[0] = Khd;
    thing->Kern_HD_blockshape[1] = kern1_Ni; // Ni!
    thing->Kern_HD_blockshape[2] = 1u;
    thing->Kern_HD_gridshape[0]  = (N*Khd + (Khd * kern1_Ni) - 1u) / (Khd * kern1_Ni);
    printf("grid shape for kernel 1: %u\n\n", thing->Kern_HD_gridshape[0]);
    
    // ~~~~~~~~~  Kernel 2: LD neighbours, determining block size and grid shape  ~~~~~~~~~
    // grid 1d ; block 2d : (Kld, Ni)
    // smem_N_floats          : Ni * Mld   +  (Ni*Kld)*(2u*Mld)
    // registers_N_floats          : (Ni*Kld) * (kern1_estimated_regs + Mld)
    thing->Kern_LD_gridshape  = malloc_uint32_t(3, 1u);
    thing->Kern_LD_blockshape = malloc_uint32_t(3, 1u);
    uint32_t kern2_estimated_regs = 24u;
    uint32_t Kern2_Ni = 1u;
    while (true) {
        uint32_t next_smem_N_floats = (Kern2_Ni + 1) * (Mld) + ((Kern2_Ni + 1) * Kld) * (2u * Mld);
        uint32_t next_reg_N_floats  = ((Kern2_Ni + 1) * Kld) * (kern2_estimated_regs + Mld);
        bool next_blocksize_ok = (Kern2_Ni + 1) < (uint32_t) prop.maxThreadsDim[1]; // for 2nd dimension (first dim is fixed to Kld)
        bool next_reg_ok  = next_reg_N_floats  <= target_n_floats_regs;
        bool next_smem_ok = next_smem_N_floats <= target_n_floats_smem;
        bool maxthreads_ok = (Kern2_Ni + 1) * Kld <= (uint32_t) prop.maxThreadsPerBlock;
        if(!next_smem_ok || !next_reg_ok || !next_blocksize_ok || !maxthreads_ok){
            break;}
        Kern2_Ni++;
    }
    uint32_t smem_N_floats_Kern2 = Kern2_Ni * Mld + (Kern2_Ni * Kld) * (2u * Mld);
    uint32_t reg_N_floats_Kern2  = (Kern2_Ni * Kld) * (kern2_estimated_regs + Mld);
    bool smem_ok_Kern2      = smem_N_floats_Kern2 <= target_n_floats_smem;
    bool blocksize_ok_Kern2 = (Kern2_Ni) <= (uint32_t) prop.maxThreadsDim[1];
    nthreads_ok = (Kern2_Ni) * Kld <= (uint32_t) prop.maxThreadsPerBlock;
    if(!smem_ok_Kern2 || !blocksize_ok_Kern2 || !nthreads_ok){
        dying_breath("could not find a suitable block size for the kernel 2");}
    printf("block shapes for kernel 2: (%u, %u)\n", Kld, Kern2_Ni);
    printf("memory usage and maxima for kernel 2: smem_N_floats, %u target_n_floats_smem, %u  \n", smem_N_floats_Kern2, target_n_floats_smem);
    printf("number of threads per block: %u\n", Kld* Kern2_Ni);
    thing->Kern_LD_blockshape[0] = Kld;
    thing->Kern_LD_blockshape[1] = Kern2_Ni; // Ni!
    thing->Kern_LD_blockshape[2] = 1u;
    thing->Kern_LD_gridshape[0]  = (N*Kld + (Kld * Kern2_Ni) - 1u) / (Kld * Kern2_Ni);
    printf("grid shape for kernel 2: %u\n\n", thing->Kern_LD_gridshape[0]);


    // ~~~~~~~~~  Kernel 3: random far repulsion, determining block size and grid shape  ~~~~~~~~~
    // grid 1d ; block 2d : (NB_RANDOM_POINTS_FAR_REPULSION, Ni)
    // smem_N_floats          : Ni * Mld   +  (Ni*NB_RANDOM_POINTS_FAR_REPULSION)*(2u*Mld)
    // registers_N_floats          : (Ni*NB_RANDOM_POINTS_FAR_REPULSION) * (kern1_estimated_regs + Mld)
    thing->Kern_FAR_gridshape  = malloc_uint32_t(3, 1u);
    thing->Kern_FAR_blockshape = malloc_uint32_t(3, 1u);
    uint32_t kern3_estimated_regs = 24u;
    uint32_t Kern3_Ni = 1u;
    while (true) {
        uint32_t next_smem_N_floats = (Kern3_Ni + 1) * (Mld) + ((Kern3_Ni + 1) * NB_RANDOM_POINTS_FAR_REPULSION) * (2u * Mld);
        uint32_t next_reg_N_floats  = ((Kern3_Ni + 1) * NB_RANDOM_POINTS_FAR_REPULSION) * (kern3_estimated_regs + Mld);
        bool next_blocksize_ok = (Kern3_Ni + 1) < (uint32_t) prop.maxThreadsDim[1]; // for 2nd dimension (first dim is fixed to NB_RANDOM_POINTS_FAR_REPULSION)
        bool next_reg_ok  = next_reg_N_floats  <= target_n_floats_regs;
        bool next_smem_ok = next_smem_N_floats <= target_n_floats_smem;
        bool maxthreads_ok = (Kern3_Ni + 1) * NB_RANDOM_POINTS_FAR_REPULSION <= (uint32_t) prop.maxThreadsPerBlock;
        if(!next_smem_ok || !next_reg_ok || !next_blocksize_ok || !maxthreads_ok){
            break;}
        Kern3_Ni++;
    }
    uint32_t smem_N_floats_Kern3 = Kern3_Ni * Mld + (Kern3_Ni * NB_RANDOM_POINTS_FAR_REPULSION) * (2u * Mld);
    bool smem_ok_Kern3 = smem_N_floats_Kern3 <= target_n_floats_smem;
    bool blocksize_ok_Kern3 = (Kern3_Ni) <= (uint32_t) prop.maxThreadsDim[1];
    nthreads_ok = (Kern3_Ni) * NB_RANDOM_POINTS_FAR_REPULSION <= (uint32_t) prop.maxThreadsPerBlock;
    if(!smem_ok_Kern3 || !blocksize_ok_Kern3 || !nthreads_ok){
        dying_breath("could not find a suitable block size for the kernel 3");}
    printf("block shapes for kernel 3: (%u, %u)\n", NB_RANDOM_POINTS_FAR_REPULSION, Kern3_Ni);
    printf("memory usage and maxima for kernel 3: smem_N_floats, %u target_n_floats_smem, %u    \n", smem_N_floats_Kern3, target_n_floats_smem);
    printf("number of threads per block: %u\n", NB_RANDOM_POINTS_FAR_REPULSION * Kern3_Ni);
    thing->Kern_FAR_blockshape[0] = NB_RANDOM_POINTS_FAR_REPULSION;
    thing->Kern_FAR_blockshape[1] = Kern3_Ni; // Ni!
    thing->Kern_FAR_blockshape[2] = 1u;
    thing->Kern_FAR_gridshape[0]  = (N*NB_RANDOM_POINTS_FAR_REPULSION + (NB_RANDOM_POINTS_FAR_REPULSION * Kern3_Ni) - 1u) / (NB_RANDOM_POINTS_FAR_REPULSION * Kern3_Ni);
    printf("grid shape for kernel 3: %u\n\n", thing->Kern_FAR_gridshape[0]);

    // ~~~~~~~~~  Kernel 4: computes the sum of the Qdenom elements  ~~~~~~~~~
    // some Qdenom estimation things
    thing->N_elements_of_Qdenom = thing->Kern_HD_gridshape[0] + thing->Kern_LD_gridshape[0] + thing->Kern_FAR_gridshape[0]; // each block for each of the 3 kernels will compute one element of Qdenom
    malloc_1d_double_cuda(&thing->cu_elements_of_Qdenom, thing->N_elements_of_Qdenom);
    malloc_1d_float_cuda(&thing->cu_sum_Qdenom_elements, 1u);
    thing->Kern_Qdenomsum_blockshape = malloc_uint32_t(3, 1u);
    thing->Kern_Qdenomsum_gridshape  = malloc_uint32_t(3, 1u);
    uint32_t Kern4_n_blocks = 1u;
    while(true){
        uint32_t next_Kern4_n_blocks = Kern4_n_blocks + 1;
        uint32_t next_smem_N_floats  = next_Kern4_n_blocks;
        bool next_smem_ok = next_smem_N_floats <= target_n_floats_smem;
        bool next_blocksize_ok = next_Kern4_n_blocks < (uint32_t) prop.maxThreadsDim[0];
        bool next_nthreads_ok  = next_Kern4_n_blocks <= (uint32_t) prop.maxThreadsPerBlock;
        if(!next_smem_ok || !next_blocksize_ok || !next_nthreads_ok){
            break;}
        Kern4_n_blocks++;
    }
    thing->Kern_Qdenomsum_blockshape[0] = Kern4_n_blocks;
    thing->Kern_Qdenomsum_blockshape[1] = 1u;
    thing->Kern_Qdenomsum_blockshape[2] = 1u;
    thing->Kern_Qdenomsum_gridshape[0]  = (thing->N_elements_of_Qdenom + Kern4_n_blocks - 1u) / Kern4_n_blocks;
    printf("block shapes for kernel 4: (%u, %u)  (grid: %u)  (n elements %d)\n", thing->Kern_Qdenomsum_blockshape[0], 1u, thing->Kern_Qdenomsum_gridshape[0], thing->N_elements_of_Qdenom);


    // ~~~~~~~~~  Kernel 5: leaking momenta  ~~~~~~~~~
    // grid 1d ; block 2d : (Kld, Ni)
    // smem_N_floats        : Ni * Mld  + Ni*Kld*Mld
    // registers_N_floats   : Ni * Mld
    thing->Kern_leak_gridshape  = malloc_uint32_t(3, 1u);
    thing->Kern_leak_blockshape = malloc_uint32_t(3, 1u);
    uint32_t Kern5_Ni = 1u;
    while(true){
        uint32_t next_smem_N_floats = (Kern5_Ni + 1u) * Mld + (Kern5_Ni + 1u) * Kld * Mld;
        uint32_t next_reg_N_floats  = (Kern5_Ni + 1u) * Mld;
        bool next_smem_ok = next_smem_N_floats <= target_n_floats_smem;
        bool next_reg_ok  = next_reg_N_floats  <= target_n_floats_regs;
        bool next_blocksize_ok = (Kern5_Ni + 1u) < (uint32_t) prop.maxThreadsDim[1];
        bool next_nthreads_ok  = (Kern5_Ni + 1u) * Kld <= (uint32_t) prop.maxThreadsPerBlock;
        if(!next_smem_ok || !next_reg_ok || !next_blocksize_ok || !next_nthreads_ok){
            printf(" %u %u %u %u\n", next_smem_ok, next_reg_ok, next_blocksize_ok, next_nthreads_ok);
            break;}
        
        Kern5_Ni++;
    }
    uint32_t smem_N_floats_Kern5 = Kern5_Ni * Mld + Kern5_Ni * Kld * Mld;
    bool smem_ok_Kern5 = smem_N_floats_Kern5 <= target_n_floats_smem;
    bool blocksize_ok_Kern5 = (Kern5_Ni) <= (uint32_t) prop.maxThreadsDim[1];
    nthreads_ok = (Kern5_Ni) * Kld <= (uint32_t) prop.maxThreadsPerBlock;
    if(!smem_ok_Kern5 || !blocksize_ok_Kern5 || !nthreads_ok){
        dying_breath("could not find a suitable block size for the kernel 5");}
    printf("\nblock shapes for kernel 5: (%u, %u)\n", Kld, Kern5_Ni);
    printf("memory usage and maxima for kernel 5: smem_N_floats, %u target_n_floats_smem, %u  \n", smem_N_floats_Kern5, target_n_floats_smem);
    printf("number of threads per block: %u\n", Kld* Kern5_Ni);
    printf("\n");
    printf("\n");
    printf("\n");
}

// 1: gradient descent: fill momenta_attraction, momenta_repulsion_far, momenta_repulsion
// 2: this also recomputes the furthest_neighdists_LD
void fill_nudges_GPU(EmbeddingMaker_GPU* thing){
    printf("ajouter -fastmath et cie\n");
    printf("liste des optimisations: -ffast-math  -funsafe-math-optimizations -ffinite-math-only -fno-trapping-math -fassociative-math -freciprocal-math -fmerge-all-constants -Ofast  (YOLO)\n");

    // get the alpha hyperparameter, for the simplified Cauchy kernel
    pthread_mutex_lock(thing->mutex_hparam_LDkernel_alpha);
    float cauchy_alpha = thing->hparam_LDkernel_alpha[0];
    pthread_mutex_unlock(thing->mutex_hparam_LDkernel_alpha);

    // ----------- 1: determine which momtenum will be leaked, which will be source -----------
    float* mmtm_src;
    float* mmtm_rcv;
    if(thing->leak_phase){
        mmtm_src = thing->cu_momenta_repuls_far___0;
        mmtm_rcv = thing->cu_momenta_repuls_far___1;
    } else {
        mmtm_src = thing->cu_momenta_repuls_far___1;
        mmtm_rcv = thing->cu_momenta_repuls_far___0;
    }

    // ----------- 2: gradient descent: fill nudge_attraction, nudge_repulsion_far, nudge_repulsion -----------
    float sum_Qdenom_elements_cpu = 0.0f;
    cuda_launch___fill_nudges_and_leak(thing->stream_nudge_HD, thing->stream_nudge_LD, thing->stream_nudge_FAR, thing->stream_Qdenomsum, thing->stream_leak,\
        thing->Kern_HD_blockshape, thing->Kern_HD_gridshape, thing->Kern_LD_blockshape, thing->Kern_LD_gridshape, thing->Kern_FAR_blockshape, thing->Kern_FAR_gridshape, thing->Kern_Qdenomsum_blockshape, thing->Kern_Qdenomsum_gridshape, thing->Kern_leak_blockshape, thing->Kern_leak_gridshape,\
         thing->N, thing->Khd, thing->cu_P,\
         thing->cu_Xld_nesterov, thing->cu_neighsHD, thing->cu_neighsLD, thing->cu_furthest_neighdists_LD, thing->Qdenom_EMA,\
          cauchy_alpha, thing->cu_elements_of_Qdenom, thing->cu_sum_Qdenom_elements, &sum_Qdenom_elements_cpu, thing->N_elements_of_Qdenom,\
           thing->cu_nudge_attrac_HD, thing->cu_nudge_repuls_HDLD, thing->cu_nudge_FAR, thing->cu_temporary_furthest_neighdists_LD,\
            thing->cu_random_numbers_size_NxRand,\
            mmtm_src, mmtm_rcv);
    
    // ----------- 3: update EMA of Qdenom -----------
    // the estimation of Qdenom for this iteration
    uint32_t n_samples_estim = thing->N * (thing->Khd + Kld + NB_RANDOM_POINTS_FAR_REPULSION);
    uint32_t matrix_area = thing->N * (thing->N-1u);
    float scaling_factor = (float) ((double) matrix_area / (double) n_samples_estim);
    float new_Qdenom = sum_Qdenom_elements_cpu * scaling_factor;
    // update the EMA of Qdenom
    // thing->Qdenom_EMA = 0.9f * thing->Qdenom_EMA + 0.1f * new_Qdenom;
    thing->Qdenom_EMA = new_Qdenom;
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
    memcpy_CUDA_to_CPU_float(as_float_1d(thing->Xld_cpu, thing->N, Mld), thing->cu_Xld_base, thing->N*Mld);
    // furthest_neighdists_LD: GPU to CPU
    memcpy_CUDA_to_CPU_float(thing->furthest_neighdists_LD_cpu, thing->cu_furthest_neighdists_LD, thing->N);
}


// this function receives the neighs and P from the CPU, in a SAFE manner
//  Read if ready, then request sync
static void receive_neighs_and_P_from_CPU(EmbeddingMaker_GPU* thing){
    // 1) neighsLD: CPU to GPU.
    GPU_CPU_sync* sync_neigh_LD = &thing->GPU_CPU_comms_neighsLD->sync;

    if(is_ready_now(sync_neigh_LD)){
        pthread_mutex_lock(sync_neigh_LD->mutex_buffer);
        memcpy_CPU_to_CUDA_uint32(thing->cu_neighsLD, thing->GPU_CPU_comms_neighsLD->buffer, thing->N*Kld);
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
        memcpy_CPU_to_CUDA_uint32(thing->cu_neighsHD, thing->GPU_CPU_comms_neighsHD->buffer, thing->N*thing->Khd);
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
        memcpy_CPU_to_CUDA_float(thing->cu_P, thing->GPU_CPU_comms_P->buffer, thing->N*thing->Khd);
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
        // ~~~~~~~~~~ move points around in the embedding ~~~~~~~~~~
        // gradient descent: nudge things around a little bit, on the GPU 
        fill_nudges_GPU(thing);

        // momentum leak: momenta_repulsion_far gets smoothed across neighbours (with conservation of movement)
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

        // ~~~~~~~~~~ phase toggle for momentum leaks ~~~~~~~~~~
        thing->leak_phase = (bool) (1 - thing->leak_phase);
        printf("leak phase: %d\n", thing->leak_phase);
    }
    return NULL; 
}



/*
sous-poudrer le tout avec des gradients de MDS, qu'on peut leak aussi
*/

// thing->estimated_Qdenom = (float) (dbl_acc_denom * ( ((double) (thing->N*thing->N - thing->N)) / (double) n_votes));
// thing->ptr_Qdenom[0] = thing->ptr_Qdenom[0]*ALPHA_QDENOM  + (1.0f - ALPHA_QDENOM) * subthread_estimation_of_denom;

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