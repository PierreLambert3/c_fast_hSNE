#include "includes_global.h"
#include "gui_manager.h"
#include "dataset_loaders.h"
#include "pthread.h"
/*
Written in C.

This project implements an accelerated version of t-SNE, in an interactive GUI environment. 
This version of tSNE works as follows:
Both the neighbourhoods in high-dimension (HD) and lower-dimension (LD) are estimated though the iterations. The tSNE iterations are performed alongside the neighbourhood estimation, using the approximated sets.

There are 4 main threads by essence:
1/ A neighbourhood discovery thread for HD neighbours.
2/ A neighbourhood discovery thread for LD neighbours.
3/ A tSNE optimisation thread which:
	- subdivides into smaller CPU threads, cutting sum(i<N) into chunks.
	- sends the gradient computation to the GPU using CUDA
4/ A GUI thread which can modify values stored inside the other threads according to user inputs, and shows Xld on screen.

GUI:  with SDL2
CUDA on the algorithm: optional


SDL2 
also need to install the SDL2_ttf library: sudo apt-get install libsdl2-ttf-dev and makefile change
*/

void gram_schmidt(float** Wproj, uint32_t Mhd) {
    for (uint32_t i = 0; i < Mhd; i++) {
        for (uint32_t j = 0; j < i; j++) {
            float dot = 0.0f;
            for (uint32_t k = 0; k < Mld; k++) {
                dot += Wproj[i][k] * Wproj[j][k];
            }
            for (uint32_t k = 0; k < Mld; k++) {
                Wproj[i][k] -= dot * Wproj[j][k];
            }
        }
        float norm = FLOAT_EPS;
        for (uint32_t k = 0; k < Mld; k++) {
            norm += Wproj[i][k] * Wproj[i][k];
        }
        norm = sqrtf(norm);
        for (uint32_t k = 0; k < Mld; k++) {
            Wproj[i][k] /= norm;
        }
    }
}

void init_Xld(float** Xhd, float** Xld, uint32_t N, uint32_t Mhd) {
    uint32_t rand_state = (uint32_t)time(NULL);
    // initialise a random projection matrix Wproj of size Mhd x Mld, on the stack
    float** Wproj = malloc_float_matrix(Mhd, Mld, 0.0f);
    for (uint32_t i = 0; i < Mhd; i++) {
        for (uint32_t j = 0; j < Mld; j++) {
            Wproj[i][j] = rand_float_between(&rand_state, -1.0f, 1.0f);
        }
    }
    // make it orthogonal
    gram_schmidt(Wproj, Mhd);
    // project Xhd onto Xld
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < Mld; j++) {
            Xld[i][j] = 0.0f;
            for (uint32_t u = 0; u < Mhd; u++) {
                Xld[i][j] += Xhd[i][u] * Wproj[u][j];
            }
        }
    }
    // free proj matrix 
    free_matrix((void**)Wproj, Mhd);
    // normalise Xld to have mean 0 and variance 1
    normalise_float_matrix(Xld, N, Mld);
}
    
void init_neighbours_randomly(uint32_t N, uint32_t M, float** X, uint32_t K, uint32_t** neighs, float* furthest_neighdists) {
    uint32_t rand_state = (uint32_t)time(NULL);
    for (uint32_t i = 0; i < N; i++) {
        float furthest_neighdist = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
            uint32_t j_candidate = rand_uint32_between(&rand_state, 0, N-1);
            while(j_candidate==i) {
                j_candidate = rand_uint32_between(&rand_state, 0, N-1);}
            neighs[i][k] = j_candidate;
            float dist = f_euclidean_sq(X[i], X[j_candidate], M);
            if (dist > furthest_neighdist) {
                furthest_neighdist = dist;}
        }
        furthest_neighdists[i] = furthest_neighdist;
    }
}


/*
  __  __       _       
 |  \/  | __ _(_)_ __  
 | |\/| |/ _` | | '_ \ 
 | |  | | (_| | | | | |
 |_|  |_|\__,_|_|_| |_|
*/
int main() {
    // ~~~~~  general startup  ~~~~~
    reset_console_colour();
    printf("starting program...\n");
    print_system_info();
    uint32_t machine_nb_processors = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
    // seed the random number generator for the threads
    uint32_t rand_state_main_thread = (uint32_t)time(NULL);

    // ~~~~~  initialise the common variables for the threads  ~~~~~
    float     perplexity = 50.0f;
    pthread_mutex_t* mutex_perplexity = mutex_allocate_and_init();
    float     LD_kernel_alpha   = 1.0f;
    pthread_mutex_t* mutex_kernel_LD_alpha = mutex_allocate_and_init();
    uint32_t  Khd = (uint32_t)roundf(3.0f * perplexity);
    float     momentum_alpha    = 0.95f; // TODO : modulated by temporal alignment
    float     nesterov_alpha    = 0.05f;
   /*  uint32_t  n_threads_HDneigh, n_threads_LDneigh, n_threads_embedding;
    if(USE_GPU){
        n_threads_HDneigh   = 1u + (uint32_t) ((float)machine_nb_processors * 0.3);
        n_threads_LDneigh   = 1u + (uint32_t) ((float)machine_nb_processors * 0.7);
        n_threads_embedding = 1u;
    }
    else{
        n_threads_HDneigh   = 1u + (uint32_t) ((float)machine_nb_processors * 0.2);
        n_threads_LDneigh   = 1u + (uint32_t) ((float)machine_nb_processors * 0.3);
        n_threads_embedding = 1u + (uint32_t) ((float)machine_nb_processors * 0.4);
    } */
    if(!USE_GPU){
        dying_breath("CUDA is not implemented yet. \n\
        Once done, don't forget to modify the DYNAMIC LD/HD thread balance accordingly");
    }

    uint32_t  n_threads_HDneigh   = machine_nb_processors;
    uint32_t  n_threads_LDneigh   = machine_nb_processors;
    uint32_t  n_threads_embedding = USE_GPU ? 1u : machine_nb_processors;
    printf("\n\nnb of threads for each role: HDneigh=%d, LDneigh=%d, embedding=%d, SDL=%d", n_threads_HDneigh, n_threads_LDneigh, n_threads_embedding, 1);
    // 1: load & normalise the MNIST dataset
    printf("\nloading MNIST dataset...\n");
    uint32_t  N = 60000;
    // uint32_t  Mhd = 28*28;
    uint32_t  Mhd = 50;
    uint32_t* Y   = malloc_uint32_t(N, 0);
    float**   Xhd = malloc_float_matrix(N, Mhd, -42.0f);
    load_mnist(&N, &Mhd, Xhd, Y);
    printf("normalising MNIST dataset...\n");
    normalise_float_matrix(Xhd, N, Mhd); 
    printf("allocating internals...\n ");
    // create the mutex for each observation i
    pthread_mutex_t* mutexes_sizeN = mutexes_allocate_and_init(N);
    // 3: initialise the LD representation of the dataset as a random projection of Xhd
    float**    Xld = malloc_float_matrix(N, Mld, -41.0f);
    init_Xld(Xhd, Xld, N, Mhd);
    // 4: initialise the Nesterov momentum acceleration parameters
    float**    Xld_momentum = malloc_float_matrix(N, Mld, 0.0f);
    /* float** Xld_EMA_gradalignement = malloc_float_matrix(N, Mld, 1.0f); // TODO : this array will capture how well the recent gradients align with the momentum at their iteration. This is used to modulates the momentum_alpha*/
    float**    Xld_nesterov = malloc_float_matrix(N, Mld, 0.0f);
    float**    Xld_ghost    = malloc_float_matrix(N, Mld, 0.0f);
    memcpy(Xld_nesterov[0], Xld[0], N*Mld*sizeof(float)); 
    memcpy(Xld_ghost[0], Xld[0], N*Mld*sizeof(float)); 
    // 5: allocate for the estimated neighbour sets in both spaces
    uint32_t** neighsHD = malloc_uint32_t_matrix(N, Khd, 0); 
    uint32_t** neighsLD = malloc_uint32_t_matrix(N, Kld, 0);
    float*     furthest_neighdists_HD = malloc_float(N, 0.0f);
    float*     furthest_neighdists_LD = malloc_float(N, 0.0f);
    printf("initialising neighbours in HD...\n");
    init_neighbours_randomly(N, Mhd, Xhd, Khd, neighsHD, furthest_neighdists_HD);
    printf("initialising neighbours in LD...\n");
    init_neighbours_randomly(N, Mld, Xld, Kld, neighsLD, furthest_neighdists_LD);
    // 6:  P matrix, which is shared between embedding_maker and HD_discoverer
    float**    Psym    = malloc_float_matrix(N, Khd, 1.0f);
    // initialise the Q_denom mutex
    pthread_mutex_t* mutex_Qdenom = mutex_allocate_and_init();
    
    // initialise the LD/HD balance mutex
    pthread_mutex_t* mutex_LDHD_balance = mutex_allocate_and_init();
    // initialise the mutex for changing Psym
    pthread_mutex_t* mutex_P = mutex_allocate_and_init();

    printf("initialising workers...\n");
    // create neighbourhood discoverers (in both spaces, LD and HD)
    NeighHDDiscoverer* neighHD_discoverer = (NeighHDDiscoverer*)malloc(sizeof(NeighHDDiscoverer));
    NeighLDDiscoverer* neighLD_discoverer = (NeighLDDiscoverer*)malloc(sizeof(NeighLDDiscoverer));
    new_NeighHDDiscoverer(neighHD_discoverer, N, Mhd, &rand_state_main_thread, n_threads_HDneigh,\
        mutexes_sizeN, Xhd, Khd, neighsHD, neighsLD,\
        furthest_neighdists_HD, Psym,\
        &perplexity, mutex_perplexity, mutex_LDHD_balance, &neighLD_discoverer->pct_new_neighs);
    new_NeighLDDiscoverer(neighLD_discoverer, N, &rand_state_main_thread, n_threads_LDneigh,\
        mutexes_sizeN, Xld, Xhd, Khd, neighsLD, neighsHD, furthest_neighdists_LD,\
        &LD_kernel_alpha, mutex_kernel_LD_alpha, mutex_LDHD_balance, &neighHD_discoverer->pct_new_neighs);

    // create the embedding maker
    EmbeddingMaker* embedding_maker = (EmbeddingMaker*)malloc(sizeof(EmbeddingMaker));
    new_EmbeddingMaker(embedding_maker, N, &rand_state_main_thread, mutexes_sizeN,\
        Xld, Khd, neighsLD, neighsHD, furthest_neighdists_LD,\
        Psym, mutex_P,\
        neighHD_discoverer->GPU_CPU_comms_neighsHD, neighLD_discoverer->GPU_CPU_comms_neighsLD, neighHD_discoverer->GPU_CPU_comms_Psym);
    
    // create & start the GUI, which in term will start the other threads
    GuiManager* gui_manager = (GuiManager*)malloc(sizeof(GuiManager));
    new_GuiManager(gui_manager, N, Y, neighHD_discoverer, neighLD_discoverer, embedding_maker, &rand_state_main_thread);
    start_thread_GuiManager(gui_manager);

    // wait for the GUI thread to finish
    pthread_join(neighHD_discoverer->thread, NULL);
    pthread_join(neighLD_discoverer->thread, NULL);
    pthread_join(embedding_maker->thread, NULL);
    int threadReturnValue;
    SDL_WaitThread(gui_manager->sdl_thread, &threadReturnValue);
    
    /*
    TODO:
    learn about SIMD instructions :
    SIMD stands for Single Instruction, Multiple Data. It's a class of parallel computers in Flynn's taxonomy. SIMD describes computers with multiple processing elements that perform the same operation on multiple data points simultaneously.

In the context of CPU architectures, SIMD instructions allow a single operation to be performed on multiple data points at once. For example, if you have an array of integers and you want to add a specific value to each integer, a SIMD instruction could perform these additions in parallel, rather than sequentially.

Most modern CPU architectures, including x86 (with SSE and AVX extensions) and ARM (with NEON extension), support SIMD instructions. These instructions can significantly speed up certain operations, especially in fields like image processing, machine learning, and scientific computing, where the same operation is often performed on large arrays of data.

In C, you can use SIMD instructions either by using compiler intrinsics, which are special functions provided by the compiler that map directly to SIMD instructions, or by using libraries like OpenMP that can automatically vectorize certain loops.
    */

    // ~~~~~  cleanup  ~~~~~
    for (uint32_t i = 0; i < N; i++) {
        pthread_mutex_destroy(&mutexes_sizeN[i]);}
    pthread_mutex_destroy(mutex_Qdenom);

    set_console_colour_success();
    printf("reached last instruction\n");
    return 0;

}


