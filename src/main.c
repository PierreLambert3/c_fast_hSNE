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
*/

void init_Xld(float** Xhd, float** Xld, uint32_t N, uint32_t Mhd, uint32_t Mld) {
    uint32_t rand_state = (uint32_t)time(NULL);
    // initialise a random projection matrix Wproj of size Mhd x Mld, on the stack
    float Wproj[Mhd][Mld];
    for (uint32_t i = 0; i < Mhd; i++) {
        for (uint32_t j = 0; j < Mld; j++) {
            if(rand_uint32_between(&rand_state, 0, 2) == 0){
                Wproj[i][j] = rand_float_between(&rand_state, -0.5f, 1.0f);
            }
            else {
                Wproj[i][j] = rand_float_between(&rand_state, -0.2f, 0.02f);
            }
        }
    }

    // project Xhd onto Xld
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < Mld; j++) {
            Xld[i][j] = 0.0f;
            for (uint32_t u = 0; u < Mhd; u++) {
                Xld[i][j] += Xhd[i][u] * Wproj[u][j];
            }
        }
    }
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




int main() {
    // ~~~~~  general startup  ~~~~~
    reset_console_colour();
    printf("starting program...\n");
    print_system_info();
    uint32_t machine_nb_processors = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
    // seed the random number generator for the threads
    uint32_t rand_state_main_thread = (uint32_t)time(NULL);

    

    // ~~~~~  initialise the common variables for the threads  ~~~~~
    const uint32_t  Mld = 2;
    float     target_perplexity = 50.0f;
    float     LD_kernel_alpha   = 1.0f;
    pthread_mutex_t mutex_kernel_LD_alpha;
    if(pthread_mutex_init(&mutex_kernel_LD_alpha, NULL) != 0) {
        dying_breath("pthread_mutex_init mutex_kernel_LD_alpha failed");}
    uint32_t  Khd = (uint32_t)roundf(3.0f * target_perplexity);
    uint32_t  Kld = 10;
    float     momentum_alpha    = 0.95f; // TODO : modulated by temporal alignment
    float     nesterov_alpha    = 0.05f;
    const uint32_t  n_threads_HDneigh   = 1;
    const uint32_t  n_threads_LDneigh   = 4;
    const uint32_t  n_threads_embedding = (machine_nb_processors - (1+1+n_threads_HDneigh+n_threads_LDneigh)) < n_threads_LDneigh ? n_threads_LDneigh : (machine_nb_processors - (1+1+n_threads_HDneigh+n_threads_LDneigh));
    printf("\n\nnb of threads for each role: HDneigh=%d, LDneigh=%d, embedding=%d, SDL=%d", n_threads_HDneigh, n_threads_LDneigh, n_threads_embedding, 1);
    
    // 1: load & normalise the MNIST dataset
    printf("\nloading MNIST dataset...\n");
    uint32_t  N = 60000;
    uint32_t  Mhd = 28*28;
    uint32_t* Y   = malloc_uint32_t(N, 0);
    float**   Xhd = malloc_float_matrix(N, Mhd, -42.0f);
    load_mnist(&N, &Mhd, Xhd, Y);
    printf("normalising MNIST dataset...\n");
    normalise_float_matrix(Xhd, N, Mhd); 

    /* ok cest mieux quand je normalise pas.
     dans l ordre : 
     1/  trouver pk ca prend du temps avant de runm: probablement un sleep 
     2/  colorer avec les Y: faire des random colours avec Kmeans 
     3/  voir si , sans normalisation, ca donne un PCA, et si avec, ca donne pareil  */

    printf("allocating internals...\n ");
    // create the mutex for each observation i
    pthread_mutex_t* mutexes_sizeN = mutexes_allocate_and_init(N);
    
    // 3: initialise the LD representation of the dataset as a random projection of Xhd
    float**    Xld = malloc_float_matrix(N, Mld, -41.0f);
    init_Xld(Xhd, Xld, N, Mhd, Mld);

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

    // 6: allocate Q and P matrices, the HD radii, as well as Q_denom scalar
    float**    Q       = malloc_float_matrix(N, Kld, 1.0f);
    float**    P       = malloc_float_matrix(N, Khd, 1.0f);
    float*     radii   = malloc_float(N, 1.0f);
    float      Q_denom = 1.0f * N * N;
    // initialise the Q_denom mutex
    pthread_mutex_t mutex_Qdenom;
    if(pthread_mutex_init(&mutex_Qdenom, NULL) != 0) {
        dying_breath("pthread_mutex_init mutex_Qdenom failed");}

    printf("initialising workers...\n");
    // create HD neighbourhood discoverer
    NeighHDDiscoverer* neighHD_discoverer = (NeighHDDiscoverer*)malloc(sizeof(NeighHDDiscoverer));
    new_NeighHDDiscoverer(neighHD_discoverer, N, &rand_state_main_thread, n_threads_HDneigh);
    // create LD neighbourhood discoverer
    NeighLDDiscoverer* neighLD_discoverer = (NeighLDDiscoverer*)malloc(sizeof(NeighLDDiscoverer));
    new_NeighLDDiscoverer(neighLD_discoverer, N, &rand_state_main_thread, n_threads_LDneigh,\
        &mutex_Qdenom, mutexes_sizeN, Xld, Xhd, Mld, Khd, Kld, neighsLD, neighsHD, furthest_neighdists_LD, Q, &Q_denom,\
        &LD_kernel_alpha, &mutex_kernel_LD_alpha);


    // create the embedding maker
    EmbeddingMaker* embedding_maker = (EmbeddingMaker*)malloc(sizeof(EmbeddingMaker));
    new_EmbeddingMaker(embedding_maker, N, &rand_state_main_thread, n_threads_embedding);
    // create GUI manager
    GuiManager* gui_manager = (GuiManager*)malloc(sizeof(GuiManager));
    new_GuiManager(gui_manager, N, Y, neighHD_discoverer, neighLD_discoverer, embedding_maker, &rand_state_main_thread);
    // start the GUI manager thread (which will start the HD neighbourhood discoverer thread too)
    printf("run\n");    
    start_thread_GuiManager(gui_manager);
     

    // wait for the GUI thread to finish
    pthread_join(neighHD_discoverer->thread, NULL);
    pthread_join(neighLD_discoverer->thread, NULL);
    pthread_join(embedding_maker->thread, NULL);
    int threadReturnValue;
    SDL_WaitThread(gui_manager->sdl_thread, &threadReturnValue);
    /* // Wait for pthreads to finish
    pthread_join(thing->neighHD_discoverer->thread, NULL);
    pthread_join(thing->neighLD_discoverer->thread, NULL);
    pthread_join(thing->embedding_maker->thread, NULL); */

    /*
    TODO:
    check SIMD instructions to :
    SIMD stands for Single Instruction, Multiple Data. It's a class of parallel computers in Flynn's taxonomy. SIMD describes computers with multiple processing elements that perform the same operation on multiple data points simultaneously.

In the context of CPU architectures, SIMD instructions allow a single operation to be performed on multiple data points at once. For example, if you have an array of integers and you want to add a specific value to each integer, a SIMD instruction could perform these additions in parallel, rather than sequentially.

Most modern CPU architectures, including x86 (with SSE and AVX extensions) and ARM (with NEON extension), support SIMD instructions. These instructions can significantly speed up certain operations, especially in fields like image processing, machine learning, and scientific computing, where the same operation is often performed on large arrays of data.

In C, you can use SIMD instructions either by using compiler intrinsics, which are special functions provided by the compiler that map directly to SIMD instructions, or by using libraries like OpenMP that can automatically vectorize certain loops.
    */

    // ~~~~~  cleanup  ~~~~~
    for (uint32_t i = 0; i < N; i++) {
        pthread_mutex_destroy(&mutexes_sizeN[i]);}
    pthread_mutex_destroy(&mutex_Qdenom);

    set_console_colour_success();
    printf("reached last instruction\n");
    return 0;

}


