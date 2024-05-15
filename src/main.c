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
            Wproj[i][j] = rand_float_between(&rand_state, -1.0f, 1.0f);
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

int main() {
    // ~~~~~  general startup  ~~~~~
    reset_console_colour();
    printf("starting program...\n");
    print_system_info();
    uint32_t machine_nb_processors = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);

    // ~~~~~  initialise SDL  ~~~~~
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init  Error: %s\n", SDL_GetError());
        return 1;
    }

    // ~~~~~  initialise the common variables for the threads  ~~~~~
    uint32_t  Mld = 2;
    float     target_perplexity = 50.0f;
    float     momentum_alpha    = 0.95f; // TODO : modulated by temporal alignment
    float     nesterov_alpha    = 0.05f;
    const uint32_t  n_threads_HDneigh   = 1;
    const uint32_t  n_threads_LDneigh   = 4;
    const uint32_t  n_threads_embedding = (machine_nb_processors - (1+1+n_threads_HDneigh+n_threads_LDneigh)) < n_threads_LDneigh ? n_threads_LDneigh : (machine_nb_processors - (1+1+n_threads_HDneigh+n_threads_LDneigh));
    printf("\n\nnb of threads for each role: HDneigh=%d, LDneigh=%d, embedding=%d, SDL=%d", n_threads_HDneigh, n_threads_LDneigh, n_threads_embedding, 1);
    // 1: load the MNIST dataset
    uint32_t  N = 60000;
    uint32_t  Mhd = 28*28;
    float**   Xhd = float_matrix_initval(N, Mhd, -42.0f);
    uint32_t* Y = uint32_t_array_initval(N, 0);
    load_mnist(&N, &Mhd, Xhd, Y);
    // create the mutex for each observation i
    pthread_mutex_t* mutexes_sizeN = (pthread_mutex_t*) malloc(N * sizeof(pthread_mutex_t));
    if (mutexes_sizeN == NULL) {
        dying_breath("cant do mutex"); }
    for (uint32_t i = 0; i < N; i++) {
        pthread_mutex_init(&mutexes_sizeN[i], NULL);}

    // 2: normalise Xhd to have mean 0 and variance 1 on each dimension
    normalise_float_matrix(Xhd, N, Mhd);

    // 3: initialise the LD representation of the dataset as a random projection of Xhd
    float**    Xld = float_matrix_initval(N, Mld, -41.0f);
    init_Xld(Xhd, Xld, N, Mhd, Mld);

    // 4: initialise the Nesterov momentum acceleration parameters
    float**    Xld_momentum = float_matrix_initval(N, Mld, 0.0f);
    /* float** Xld_EMA_gradalignement = float_matrix_initval(N, Mld, 1.0f); // TODO : this array will capture how well the recent gradients align with the momentum at their iteration. This is used to modulates the momentum_alpha*/
    float**    Xld_nesterov = float_matrix(N, Mld);
    float**    Xld_ghost    = float_matrix(N, Mld);
    memcpy(Xld_nesterov[0], Xld[0], N*Mld*sizeof(float)); 
    memcpy(Xld_ghost[0], Xld[0], N*Mld*sizeof(float)); 

    // 5: allocate for estimated neighbour sets in both spaces
    uint32_t   Khd = (uint32_t)roundf(3.0f * target_perplexity);
    uint32_t   Kld = 10;
    uint32_t** neighsHD = uint32_t_matrix_initval(N, Khd, 0); 
    uint32_t** neighsLD = uint32_t_matrix_initval(N, Kld, 0);
    float*     furthest_neighdists_HD = float_array_initval(N, 0.0f);
    float*     furthest_neighdists_LD = float_array_initval(N, 0.0f);

    // 6: allocate Q and P matrices, the HD radii, as well as Q_denom scalar
    float**    Q       = float_matrix_initval(N, Kld, 1.0f);
    float**    P       = float_matrix_initval(N, Khd, 1.0f);
    float*     radii   = float_array_initval(N, 1.0f);
    float      Q_denom = 1.0f * N * Kld;
    // initialise the Q_denom mutex
    pthread_mutex_t mutex_Qdenom;
    if(pthread_mutex_init(&mutex_Qdenom, NULL) != 0) {
        dying_breath("pthread_mutex_init mutex_Qdenom failed");}

    // seed the random number generator for the threads
    uint32_t rand_state_main_thread = (uint32_t)time(NULL);
    // create HD neighbourhood discoverer
    NeighHDDiscoverer* neighHD_discoverer = new_NeighHDDiscoverer(N, &rand_state_main_thread, n_threads_HDneigh);
    // create LD neighbourhood discoverer
    NeighLDDiscoverer* neighLD_discoverer = new_NeighLDDiscoverer(N, &rand_state_main_thread, n_threads_LDneigh,\
        &mutex_Qdenom, mutexes_sizeN, Xld, Xhd, Mld, Khd, Kld, neighsLD, neighsHD, Q, &Q_denom);


    // create the embedding maker
    EmbeddingMaker* embedding_maker       = new_EmbeddingMaker(N, &rand_state_main_thread, n_threads_embedding);
    // create GUI manager
    GuiManager* gui_manager = new_GuiManager(N, neighHD_discoverer, neighLD_discoverer, embedding_maker, &rand_state_main_thread);
    // start the GUI manager thread (which will start the HD neighbourhood discoverer thread too)
    start_thread_GuiManager(gui_manager);
    // join on SDL thread...
    int threadReturnValue;
    SDL_WaitThread(gui_manager->sdl_thread, &threadReturnValue); 

    // ~~~~~  cleanup  ~~~~~
    for (uint32_t i = 0; i < N; i++) {
        pthread_mutex_destroy(&mutexes_sizeN[i]);}
    pthread_mutex_destroy(&mutex_Qdenom);

    set_console_colour_success();
    printf("reached last instruction\n");
    return 0;

}


