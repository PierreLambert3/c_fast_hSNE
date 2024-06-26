
// protect this header file
#ifndef CONSTANTS_GLOBAL_H
#define CONSTANTS_GLOBAL_H
// #include "includes_global.h"

/*

TODO:
refactor constans to declare them in the header file and define them in the source file
as "const <type> constName = value"
using define is very very stupid
 */


// determines if using the GPU or CPU for gradient computations
#define USE_GPU true

// number of neighs to consider in LD
#define Kld 16u  
// #define Kld 32u*10u  

// how much we leak the far relationships to nearby points (1 = full leak, 0 = no leak)
#define LEAK_ALPHA 0.5f

// target dimensionality of the embedding
#define Mld 2u

// pct bias in the HD neighbours
#define HD_PCT_BIAS 0.02f

// momentum decay
#define MOMENTUM_ALPHA 0.25f

// base LR
#define BASE_LR 0.1f

// a global epsilon for floating points equal to 1e-16
#define FLOAT_EPS 1e-12f

// size of chunks for subthreads, as a percentage of N
#define SUBTHREADS_CHUNK_SIZE_PCT 0.05f

// boolean that determines whether a custom fatser kernel should be used for q_ij
#define USE_CUSTOM_QIJ_KERNEL false

// speed at which the Q denominator is updated
#define ALPHA_QDENOM (0.95f + ((1.f - 0.95f) * (1.f - SUBTHREADS_CHUNK_SIZE_PCT)))

#define NB_RANDOM_POINTS_FAR_REPULSION 32u // was 32u in original prototype

// the number of random points randomly sampled during neighbour discovery, on all the dataset
#define NEIGH_FAR_EXPLORATION_N_SAMPLES      142u
#define NEIGH_NEAR_EXPLOITATION_LD_N_SAMPLES 142u // keep higher than 12u
#define NEIGH_NEAR_EXPLOITATION_HD_N_SAMPLES 142u // same, for neighbours of neighbours and neighbours in other spaces

// how often (a double, in seconds) the GUI and CPU threads should synchronise variables
#define GUI_CPU_SYNC_PERIOD 0.5

// int value for window size of the GUI
#define GUI_W 1050
// 0.75 * GUI_W
#define GUI_H (int) (0.68f * (float)GUI_W)

#define GUI_MS_UPDATE_QDENOM   50

// a set of 3 uint8_t values representing base terminal text colour
#define TERMINAL_TEXT_COLOUR_R 220
#define TERMINAL_TEXT_COLOUR_G 130
#define TERMINAL_TEXT_COLOUR_B 20
// a set of 3 uint8_t values representing error terminal text colour
#define TERMINAL_ERROR_COLOUR_R 255
#define TERMINAL_ERROR_COLOUR_G 130
#define TERMINAL_ERROR_COLOUR_B 220
// a set of 3 uint8_t values representing success terminal text colour
#define TERMINAL_SUCCESS_COLOUR_R 80
#define TERMINAL_SUCCESS_COLOUR_G 255
#define TERMINAL_SUCCESS_COLOUR_B 20

#endif // CONSTANTS_GLOBAL_H
