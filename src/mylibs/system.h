#ifndef SYSTEM_H
#define SYSTEM_H

#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void dying_breath(const char* message);
void die();
void sleep_ms(uint32_t n_ms);
double time_seconds();

/***
 *                   _                   _        __      
 *                  | |                 (_)      / _|     
 *     ___ _   _ ___| |_ ___ _ __ ___    _ _ __ | |_ ___  
 *    / __| | | / __| __/ _ \ '_ ` _ \  | | '_ \|  _/ _ \ 
 *    \__ \ |_| \__ \ ||  __/ | | | | | | | | | | || (_) |
 *    |___/\__, |___/\__\___|_| |_| |_| |_|_| |_|_| \___/ 
 *          __/ |                                         
 *         |___/                                          
 */
//compares the time taken to multiply floats and doubles, without cache effects
void test_speed_flt_vs_dbl_no_cache_effects();
//compares the time taken to multiply floats and doubles, with cache effects
void test_speed_flt_vs_dbl_yes_cache_effects();
// prints info on the hardware and software of the system
void print_system_info();

/***
 *                 _               _                                                                 _   
 *                | |             | |                                                               | |  
 *      ___  _   _| |_ _ __  _   _| |_   _ __ ___   __ _ _ __   __ _  __ _  ___ _ __ ___   ___ _ __ | |_ 
 *     / _ \| | | | __| '_ \| | | | __| | '_ ` _ \ / _` | '_ \ / _` |/ _` |/ _ \ '_ ` _ \ / _ \ '_ \| __|
 *    | (_) | |_| | |_| |_) | |_| | |_  | | | | | | (_| | | | | (_| | (_| |  __/ | | | | |  __/ | | | |_ 
 *     \___/ \__,_|\__| .__/ \__,_|\__| |_| |_| |_|\__,_|_| |_|\__,_|\__, |\___|_| |_| |_|\___|_| |_|\__|
 *                    | |                                             __/ |                              
 *                    |_|                                            |___/                               
 */
// sets the console colour to the specified RGB values
void set_console_colour(uint8_t r, uint8_t g, uint8_t b);
void set_console_colour_error();
void set_console_colour_success();
void reset_console_colour();

/***
 *                                               
 *                                               
 *     _ __ ___   ___ _ __ ___   ___  _ __ _   _ 
 *    | '_ ` _ \ / _ \ '_ ` _ \ / _ \| '__| | | |
 *    | | | | | |  __/ | | | | | (_) | |  | |_| |
 *    |_| |_| |_|\___|_| |_| |_|\___/|_|   \__, |
 *                                          __/ |
 *                                         |___/ 
 */
pthread_mutex_t* mutexes_allocate_and_init(uint32_t n_elements);
pthread_mutex_t* mutex_allocate_and_init();
// malloc handlers for 1d arrays
bool*      malloc_bool(uint32_t n_elements, bool init_val);
float*     malloc_float(uint32_t n_elements, float init_val);
double*    malloc_double(uint32_t n_elements, double init_val);
uint32_t*  malloc_uint32_t(uint32_t n_elements, uint32_t init_val);
uint16_t*  malloc_uint16_t(uint32_t n_elements, uint16_t init_val);
uint8_t*   malloc_uint8_t(uint32_t n_elements, uint8_t init_val);
// malloc handlers for 2d matrices
bool**     malloc_bool_matrix(uint32_t n, uint32_t m, bool init_val);
float**    malloc_float_matrix(uint32_t n, uint32_t m, float init_val);
double**   malloc_double_matrix(uint32_t n, uint32_t m, double init_val);
uint32_t** malloc_uint32_t_matrix(uint32_t n, uint32_t m, uint32_t init_val);
uint16_t** malloc_uint16_t_matrix(uint32_t n, uint32_t m, uint16_t init_val);
uint8_t**  malloc_uint8_t_matrix(uint32_t n, uint32_t m, uint8_t init_val);
// matrix shape handlers
float*     as_float_1d(float** matrix, uint32_t n, uint32_t m);
uint32_t*     as_uint32_1d(uint32_t** matrix, uint32_t n, uint32_t m);
// matrix copy functions
void       memcpy_float_matrix(float** recipient, float** original, uint32_t n, uint32_t m);
// ... more to come

// free handlers for mallocs
void       free_matrix(void** matrix, uint32_t n);
void       free_array(void* array);

/***
 *       ____ _   _ ____    _                                          _ 
 *      / ___| | | |  _ \  / \     _    __ _  ___ _ __   ___ _ __ __ _| |
 *     | |   | | | | | | |/ _ \   (_)  / _` |/ _ \ '_ \ / _ \ '__/ _` | |
 *     | |___| |_| | |_| / ___ \   _  | (_| |  __/ | | |  __/ | | (_| | |
 *      \____|\___/|____/_/   \_\ (_)  \__, |\___|_| |_|\___|_|  \__,_|_|
 *                                     |___/                             
 */

struct cudaDeviceProp initialise_cuda();
void print_cuda_device_info(struct cudaDeviceProp prop);
void malloc_1d_float_cuda(float** ptr_array_GPU, uint32_t n_elements);
void malloc_1d_uint32_cuda(uint32_t** ptr_array_GPU, uint32_t n_elements);
void memcpy_CPU_to_CUDA_float(float* ptr_array_GPU, float* ptr_array_CPU, uint32_t n_elements);
void memcpy_CPU_to_CUDA_uint32(uint32_t* ptr_array_GPU, uint32_t* ptr_array_CPU, uint32_t n_elements);
void memcpy_CUDA_to_CPU_float(float* ptr_array_CPU, float* ptr_array_GPU, uint32_t n_elements);
void memcpy_CUDA_to_CPU_uint32(uint32_t* ptr_array_CPU, uint32_t* ptr_array_GPU, uint32_t n_elements);

/***
 *       ____ _   _ ____    _           ____ ____  _   _   ______ ____  _   _                                     
 *      / ___| | | |  _ \  / \     _   / ___|  _ \| | | | / / ___|  _ \| | | |   ___ ___  _ __ ___  _ __ ___  ___ 
 *     | |   | | | | | | |/ _ \   (_) | |   | |_) | | | |/ / |  _| |_) | | | |  / __/ _ \| '_ ` _ \| '_ ` _ \/ __|
 *     | |___| |_| | |_| / ___ \   _  | |___|  __/| |_| / /| |_| |  __/| |_| | | (_| (_) | | | | | | | | | | \__ \
 *      \____|\___/|____/_/   \_\ (_)  \____|_|    \___/_/  \____|_|    \___/   \___\___/|_| |_| |_|_| |_| |_|___/
 *                                                                                                                
 */

/*
Example uses:


CPU ------>   BUFFER
OJO!  need to represent as 1d array

GPU_CPU_sync* sync_neighsHD = &thing->GPU_CPU_comms_neighsHD->sync;
if(is_requesting_now(sync_neighsHD) && !is_ready_now(sync_neighsHD)){
    // wait for the subthreads to finish
    wait_full_path_finished(thing);
    // copy the neighsHD to the buffer, safely
    pthread_mutex_lock(thing->GPU_CPU_comms_neighsHD->sync.mutex_buffer);
    memcpy(thing->GPU_CPU_comms_neighsHD->buffer, as_uint32_1d(thing->neighsHD, thing->N, thing->Khd), thing->N*thing->Khd*sizeof(uint32_t));
    pthread_mutex_unlock(thing->GPU_CPU_comms_neighsHD->sync.mutex_buffer);
    // notify the GPU that the data is ready
    notify_ready(sync_neighsHD);
}



BUFFER ------> GPU
OJO!  - use cudaMemcpy() , not memcpy()  (ffs... I'm dumb)
      - dont mix up cudaMemcpyDeviceToHost and cudaMemcpyHostToDevice

GPU_CPU_sync* sync = &thing->GPU_CPU_comms_neighsLD->sync;
if(is_ready_now(sync)){
    pthread_mutex_lock(sync->mutex_buffer);
    cudaMemcpy(thing->neighsLD_cuda, thing->GPU_CPU_comms_neighsLD->buffer, thing->N*thing->Kld*sizeof(uint32_t), cudaMemcpyHostToDevice);
    pthread_mutex_unlock(sync->mutex_buffer);
    set_ready(sync, false);
}
if(!is_requesting_now(sync)){
    notify_request(sync); // request for the next sync
}
*/

// a struct that contains flags and mutexes for synchronisation
typedef struct {
    // request a buffer update
    pthread_mutex_t* mutex_request;
    bool             flag_request;
    // notify that the buffer has been updated 
    pthread_mutex_t* mutex_ready;
    bool             flag_ready;
    // mutex for the buffer itself
    pthread_mutex_t* mutex_buffer;
} GPU_CPU_sync;

// a struct that contains a float buffer
typedef struct {
    GPU_CPU_sync     sync;
    float*           buffer;
} GPU_CPU_float_buffer;

// a struct that contains a uint32_t buffer
typedef struct {
    GPU_CPU_sync     sync;
    uint32_t*        buffer;
} GPU_CPU_uint32_buffer;

GPU_CPU_float_buffer*  malloc_GPU_CPU_float_buffer(uint32_t size);
GPU_CPU_uint32_buffer* malloc_GPU_CPU_uint32_buffer(uint32_t size);
bool is_requesting_now(GPU_CPU_sync* sync);
bool is_ready_now(GPU_CPU_sync* sync);
void notify_ready(GPU_CPU_sync* sync);
void notify_request(GPU_CPU_sync* sync);

void set_ready(GPU_CPU_sync* sync, bool value);
void set_request(GPU_CPU_sync* sync, bool value);

#endif // SYSTEM_H
