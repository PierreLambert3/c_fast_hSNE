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

// ------------------- output management -------------------
// sets the console colour to the specified RGB values
void set_console_colour(uint8_t r, uint8_t g, uint8_t b);
void set_console_colour_error();
void set_console_colour_success();
void reset_console_colour();


// ------------------- memory allocation -------------------
pthread_mutex_t* mutexes_allocate_and_init(uint32_t size);
pthread_mutex_t* mutex_allocate_and_init();
// malloc handlers for 1d arrays
bool*      malloc_bool(uint32_t size, bool init_val);
float*     malloc_float(uint32_t size, float init_val);
double*    malloc_double(uint32_t size, double init_val);
uint32_t*  malloc_uint32_t(uint32_t size, uint32_t init_val);
uint16_t*  malloc_uint16_t(uint32_t size, uint16_t init_val);
uint8_t*   malloc_uint8_t(uint32_t size, uint8_t init_val);
// malloc handlers for 2d matrices
bool**     malloc_bool_matrix(uint32_t n, uint32_t m, bool init_val);
float**    malloc_float_matrix(uint32_t n, uint32_t m, float init_val);
double**   malloc_double_matrix(uint32_t n, uint32_t m, double init_val);
uint32_t** malloc_uint32_t_matrix(uint32_t n, uint32_t m, uint32_t init_val);
uint16_t** malloc_uint16_t_matrix(uint32_t n, uint32_t m, uint16_t init_val);
uint8_t**  malloc_uint8_t_matrix(uint32_t n, uint32_t m, uint8_t init_val);
// matrix shape handlers
float*     as_float_1d(float** matrix, uint32_t n, uint32_t m);
// matrix copy functions
void       memcpy_float_matrix(float** recipient, float** original, uint32_t n, uint32_t m);
// ... more to come

// free handlers for mallocs
void       free_matrix(void** matrix, uint32_t n);
void       free_array(void* array);



// ------------------- info on the system -------------------
//compares the time taken to multiply floats and doubles, without cache effects
void test_speed_flt_vs_dbl_no_cache_effects();
//compares the time taken to multiply floats and doubles, with cache effects
void test_speed_flt_vs_dbl_yes_cache_effects();
// prints info on the hardware and software of the system
void print_system_info();

#endif // SYSTEM_H
