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
pthread_mutex_t* mutexes_allocate_and_init(size_t size);
// malloc handlers for 1d arrays
bool*      malloc_bool(size_t size, bool init_val);
float*     malloc_float(size_t size, float init_val);
double*    malloc_double(size_t size, double init_val);
uint32_t*  malloc_uint32_t(size_t size, uint32_t init_val);
uint16_t*  malloc_uint16_t(size_t size, uint16_t init_val);
uint8_t*   malloc_uint8_t(size_t size, uint8_t init_val);
// malloc handlers for 2d matrices
bool**     malloc_bool_matrix(size_t n, size_t m, bool init_val);
float**    malloc_float_matrix(size_t n, size_t m, float init_val);
double**   malloc_double_matrix(size_t n, size_t m, double init_val);
uint32_t** malloc_uint32_t_matrix(size_t n, size_t m, uint32_t init_val);
uint16_t** malloc_uint16_t_matrix(size_t n, size_t m, uint16_t init_val);
uint8_t**  malloc_uint8_t_matrix(size_t n, size_t m, uint8_t init_val);
// free handlers for mallocs
void       free_matrix(void** matrix);
void       free_array(void* array);



// ------------------- info on the system -------------------
//compares the time taken to multiply floats and doubles, without cache effects
void test_speed_flt_vs_dbl_no_cache_effects();
//compares the time taken to multiply floats and doubles, with cache effects
void test_speed_flt_vs_dbl_yes_cache_effects();
// prints info on the hardware and software of the system
void print_system_info();

#endif // SYSTEM_H
