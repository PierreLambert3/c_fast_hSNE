#ifndef SYSTEM_H
#define SYSTEM_H

#include <includes_global.h>
#include <time.h>
#include <omp.h>

// ------------------- output -------------------
// sets the console colour to the specified RGB values
void set_console_colour(uint8_t r, uint8_t g, uint8_t b);


// ------------------- memory allocation -------------------



// ------------------- info on the system -------------------
//compares the time taken to multiply floats and doubles, without cache effects
void test_speed_flt_vs_dbl_no_cache_effects();
//compares the time taken to multiply floats and doubles, with cache effects
void test_speed_flt_vs_dbl_yes_cache_effects();
// prints info on the hardware and software of the system
void print_system_info();

#endif // SYSTEM_H
