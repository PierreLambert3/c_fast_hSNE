#ifndef PROBABILITIES_H
#define PROBABILITIES_H

#include "system.h"

// gives a random uint32_t value, given a state
uint32_t rand_uint32(uint32_t* state);
uint32_t rand_uint32_between(uint32_t* state, uint32_t min, uint32_t max);

float rand_float(uint32_t* state);
float rand_float_between(uint32_t* state, float min, float max);

#endif // PROBABILITIES_H