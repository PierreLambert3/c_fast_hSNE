#include "probabilities.h"

// UNSAFE for sensitive applications: that's a shitty rand generator (but fast)
inline uint32_t rand_uint32(uint32_t* rand_state) {
    *rand_state ^= *rand_state << 13;
    *rand_state ^= *rand_state >> 17;
    *rand_state ^= *rand_state << 5;
    return *rand_state;
}

// excluding max
inline uint32_t rand_uint32_between(uint32_t* rand_state, uint32_t min, uint32_t max) {
    return min + rand_uint32(rand_state) % (max - min);
}

inline float rand_float(uint32_t* rand_state) {
    return (float)rand_uint32(rand_state) / (float)UINT32_MAX;
}

inline float rand_float_between(uint32_t* rand_state, float min, float max) {
    return min + rand_float(rand_state) * (max - min);
}
