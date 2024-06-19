#include "probabilities.h"

// UNSAFE for sensitive applications: that's a shitty (but fast) rand generator
// careful, if the period is high when doing (r%period), the result will be biased (for instance if the rand did number between 0 and 4, and you do r%3, you'll get 0 twice as often as 1 or 2)
// 2^32 is a bit more than 4 billion, so be careful when getting close to that period (either re-generate or use a 64 bit generator)
inline uint32_t rand_uint32(uint32_t* rand_state) {
    *rand_state ^= *rand_state << 13u;
    *rand_state ^= *rand_state >> 17u;
    *rand_state ^= *rand_state << 5u;

    // If state is zero, reseed it with the current time
    /* if (*rand_state == 0u) {
        *rand_state = (uint32_t)time(NULL);
        dying_breath("congrats you just broke binary arithmetics");
    } */

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
