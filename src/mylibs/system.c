#include "system.h"

void set_console_colour(uint8_t r, uint8_t g, uint8_t b) {
    printf("\e[38;2;%d;%d;%dm", r, g, b);
}


void test_speed_flt_vs_dbl_no_cache_effects(){
    float float_scalar   = 1.f;
    double double_scalar = 1.;
    clock_t start = clock();
    for(int rep = 0; rep < 3; rep++) {
        float_scalar = 1.f;
         for (int i = 0; i < 100000000; i++) {
            float_scalar *= 1.0000001f;
        }
    }
    double float_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    start = clock();
    for (int rep = 0; rep < 3; rep++) {
        double_scalar = 1.;
        for (int i = 0; i < 100000000; i++) {
            double_scalar *= 1.0000001;
        }
    }
    double double_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("\nspeed test float vs double without cache effects :  float %f    %f double\n", float_time, double_time);
    printf(" values:   float %f %f double\n", float_scalar, double_scalar);
}
//compares the time taken to multiply floats and doubles, with cache effects
void test_speed_flt_vs_dbl_yes_cache_effects(){
    int Nrep = 15;
    size_t size = 500000;
    // on the stack
    float float_array[size];
    for (size_t i = 0; i < size; i++) {
        float_array[i] = 1.f + (float)i/size;
    }
    clock_t start = clock();
    for (int rep = 0; rep < Nrep; rep++) {
        for (size_t i = 0; i < size; i++) {
            float_array[i] = float_array[i] * 1.0000001f;
        }
    }
    double float_time = (double)(clock() - start) / CLOCKS_PER_SEC;

    double double_array[size];
    for (size_t i = 0; i < size; i++) {
        double_array[i] = 1. + (double)i/size;
    }
    start = clock();
    for (int rep = 0; rep < Nrep; rep++) {
        for (size_t i = 0; i < size; i++) {
            double_array[i] = double_array[i] * 1.0000001;
        }
    }
    double double_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("\nspeed test float vs double with cache effects : (on the stack)  float %f    %f double\n", float_time, double_time);
    printf(" values:   float %f %f double\n", float_array[size-1], double_array[size-1]);

    // on the heap
    float* float_array_heap = malloc(size*sizeof(float));
    if (float_array_heap == NULL) {
        printf("Failed to allocate memory for float_array_heap\n");
        return;
    }
    for (size_t i = 0; i < size; i++) {
        float_array_heap[i] = 1.f + (float)i/size;
    }
    start = clock();
    for (int rep = 0; rep < Nrep; rep++) {
        for (size_t i = 0; i < size; i++) {
            float_array_heap[i] = float_array_heap[i] * 1.0000001f;
        }
    }
    float_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    double* double_array_heap = malloc(size*sizeof(double));
    if (double_array_heap == NULL) {
        printf("Failed to allocate memory for double_array_heap\n");
        free(float_array_heap);
        return;
    }
    for (size_t i = 0; i < size; i++) {
        double_array_heap[i] = 1. + (double)i/size;
    }
    start = clock();
    for (int rep = 0; rep < Nrep; rep++) {
        for (size_t i = 0; i < size; i++) {
            double_array_heap[i] = double_array_heap[i] * 1.0000001;
        }
    }
    double_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("\nspeed test float vs double with cache effects : (on the heap)  float %f    %f double\n", float_time, double_time);
    printf(" values:   float %f %f double\n", float_array_heap[size-1], double_array_heap[size-1]);


    // on heap, but this time calls the 1. / x function
    // 1 reset the array values
    for (size_t i = 0; i < size; i++) {
        float_array_heap[i]  = 1.f + (float)i/size;
        double_array_heap[i] = 1. + (double)i/size;
    }
    // 2 time the function calls
    start = clock();
    for (int rep = 0; rep < Nrep; rep++) {
        for (size_t i = 0; i < size; i++) {
            float_array_heap[i] = 1.f / float_array_heap[i];
        }
    }
    float_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    start = clock();
    for (int rep = 0; rep < Nrep; rep++) {
        for (size_t i = 0; i < size; i++) {
            double_array_heap[i] = 1. / double_array_heap[i];
        }
    }
    double_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("\nspeed test float vs double with cache effects : (on the heap) 1/x  float %f    %f double\n", float_time, double_time);
    printf(" values:   float %f %f double\n", float_array_heap[size-1], double_array_heap[size-1]);

    // on heap, but this time calls the exp() function on -x
    // 1 reset the array values
    for (size_t i = 0; i < size; i++) {
        float_array_heap[i]  = 1.f + (float)i/size;
        double_array_heap[i] = 1. + (double)i/size;
    }
    // 2 time the function calls
    start = clock();
    for (int rep = 0; rep < Nrep; rep++) {
        for (size_t i = 0; i < size; i++) {
            float_array_heap[i] = expf(-float_array_heap[i]);
        }
    }
    float_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    start = clock();
    for (int rep = 0; rep < Nrep; rep++) {
        for (size_t i = 0; i < size; i++) {
            double_array_heap[i] = exp(-double_array_heap[i]);
        }
    }
    double_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("\nspeed test float vs double with cache effects : (on the heap) exp(-x)  float %f    %f double\n", float_time, double_time);
    printf(" values:   float %f %f double\n", float_array_heap[size-1], double_array_heap[size-1]);


    free(float_array_heap);
    free(double_array_heap);
}

// faire mon fast tSNE avec

void print_system_info(){
    printf("This computer has %d cores\n", omp_get_num_procs());
    printf("This computer can run %d threads\n", omp_get_max_threads());
    int nthreads = omp_get_max_threads();
    printf("This computer can run %d threads\n", nthreads);

    /*
    TODO: install CUDA and GPU
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    printf("This computer has %d GPUs\n", num_devices);
    for (int i = 0; i < num_devices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("GPU %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %lu bytes\n", prop.totalGlobalMem);
        printf("  Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Maximum block dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Maximum grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Memory clock rate: %d kHz\n", prop.memoryClockRate);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("  Peak memory bandwidth: %f GB/s\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
    */

    // int nthreads = omp_get_max_threads();
    // printf("This computer can run %d threads\n", nthreads);
}