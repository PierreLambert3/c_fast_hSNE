#include "system.h"
#include "constants_global.h"

void dying_breath(const char* message){
    set_console_colour_error();
    printf("\n\n\n%s\n\n\n", message);
    reset_console_colour();
    exit(1);
}

void die(){
    set_console_colour(240, 0, 120);
    printf("\n\n\n die() called \n\n\n");
    reset_console_colour();
    exit(1);
}

void set_console_colour(uint8_t r, uint8_t g, uint8_t b) {
    printf("\e[38;2;%d;%d;%dm", r, g, b);
}

void set_console_colour_error(){
    printf("\e[38;2;%d;%d;%dm", TERMINAL_ERROR_COLOUR_R, TERMINAL_ERROR_COLOUR_G, TERMINAL_ERROR_COLOUR_B);
}

void set_console_colour_success(){
    printf("\e[38;2;%d;%d;%dm", TERMINAL_SUCCESS_COLOUR_R, TERMINAL_SUCCESS_COLOUR_G, TERMINAL_SUCCESS_COLOUR_B);
}

void reset_console_colour(){
    printf("\e[38;2;%d;%d;%dm", TERMINAL_TEXT_COLOUR_R, TERMINAL_TEXT_COLOUR_G, TERMINAL_TEXT_COLOUR_B);
}

// ------------------- memory allocation -------------------
pthread_mutex_t* mutexes_allocate_and_init(size_t size){
    pthread_mutex_t* mutexes = (pthread_mutex_t*)malloc(size * sizeof(pthread_mutex_t));
    if (mutexes == NULL) {
        die("Failed to allocate memory for mutexes");}
    for (size_t i = 0; i < size; i++) {
        if (pthread_mutex_init(&mutexes[i], NULL) != 0) {
            die("Failed to initialise mutex");}
    }
    return mutexes;
}

bool* bool_array(size_t size){
    bool* array = (bool*)malloc(size * sizeof(bool));
    if (array == NULL) {
        die("Failed to allocate memory for bool array");}
    return array;
}

bool* bool_array_initval(size_t size, bool init_val){
    bool* array = bool_array(size);
    for (size_t i = 0; i < size; i++) {
        array[i] = init_val;}
    return array;
}
bool** bool_matrix(size_t n, size_t m){
    bool*  data = (bool*)malloc(n * m * sizeof(bool));
    bool** matrix = (bool**)malloc(n * sizeof(bool*));
    matrix[0] = data;
    for(size_t i = 1; i < n; i++)
        matrix[i] = &data[m * i];
    return matrix;
}

bool** bool_matrix_initval(size_t n, size_t m, bool init_val){
    bool** matrix = bool_matrix(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;
        }
    }
    return matrix;
}


float* float_array(size_t size) {
    float* array = (float*)malloc(size * sizeof(float));
    if (array == NULL) {
        die("Failed to allocate memory for float array");
    }
    return array;
}

double* double_array(size_t size) {
    double* array = (double*)malloc(size * sizeof(double));
    if (array == NULL) {
        die("Failed to allocate memory for double array");
    }
    return array;
}

int* int_array(size_t size) {
    int* array = (int*)malloc(size * sizeof(int));
    if (array == NULL) {
        die("Failed to allocate memory for int array");
    }
    return array;
}

uint8_t* uint8_t_array(size_t size) {
    uint8_t* array = (uint8_t*)malloc(size * sizeof(uint8_t));
    if (array == NULL) {
        die("Failed to allocate memory for uint8_t array");
    }
    return array;
}

uint16_t* uint16_t_array(size_t size) {
    uint16_t* array = (uint16_t*)malloc(size * sizeof(uint16_t));
    if (array == NULL) {
        die("Failed to allocate memory for uint16_t array");
    }
    return array;
}

uint32_t* uint32_t_array(size_t size) {
    uint32_t* array = (uint32_t*)malloc(size * sizeof(uint32_t));
    if (array == NULL) {
        die("Failed to allocate memory for uint32_t array");
    }
    return array;
}

float* float_array_initval(size_t size, float init_val) {
    float* array = float_array(size);
    for (size_t i = 0; i < size; i++) {
        array[i] = init_val;}
    return array;
}

double* double_array_initval(size_t size, double init_val) {
    double* array = double_array(size);
    for (size_t i = 0; i < size; i++) {
        array[i] = init_val;}
    return array;
}

retirer tous les malloc qui n ont pas d init, c est dangereux et ca permet de raccourcir le nom (mettre malloc dans le nouveau nom)
retirer tous les malloc qui n ont pas d init, c est dangereux et ca permet de raccourcir le nom (mettre malloc dans le nouveau nom)
retirer tous les malloc qui n ont pas d init, c est dangereux et ca permet de raccourcir le nom (mettre malloc dans le nouveau nom)
retirer tous les malloc qui n ont pas d init, c est dangereux et ca permet de raccourcir le nom (mettre malloc dans le nouveau nom)


int* int_array_initval(size_t size, int init_val) {
    int* array = int_array(size);
    for (size_t i = 0; i < size; i++) {
        array[i] = init_val;}
    return array;
}

uint8_t* uint8_t_array_initval(size_t size, uint8_t init_val) {
    uint8_t* array = uint8_t_array(size);
    for (size_t i = 0; i < size; i++) {
        array[i] = init_val;}
    return array;   
}

uint16_t* uint16_t_array_initval(size_t size, uint16_t init_val) {
    uint16_t* array = uint16_t_array(size);
    for (size_t i = 0; i < size; i++) {
        array[i] = init_val;}
    return array;
}

uint32_t* uint32_t_array_initval(size_t size, uint32_t init_val) {
    uint32_t* array = uint32_t_array(size);
    for (size_t i = 0; i < size; i++) {
        array[i] = init_val;}
    return array;
}

void free_matrix(void** matrix){
    free(matrix[0]);  // free the block of memory holding the NxM float values
    for (size_t i = 0; i < sizeof(matrix); i++) {
        matrix[i] = NULL;}
    free(matrix); 
}

void free_array(void* array){
    free(array);
}

float** float_matrix(size_t n, size_t m) {
    float* data = (float*)malloc(n * m * sizeof(float));
    float** matrix = (float**)malloc(n * sizeof(float*));
    matrix[0] = data;
    for(size_t i = 1; i < n; i++)
        matrix[i] = &data[m * i];
    return matrix;
}

double**  double_matrix(size_t n, size_t m) {
    double* data = (double*)malloc(n * m * sizeof(double));
    double** matrix = (double**)malloc(n * sizeof(double*));
    matrix[0] = data;
    for(size_t i = 1; i < n; i++)
        matrix[i] = &data[m * i];
    return matrix;
}

int** int_matrix(size_t n, size_t m) {
    int* data = (int*)malloc(n * m * sizeof(int));
    int** matrix = (int**)malloc(n * sizeof(int*));
    matrix[0] = data;
    for(size_t i = 1; i < n; i++)
        matrix[i] = &data[m * i];
    return matrix;
}

uint8_t** uint8_t_matrix(size_t n, size_t m) {
    uint8_t* data = (uint8_t*)malloc(n * m * sizeof(uint8_t));
    uint8_t** matrix = (uint8_t**)malloc(n * sizeof(uint8_t*));
    matrix[0] = data;
    for(size_t i = 1; i < n; i++)
        matrix[i] = &data[m * i];
    return matrix;
}

uint16_t** uint16_t_matrix(size_t n, size_t m) {
    uint16_t* data = (uint16_t*)malloc(n * m * sizeof(uint16_t));
    uint16_t** matrix = (uint16_t**)malloc(n * sizeof(uint16_t*));
    matrix[0] = data;
    for(size_t i = 1; i < n; i++)
        matrix[i] = &data[m * i];
    return matrix;
}

uint32_t** uint32_t_matrix(size_t n, size_t m) {
    uint32_t* data = (uint32_t*)malloc(n * m * sizeof(uint32_t));
    uint32_t** matrix = (uint32_t**)malloc(n * sizeof(uint32_t*));
    matrix[0] = data;
    for(size_t i = 1; i < n; i++)
        matrix[i] = &data[m * i];
    return matrix;
}

float** float_matrix_initval(size_t n, size_t m, float init_val) {
    float** matrix = float_matrix(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;
        }
    }
    return matrix;
}

double** double_matrix_initval(size_t n, size_t m, double init_val) {
    double** matrix = double_matrix(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;
        }
    }
    return matrix;
}

int** int_matrix_initval(size_t n, size_t m, int init_val) {
    int** matrix = int_matrix(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;
        }
    }
    return matrix;
}

uint8_t** uint8_t_matrix_initval(size_t n, size_t m, uint8_t init_val) {
    uint8_t** matrix = uint8_t_matrix(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;
        }
    }
    return matrix;
}

uint16_t** uint16_t_matrix_initval(size_t n, size_t m, uint16_t init_val) {
    uint16_t** matrix = uint16_t_matrix(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;
        }
    }
    return matrix;
}

uint32_t** uint32_t_matrix_initval(size_t n, size_t m, uint32_t init_val) {
    uint32_t** matrix = uint32_t_matrix(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;
        }
    }
    return matrix;
}





void test_speed_flt_vs_dbl_no_cache_effects(){
    printf("\n");
    printf("\n");
    printf("    ---------------- Speed test (no cache) ----------------\n");
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
    printf("\ntime :  float %f    %f double\n", float_time, double_time);
    printf("values :  float %f    %f double\n", float_scalar, double_scalar);
}
//compares the time taken to multiply floats and doubles, with cache effects
void test_speed_flt_vs_dbl_yes_cache_effects(){
    printf("\n");
    printf("\n");
    printf("    ---------------- Speed test (cache) ----------------\n");
    printf("\n");
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
    printf("on the stack:\n");
    printf("\ntime :  float %f    %f double\n", float_time, double_time);
    printf("values :   float %f %f double\n", float_array[size-1], double_array[size-1]);

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
    printf("\non the heap:\n");
    printf("\ntime :  float %f    %f double\n", float_time, double_time);
    printf("values:  float %f %f double\n", float_array_heap[size-1], double_array_heap[size-1]);


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
    printf("\ntime : 1/x  float %f    %f double\n", float_time, double_time);
    printf("values:   float %f %f double\n", float_array_heap[size-1], double_array_heap[size-1]);

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
    printf("\n time : exp(-x)  float %f    %f double\n", float_time, double_time);
    printf(" values:   float %f %f double\n", float_array_heap[size-1], double_array_heap[size-1]);


    free(float_array_heap);
    free(double_array_heap);
}

// faire mon fast tSNE avec

void print_system_info(){
    printf("\n");
    printf("\n    ---------------- System info ----------------\n");
    printf("This computer has %d cores\n", omp_get_num_procs());
    printf("This computer can run %d threads\n", omp_get_max_threads());

    int num_devices;
    cudaError_t cuda_error = cudaGetDeviceCount(&num_devices);
    if (cuda_error != cudaSuccess) {
        printf("Failed to get device count: %s\n", cudaGetErrorString(cuda_error));
        return;
    }
    printf("This computer has %d GPUs\n", num_devices);
    for (int i = 0; i < num_devices; i++) {
        struct cudaDeviceProp prop;
        cuda_error = cudaGetDeviceProperties(&prop, i);
        if (cuda_error != cudaSuccess) {
            printf("Failed to get device properties for GPU %d: %s\n", i, cudaGetErrorString(cuda_error));
            continue;
        }
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
    

    // int nthreads = omp_get_max_threads();
    // printf("This computer can run %d threads\n", nthreads);
}