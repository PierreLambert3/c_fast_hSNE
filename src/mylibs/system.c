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

void sleep_ms(uint32_t n_ms){
    usleep(n_ms * 1000);
}


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
pthread_mutex_t* mutexes_allocate_and_init(uint32_t size){
    pthread_mutex_t* mutexes = (pthread_mutex_t*)malloc(size * sizeof(pthread_mutex_t));
    if (mutexes == NULL) {
        die("Failed to allocate memory for mutexes");}
    for (uint32_t i = 0; i < size; i++) {
        if (pthread_mutex_init(&mutexes[i], NULL) != 0) {
            die("Failed to initialise mutex");}
    }
    return mutexes;
}

pthread_mutex_t* mutex_allocate_and_init(){
    pthread_mutex_t* mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    if (mutex == NULL) {
        die("Failed to allocate memory for mutex");}
    if (pthread_mutex_init(mutex, NULL) != 0) {
        die("Failed to initialise mutex");}
    return mutex;
}

// malloc handlers for 1d arrays
bool* malloc_bool(uint32_t size, bool init_val) {
    bool* array = (bool*)malloc(size * sizeof(bool));
    if (array == NULL) {
        die("Failed to allocate memory for bool array");
    }
    for (uint32_t i = 0; i < size; i++) {
        array[i] = init_val;
    }
    return array;
}

float* malloc_float(uint32_t size, float init_val) {
    float* array = (float*)malloc(size * sizeof(float));
    if (array == NULL) {
        die("Failed to allocate memory for float array");
    }
    for (uint32_t i = 0; i < size; i++) {
        array[i] = init_val;
    }
    return array;
}

double* malloc_double(uint32_t size, double init_val) {
    double* array = (double*)malloc(size * sizeof(double));
    if (array == NULL) {
        die("Failed to allocate memory for double array");
    }
    for (uint32_t i = 0; i < size; i++) {
        array[i] = init_val;
    }
    return array;
}

uint32_t* malloc_uint32_t(uint32_t size, uint32_t init_val) {
    uint32_t* array = (uint32_t*)malloc(size * sizeof(uint32_t));
    if (array == NULL) {
        die("Failed to allocate memory for uint32_t array");
    }
    for (uint32_t i = 0; i < size; i++) {
        array[i] = init_val;
    }
    return array;
}

uint16_t* malloc_uint16_t(uint32_t size, uint16_t init_val) {
    uint16_t* array = (uint16_t*)malloc(size * sizeof(uint16_t));
    if (array == NULL) {
        die("Failed to allocate memory for uint16_t array");
    }
    for (uint32_t i = 0; i < size; i++) {
        array[i] = init_val;
    }
    return array;
}

uint8_t* malloc_uint8_t(uint32_t size, uint8_t init_val) {
    uint8_t* array = (uint8_t*)malloc(size * sizeof(uint8_t));
    if (array == NULL) {
        die("Failed to allocate memory for uint8_t array");
    }
    for (uint32_t i = 0; i < size; i++) {
        array[i] = init_val;
    }
    return array;
}


// matrix[0] = row 0 ... matrix[n-1] = row n-1   matrix[n] = ptr to data
// !!! for use in cuda, need give the data array as a 1d array !!!
float** malloc_float_matrix(uint32_t n, uint32_t m, float init_val) {
    float** matrix = (float**)malloc((n + 1) * sizeof(float*));
    if (matrix == NULL) {
        die("Failed to allocate memory for float matrix");
    }
    float* data = (float*)malloc(n * m * sizeof(float));
    if (data == NULL) {
        die("Failed to allocate memory for float matrix data");
    }
    matrix[n] = data;  // Store a pointer to data in the matrix array
    for (uint32_t i = 0; i < n; i++) {
        matrix[i] = &data[m * i];
        for (uint32_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;
        }
    }
    return matrix;
}

uint32_t** malloc_uint32_t_matrix(uint32_t n, uint32_t m, uint32_t init_val){
    uint32_t** matrix = (uint32_t**)malloc((n + 1) * sizeof(uint32_t*));
    if (matrix == NULL) {
        die("Failed to allocate memory for uint32_t matrix");
    }
    uint32_t* data = (uint32_t*)malloc(n * m * sizeof(uint32_t));
    if (data == NULL) {
        die("Failed to allocate memory for uint32_t matrix data");
    }
    matrix[n] = data;  // Store a pointer to data in the matrix array
    for (uint32_t i = 0; i < n; i++) {
        matrix[i] = &data[m * i];
        for (uint32_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;
        }
    }
    return matrix;
}

inline float* as_float_1d(float** matrix, uint32_t n, uint32_t m){
    return matrix[n];
}

inline uint32_t* as_uint32_1d(uint32_t** matrix, uint32_t n, uint32_t m){
    return matrix[n];
}

inline void memcpy_float_matrix(float** recipient, float** original, uint32_t n, uint32_t m){
    memcpy(recipient[n], original[n], n*m*sizeof(float));
}

void free_matrix(void** matrix, uint32_t n){
    free(matrix[0]);  // Free the data array
    free(matrix);  // Free the array of pointers
}

void free_array(void* array){
    free(array);
}

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

/***
 *     _____ ______ _   _         __    _____ ______ _   _                                                _           _   _             
 *    |  __ \| ___ \ | | |       / /   /  __ \| ___ \ | | |                                              (_)         | | (_)            
 *    | |  \/| |_/ / | | |      / /    | /  \/| |_/ / | | |      ___ ___  _ __ ___  _ __ ___  _   _ _ __  _  ___ __ _| |_ _  ___  _ __  
 *    | | __ |  __/| | | |     / /     | |    |  __/| | | |     / __/ _ \| '_ ` _ \| '_ ` _ \| | | | '_ \| |/ __/ _` | __| |/ _ \| '_ \ 
 *    | |_\ \| |   | |_| |    / /      | \__/\| |   | |_| |    | (_| (_) | | | | | | | | | | | |_| | | | | | (_| (_| | |_| | (_) | | | |
 *     \____/\_|    \___/    /_/        \____/\_|    \___/      \___\___/|_| |_| |_|_| |_| |_|\__,_|_| |_|_|\___\__,_|\__|_|\___/|_| |_|
 *                                                                                                                                      
 *                                                                                                                                      
 */

static void init_GPU_CPU_sync(GPU_CPU_sync* sync) {
    sync->mutex_request = mutex_allocate_and_init();
    sync->mutex_ready   = mutex_allocate_and_init();
    sync->mutex_buffer  = mutex_allocate_and_init();
    sync->flag_request  = false;
    sync->flag_ready    = false;
}

static void init_GPU_CPU_float_buffer(GPU_CPU_float_buffer* thing, uint32_t size){
    thing->buffer = malloc_float(size, 0.f);
    init_GPU_CPU_sync(&thing->sync);
}

static void init_GPU_CPU_uint32_buffer(GPU_CPU_uint32_buffer* thing, uint32_t size){
    thing->buffer = malloc_uint32_t(size, 0u);
    init_GPU_CPU_sync(&thing->sync);
}

GPU_CPU_float_buffer* malloc_GPU_CPU_float_buffer(uint32_t size){
    GPU_CPU_float_buffer* thing = (GPU_CPU_float_buffer*)malloc(sizeof(GPU_CPU_float_buffer));
    if (thing == NULL) {
        die("Failed to allocate memory for GPU_CPU_float_buffer");}
    init_GPU_CPU_float_buffer(thing, size);
    return thing;
}

GPU_CPU_uint32_buffer* malloc_GPU_CPU_uint32_buffer(uint32_t size){
    GPU_CPU_uint32_buffer* thing = (GPU_CPU_uint32_buffer*)malloc(sizeof(GPU_CPU_uint32_buffer));
    if (thing == NULL) {
        die("Failed to allocate memory for GPU_CPU_uint32_buffer");}
    init_GPU_CPU_uint32_buffer(thing, size);
    return thing;
}

bool is_requesting_now(GPU_CPU_sync* sync){
    pthread_mutex_lock(sync->mutex_request);
    bool flag = sync->flag_request;
    pthread_mutex_unlock(sync->mutex_request);
    return flag;
}

bool is_ready_now(GPU_CPU_sync* sync){
    pthread_mutex_lock(sync->mutex_ready);
    bool flag = sync->flag_ready;
    pthread_mutex_unlock(sync->mutex_ready);
    return flag;
}

void notify_ready(GPU_CPU_sync* sync){
    set_request(sync, false);
    set_ready(sync, true);
}

void notify_request(GPU_CPU_sync* sync){
    set_ready(sync, false);
    set_request(sync, true);
}

void set_ready(GPU_CPU_sync* sync, bool value){
    pthread_mutex_lock(sync->mutex_ready);
    sync->flag_ready = value;
    pthread_mutex_unlock(sync->mutex_ready);
}

void set_request(GPU_CPU_sync* sync, bool value){
    pthread_mutex_lock(sync->mutex_request);
    sync->flag_request = value;
    pthread_mutex_unlock(sync->mutex_request);
}

