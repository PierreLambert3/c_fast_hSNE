#include "neighLD_discoverer.h"

NeighLDDiscoverer* new_NeighLDDiscoverer(uint32_t _N_, uint32_t* thread_rand_seed, uint32_t max_nb_of_subthreads,\
        pthread_mutex_t* _mutex_Qdenom_, pthread_mutex_t* mutexes_sizeN, float** _Xld_, float** _Xhd_, uint32_t _Mld_, uint32_t _Khd_,\
        uint32_t _Kld_, uint32_t** _neighsLD_, uint32_t** _neighsHD_, float** _Q_, float* _Qdenom_){
    NeighLDDiscoverer* thing = (NeighLDDiscoverer*)malloc(sizeof(NeighLDDiscoverer));
    thing->isRunning = false;
    thing->rand_state = (uint32_t)time(NULL) +  thread_rand_seed[0]++;
    thing->passes_since_reset = 0;
    thing->p_wakeup = 1.0f;
    // initialize subthreads
    thing->N_reserved_subthreads = max_nb_of_subthreads;
    thing->N_subthreads_target   = max_nb_of_subthreads;
    thing->subthreads_chunck_size = 1 + floorf(0.05f * (float)_N_);
    printf("subthreads chunck size: %d\n", thing->subthreads_chunck_size);
    dying_breath("stop here");
    thing->subthreads        = (pthread_t*)malloc(sizeof(pthread_t) * max_nb_of_subthreads);
    thing->threads_waiting_for_task = (bool*)malloc(sizeof(bool) * max_nb_of_subthreads);
    thing->subthreads_mutexes = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t) * max_nb_of_subthreads);
    thing->subthread_data    = (SubthreadData*)malloc(sizeof(SubthreadData) * max_nb_of_subthreads);
    for(uint32_t i = 0; i < max_nb_of_subthreads; i++){
        pthread_mutex_init(&thing->subthreads_mutexes[i], NULL);
        thing->threads_waiting_for_task[i] = true;
    }

    for(uint32_t i = 0; i < max_nb_of_subthreads; i++){
        thing->subthread_data[i].stop_this_thread = false;
        thing->subthread_data[i].N = _N_;
        thing->subthread_data[i].rand_state = thread_rand_seed[0]++;
        printf("(subthread) %d rand state\n", thing->subthread_data[i].rand_state);
        thing->subthread_data[i].L = 0;
        thing->subthread_data[i].R = 0;
        thing->subthread_data[i].Xld = _Xld_;
        thing->subthread_data[i].Xhd = _Xhd_;
        thing->subthread_data[i].Mld = _Mld_;
        thing->subthread_data[i].Khd = _Khd_;
        thing->subthread_data[i].Kld = _Kld_;
        thing->subthread_data[i].neighsLD = _neighsLD_;
        thing->subthread_data[i].neighsHD = _neighsHD_;
        thing->subthread_data[i].Q = _Q_;
        thing->subthread_data[i].estimated_Qdenom = 1.0f;
        thing->subthread_data[i].mutexes_sizeN = mutexes_sizeN;
        thing->subthread_data[i].thread_mutex = &thing->subthreads_mutexes[i];
        thing->subthread_data[i].thread_waiting_for_task = &thing->threads_waiting_for_task[i];
    }
    // Algorithm and subthread data: for determining LD neighbours, Q, and Qdenom
    thing->N = _N_;
    thing->Xld = _Xld_;
    thing->Xhd = _Xhd_;
    thing->Mld = _Mld_;
    thing->Khd = _Khd_;
    thing->Kld = _Kld_;
    thing->neighsLD = _neighsLD_;
    thing->neighsHD = _neighsHD_;
    thing->Q = _Q_;
    thing->Qdenom = _Qdenom_;
    thing->mutex_Qdenom = _mutex_Qdenom_;
    thing->mutexes_sizeN = mutexes_sizeN;
    return thing; 
}

void destroy_NeighLDDiscoverer(NeighLDDiscoverer* thing){
    free(thing->subthreads);
    free(thing->subthreads_mutexes);
    free(thing->subthread_data);
    free(thing);
}

// @copilot ignore these 3 comments:
    // Use data->N and data->test_variable
    // estimated_Qdenom is built on each subthread during its work, when the subthread is done, the discoverer updates Qdenom accordingly
    // Qij est deja divisé par le denominateur: ca sera rapide comme ca

// the subthreads execute this function
void* subroutine_NeighLDDiscoverer(void* arg){
    SubthreadData* data = (SubthreadData*)arg;
    while(!data->stop_this_thread){
       double dbl_acc_denom = 0.;
       uint32_t n_votes = 0;
       data->estimated_Qdenom = 0.0f;

        
        pthread_mutex_lock(thread_mutex);
        data->estimated_Qdenom = (float) (dbl_acc_denom * ( ((double) (N*N - N)) / (double) n_votes));
        pthread_mutex_unlock(thread_mutex);
    }
    // (pour le deadlock juste copier i sur la stack avec lock puis demander le lock pour les j apres)
    // (pour le deadlock juste copier i sur la stack avec lock puis demander le lock pour les j apres)
    // (pour le deadlock juste copier i sur la stack avec lock puis demander le lock pour les j apres)
    

    return NULL;
}

void* routine_NeighLDDiscoverer(void* arg){
    NeighLDDiscoverer* thing = (NeighLDDiscoverer*)arg;
    thing->isRunning = true;
    // launch subthreads
    for(uint32_t i = 0; i < thing->N_reserved_subthreads; i++){
        if(pthread_create(&thing->subthreads[i], NULL, subroutine_NeighLDDiscoverer, &thing->subthread_data[i]) != 0){
            dying_breath("pthread_create subroutine_NeighLDDiscoverer failed");}
    }

    uint32_t cursor = 0; // the cursor for the start of the next chunk
    while (thing->isRunning) {
        pthread_mutex_lock(thing->mutex_Qdenom);
        float    now_Qdenom = Qdenom[0];
        pthread_mutex_unlock(thing->mutex_Qdenom);
        for(uint32_t i = 0; i < thing->N_subthreads_target; i++){
            // Check if the thread is waiting for a task
            float estimation_of_denom = -1.0f; 
            pthread_mutex_lock(&thing->subthreads_mutexes[i]);
            if(thing->threads_waiting_for_task[i]){ // subthread is waiting for a task
                // 1: temporarily save estimated denominator value
                estimation_of_denom = thing->subthread_data[i].estimated_Qdenom;
                
                // 2: Assign a new task to the thread
                thing->threads_waiting_for_task[i] = false;
                thing->subthread_data[i].L = cursor;
                thing->subthread_data[i].R = cursor + thing->subthreads_chunck_size > thing->N ? thing->N : cursor + thing->subthreads_chunck_size;
                thing->subthread_data[i].estimated_Qdenom = now_Qdenom;
                thing->

                // update the cursor for the next subthread
                cursor += thing->subthreads_chunck_size;
                if(cursor >= thing->N){
                    cursor = 0;}
            }
            pthread_mutex_unlock(&thing->subthreads_mutexes[i]);
            if(estimation_of_denom > 0.0f){
                pthread_mutex_lock(thing->mutex_Qdenom);
                double dbl_old_Qdenom = (double) Qdenom[0];
                dbl_old_Qdenom = alphaQ*dbl_old_Qdenom + (1.0 - alphaQ)*(double)estimation_of_denom;
                Qdenom[0] = (float) dbl_old_Qdenom;
                pthread_mutex_unlock(thing->mutex_Qdenom);
            }
        }
        usleep(40000); // 4% of a second, prevent the thread from taking too much CPU time
    }
    return NULL;
}

void start_thread_NeighLDDiscoverer(NeighLDDiscoverer* thing){
    if(pthread_create(&thing->thread, NULL, routine_NeighLDDiscoverer, thing) != 0){
        dying_breath("pthread_create routine_NeighLDDiscoverer failed");}
}

