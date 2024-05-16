#include "gui_manager.h"

void new_GuiManager(GuiManager* thing, uint32_t _N_, NeighHDDiscoverer* _neighHD_discoverer_,  NeighLDDiscoverer* _neighLD_discoverer_, EmbeddingMaker* _embedding_maker_, uint32_t* thread_rand_seed) {
    thing->isRunning = false;
    thing->rand_state = (uint32_t)time(NULL) + thread_rand_seed[0]++;
    thing->N = _N_;
    thing->neighHD_discoverer = _neighHD_discoverer_;
    thing->neighLD_discoverer = _neighLD_discoverer_;
    thing->embedding_maker = _embedding_maker_;
    printf("%d rand state\n", thing->rand_state);
}

void destroy_GuiManager(GuiManager* thing) {
    destroy_NeighHDDiscoverer(thing->neighHD_discoverer);
    destroy_NeighLDDiscoverer(thing->neighLD_discoverer);
    destroy_EmbeddingMaker(thing->embedding_maker);
    free(thing);
}


int routine_GuiManager(void* arg) {
    GuiManager* thing = (GuiManager*)arg;
    thing->isRunning = true;
    while (thing->isRunning) {
        //check random value
        uint32_t random_value = rand_uint32_between(&thing->rand_state, 0, 10);
        printf("%d                                      gui\n", random_value);
        sleep(rand_uint32_between(&thing->rand_state, 1, 2));
        if(random_value < 2) {
            thing->isRunning = false;}
    }
    // join on pthreads...
    pthread_join(thing->neighHD_discoverer->thread, NULL);
    pthread_join(thing->neighLD_discoverer->thread, NULL);
    pthread_join(thing->embedding_maker->thread, NULL);
    // destroy everything
    destroy_GuiManager(thing);
    return 0;
}

void start_thread_GuiManager(GuiManager* thing) {
    // launch worker threads
    start_thread_NeighHDDiscoverer(thing->neighHD_discoverer);
    start_thread_NeighLDDiscoverer(thing->neighLD_discoverer);
    start_thread_EmbeddingMaker(thing->embedding_maker);
    // create & launch the GUI thread
    thing->sdl_thread = SDL_CreateThread(routine_GuiManager, "GuiManagerThread", thing);
    if (thing->sdl_thread == NULL) {
        dying_breath("SDL_CreateThread routine_GuiManager failed");}
}