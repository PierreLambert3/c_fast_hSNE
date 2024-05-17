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

void manage_events(SDL_Event* event, GuiManager* thing) {
    while (SDL_PollEvent(event)) {
        // if escape key is pressed, then quit
        if (event->type == SDL_KEYDOWN) {
            if (event->key.keysym.sym == SDLK_ESCAPE) {
                thing->isRunning = false;
                dying_breath("SDL_KEYDOWN event");
            }
        }
        printf("event\n");
    }
    printf("  NO event\n");
}

void draw_screen_block(SDL_Renderer* renderer, GuiManager* thing) {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);
}

void manage_frame_rate(GuiManager* thing, uint32_t elapsed_time, uint32_t target_frame_time) {
    // if point drawing too slow: draw points once every three frames
    if (elapsed_time < target_frame_time) {
        printf("sleep for %d\n", target_frame_time - elapsed_time);
        SDL_Delay(target_frame_time - elapsed_time);
    }
    else{
        printf("drawing takes too long!\n");
    }
}

int routine_GuiManager(void* arg) {
    // create the SDL window
    GuiManager* thing = (GuiManager*)arg;
    SDL_Window* window = SDL_CreateWindow(
        "my cool title",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        GUI_W,
        GUI_H,
        SDL_WINDOW_OPENGL
    );    
    if (window == NULL) {
        printf("Could not create window: %s\n", SDL_GetError());
        return 1;
    }
    // create the SDL renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    // create event
    SDL_Event event;
    // main loop
    const uint32_t target_frame_rate = 30; // in frames per second
    const uint32_t target_frame_time = 1000 / target_frame_rate; // in milliseconds
    thing->isRunning = true;
    while (thing->isRunning) {
        uint32_t start_time = SDL_GetTicks();
        manage_events(&event, thing);
        draw_screen_block(renderer, thing);
        uint32_t elapsed_time = SDL_GetTicks() - start_time;
        manage_frame_rate(thing, elapsed_time, target_frame_time);
        printf("frame\n");
    }
    dying_breath("routine_GuiManager");
    SDL_DestroyWindow(window);
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
    // initialise SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        dying_breath("SDL_Init failed");
    }
    // create & launch the GUI thread
    thing->sdl_thread = SDL_CreateThread(routine_GuiManager, "GuiManagerThread", thing);
    if (thing->sdl_thread == NULL) {
        dying_breath("SDL_CreateThread routine_GuiManager failed");}
    // wait for the GUI thread to finish
    int threadReturnValue;
    SDL_WaitThread(thing->sdl_thread, &threadReturnValue);
}