#ifndef GUIMANAGER_H
#define GUIMANAGER_H

#include "includes_global.h"
#include "neighHD_discoverer.h"
#include "neighLD_discoverer.h"
#include "embedding_maker.h"

typedef struct {
    SDL_Thread* sdl_thread;
    bool     isRunning;
    uint32_t rand_state;
    uint32_t N;
    uint32_t   N_random_colours;
    uint32_t** random_colours;
    uint32_t* Y;
    uint32_t  ms_since_Qdenom_drawn;
    uint32_t  period1;
    uint32_t  periodic_counter1;
    float     Qdenom_EMA;
    float     pct_new_LD_neighs_EMA;
    NeighHDDiscoverer* neighHD_discoverer;
    NeighLDDiscoverer* neighLD_discoverer;
    EmbeddingMaker* embedding_maker;
} GuiManager;
void amber_colour(SDL_Renderer* renderer);

void new_GuiManager(GuiManager* thing, uint32_t _N_, uint32_t* _Y_, NeighHDDiscoverer* _neighHD_discoverer_,  NeighLDDiscoverer* _neighLD_discoverer_, EmbeddingMaker* _embedding_maker_, uint32_t* thread_rand_seed);
void  destroy_GuiManager(GuiManager* thing);
int routine_GuiManager(void* arg);
void start_thread_GuiManager(GuiManager* thing);

void manage_events(SDL_Event* event, GuiManager* thing);
void draw_screen_block(SDL_Renderer* renderer, GuiManager* thing);
void manage_frame_rate(GuiManager* thing, uint32_t elapsed_time, uint32_t target_frame_time);

#endif // GUIMANAGER_H