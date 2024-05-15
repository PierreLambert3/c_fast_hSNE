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
    NeighHDDiscoverer* neighHD_discoverer;
    NeighLDDiscoverer* neighLD_discoverer;
    EmbeddingMaker* embedding_maker;
} GuiManager;

GuiManager* new_GuiManager(uint32_t _N_, NeighHDDiscoverer* _neighHD_discoverer_,  NeighLDDiscoverer* _neighLD_discoverer_, EmbeddingMaker* _embedding_maker_, uint32_t* thread_rand_seed);
void  destroy_GuiManager(GuiManager* thing);
int routine_GuiManager(void* arg);
void start_thread_GuiManager(GuiManager* thing);

#endif // GUIMANAGER_H