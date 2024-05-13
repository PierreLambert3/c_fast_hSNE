#include "includes_global.h"

/*
initialises SDL, and launches thread_gui0 an thread_worker0

thread_gui0:    listens for user input, updates the screen, and sends messages to thread_worker0
thread_worker0: runs the simulation, and listens for messages from thread_gui0
*/



int main() {
    set_console_colour(220, 130, 20);
    printf("starting program...\n");

    test_speed_flt_vs_dbl_no_cache_effects();
    test_speed_flt_vs_dbl_yes_cache_effects();
    print_system_info();
    
    // initialise SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }
    set_console_colour(101, 255, 120);
    printf("reached last instruction\n");
    return 0;

// 1/ mettre la set colour dans system_import 

//  2/ faire une utils rgb to terminal colour code qui retur le bon string et mettre 
//     le terminal en mode orange

    
}


