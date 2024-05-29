#include "gui_manager.h"

void lavender_colour(SDL_Renderer* renderer) {
    const int lavender_r = 220;
    const int lavender_g = 220;
    const int lavender_b = 250;
    SDL_SetRenderDrawColor(renderer, lavender_r, lavender_g, lavender_b, 255);
}

void amber_colour(SDL_Renderer* renderer) {
    const int amber_r = 167;
    const int amber_g = 80;
    const int amber_b = 0;
    SDL_SetRenderDrawColor(renderer, amber_r, amber_g, amber_b, 255);
}

void draw_text(char* text, SDL_Renderer* renderer, TTF_Font* font, int x, int y, SDL_Color colour) {
    SDL_Surface* text_surface = TTF_RenderText_Solid(font, text, colour);
    SDL_Texture* text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
    SDL_Rect text_rect = {x, y, text_surface->w, text_surface->h};
    SDL_RenderCopy(renderer, text_texture, NULL, &text_rect);
    SDL_FreeSurface(text_surface);
    SDL_DestroyTexture(text_texture);
}

void new_GuiManager(GuiManager* thing, uint32_t _N_, uint32_t* _Y_, NeighHDDiscoverer* _neighHD_discoverer_,  NeighLDDiscoverer* _neighLD_discoverer_, EmbeddingMaker* _embedding_maker_, uint32_t* thread_rand_seed) {
    thing->isRunning = false;
    thing->rand_state = (uint32_t)time(NULL) + ++thread_rand_seed[0];
    thing->N = _N_;
    thing->Y = _Y_;
    thing->period1 = 150;
    thing->periodic_counter1 = 0;
    thing->elapsed_1 = 0u;
    thing->timestamp_1 = SDL_GetTicks();
    thing->Qdenom_EMA = 1.0f;
    thing->N_random_colours = 20;
    thing->ms_since_Qdenom_drawn = 0;
    thing->random_colours = malloc_uint32_t_matrix(thing->N_random_colours, 3, 254);
    for (uint32_t i = 0; i < thing->N_random_colours; i++) {
        float flt_r = rand_float_between(&thing->rand_state, 0.0f, 1.0f);
        float flt_g = rand_float_between(&thing->rand_state, 0.0f, 1.0f);
        float flt_b = rand_float_between(&thing->rand_state, 0.0f, 1.0f);
        float desired_norm = (0.05 + rand_float_between(&thing->rand_state, 0.05f, 1.0f)) * sqrt(255.0f*255.0f + 255.0f*255.0f + 255.0f*255.0f);
        float norm = sqrt(flt_r*flt_r + flt_g*flt_g + flt_b*flt_b);
        float factor = desired_norm / norm;
        thing->random_colours[i][0] = (uint32_t)(flt_r * factor);
        thing->random_colours[i][1] = (uint32_t)(flt_g * factor);
        thing->random_colours[i][2] = (uint32_t)(flt_b * factor);
    }
    thing->neighHD_discoverer = _neighHD_discoverer_;
    thing->neighLD_discoverer = _neighLD_discoverer_;
    thing->embedding_maker    = _embedding_maker_;
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
        // printf("event\n");
    }
    // printf("  NO event\n");
}


void draw_screen_block(SDL_Renderer* renderer, GuiManager* thing) {

    // ------------- draw the points in LD: the first two dimensions of the embedding -------------
/***
 *         _                                     _              _     _ _             
 *        | |                                   | |            | |   | (_)            
 *      __| |_ __ __ ___      __   ___ _ __ ___ | |__   ___  __| | __| |_ _ __   __ _ 
 *     / _` | '__/ _` \ \ /\ / /  / _ \ '_ ` _ \| '_ \ / _ \/ _` |/ _` | | '_ \ / _` |
 *    | (_| | | | (_| |\ V  V /  |  __/ | | | | | |_) |  __/ (_| | (_| | | | | | (_| |
 *     \__,_|_|  \__,_| \_/\_/    \___|_| |_| |_|_.__/ \___|\__,_|\__,_|_|_| |_|\__, |
 *                                                                               __/ |
 *                                                                              |___/ 
 */
    float** Xld = thing->neighLD_discoverer->Xld;
    uint32_t N  = thing->N;
    //find min and max values for each dimension
    float min_x = Xld[0][0];
    float max_x = Xld[0][0];
    float min_y = Xld[0][1];
    float max_y = Xld[0][1];
    for (uint32_t i = 0; i < N; i++) {
        if (Xld[i][0] < min_x) {min_x = Xld[i][0];}
        if (Xld[i][0] > max_x) {max_x = Xld[i][0];}
        if (Xld[i][1] < min_y) {min_y = Xld[i][1];}
        if (Xld[i][1] > max_y) {max_y = Xld[i][1];}
    }
    float x_span = max_x - min_x;
    float y_span = max_y - min_y;
    // draw the points
    float embedding_pixel_size = GUI_H * 0.82;
    float x_btm_left = 0.5f * (GUI_W - embedding_pixel_size);
    float y_btm_left = GUI_H;
    float x_multiplier = embedding_pixel_size / x_span;
    float y_multiplier = embedding_pixel_size / y_span;
    // clear embedding area with a rectangle
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_Rect rect = {x_btm_left, y_btm_left - embedding_pixel_size, embedding_pixel_size, embedding_pixel_size};
    SDL_RenderFillRect(renderer, &rect);
    for(uint32_t i = 0; i < N; i++) {
        uint32_t colour_index = thing->Y[i];
        SDL_SetRenderDrawColor(renderer, thing->random_colours[colour_index][0], thing->random_colours[colour_index][1], thing->random_colours[colour_index][2], 255);
        int pixel_x = (int) (x_btm_left + (Xld[i][0] - min_x) * x_multiplier);
        int pixel_y = (int) (y_btm_left - (Xld[i][1] - min_y) * y_multiplier);
        SDL_RenderDrawPoint(renderer, pixel_x, pixel_y);
    }   
    // rectangle around the embedding
    amber_colour(renderer);
    SDL_Rect rect2 = {x_btm_left, y_btm_left - embedding_pixel_size, embedding_pixel_size, embedding_pixel_size};    
    SDL_RenderDrawRect(renderer, &rect2);


/***
 *         _                                                     
 *        | |                                                    
 *      __| |_ __ __ ___      __   ___ _   _ _ ____   _____  ___ 
 *     / _` | '__/ _` \ \ /\ / /  / __| | | | '__\ \ / / _ \/ __|
 *    | (_| | | | (_| |\ V  V /  | (__| |_| | |   \ V /  __/\__ \
 *     \__,_|_|  \__,_| \_/\_/    \___|\__,_|_|    \_/ \___||___/
 *                                                               
 *                                                               
 */
    if(thing->ms_since_Qdenom_drawn > GUI_MS_UPDATE_QDENOM){
        int x, y;
        thing->ms_since_Qdenom_drawn = 0;
        float graph_W = embedding_pixel_size / 4;
        float graph_H = 0.17f * GUI_H;
        float x_btm_left = 0.55f * GUI_W;
        float y_btm_left = 0.17f * GUI_H;
        /* // ------------ draw evolution of LD denominator comapred to its EMA -------------
        // update EMA of Qdenom
        // pthread_mutex_lock(thing->embedding_maker->mutex_Qdenom);
        // float qdenom = thing->neighLD_discoverer->ptr_Qdenom[0];
        // thing->Qdenom_EMA = thing->Qdenom_EMA * 0.98f + (1.0f - 0.98f) * qdenom;
        float qdenom      = rand_float(&thing->rand_state);
        thing->Qdenom_EMA = thing->Qdenom_EMA * 0.98f + (1.0f - 0.98f) * qdenom;
        // pthread_mutex_unlock(thing->embedding_maker->mutex_Qdenom);
        // draw Qdenom according to how it evolves with its EMA
        float pct_diff = (qdenom - thing->Qdenom_EMA) / (FLOAT_EPS + thing->Qdenom_EMA);
        int x = (int) (thing->periodic_counter1 * graph_W / thing->period1);
        float y_mid = y_btm_left - 0.5f * graph_H;
        // draw a thing vertical rect as black to reset previously drawn things
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_Rect rect = {x_btm_left + x, y_btm_left-graph_H, 1, graph_H};
        SDL_RenderFillRect(renderer, &rect);
        //draw point corresponding to the current value of Qdenom
        amber_colour(renderer);
        int y = (int) (y_mid - pct_diff * graph_H);
        SDL_RenderDrawPoint(renderer, x_btm_left + x, y); */

        // ------------ draw the pct of new LD neighbours -------------
        graph_W = embedding_pixel_size / 4;
        graph_H = 0.17f * GUI_H;
        pthread_mutex_lock(thing->neighLD_discoverer->mutex_LDHD_balance);
        float pct_now    = thing->neighLD_discoverer->pct_new_neighs;
        float pct_now_HD = thing->neighHD_discoverer->pct_new_neighs;
        pthread_mutex_unlock(thing->neighLD_discoverer->mutex_LDHD_balance);
        
        x = (int) (thing->periodic_counter1 * graph_W / thing->period1);
        // draw x axis as a line
        SDL_SetRenderDrawColor(renderer, 255, 205, 105, 255);
        SDL_RenderDrawLine(renderer, x_btm_left, y_btm_left, x_btm_left + graph_W, y_btm_left);
        // draw a thin vertical rect as black to reset previously drawn things
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_Rect rect2 = {x_btm_left + x, y_btm_left-graph_H, 1, graph_H};
        SDL_RenderFillRect(renderer, &rect2);
        //draw point corresponding to the current value of pct of new neighbours in LD
        amber_colour(renderer);
        y = (int) (y_btm_left - pct_now * graph_H);
        SDL_RenderDrawPoint(renderer, x_btm_left + x, y);
        //draw point corresponding to the current value of pct of new neighbours in HD
        lavender_colour(renderer);
        y = (int) (y_btm_left - pct_now_HD * graph_H);
        SDL_RenderDrawPoint(renderer, x_btm_left + x, y);
    }

   
    //clear a black rect 
    int x0 = 5;
    int y0 = (int) 0.9f * GUI_H;
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_Rect rect0 = {x0, y0-30, 100, 60};
    SDL_RenderFillRect(renderer, &rect0);
    char elapsed_time_str[100];
    sprintf(elapsed_time_str, "%d", (int)thing->elapsed_1);
    SDL_Color textColor = {255, 255, 255, 0};
    draw_text(elapsed_time_str, renderer, thing->font, 10, 10, textColor);


    // render the screen
    SDL_RenderPresent(renderer);
}



void manage_frame_rate(GuiManager* thing, uint32_t elapsed_time, uint32_t target_frame_time) {
    // if point drawing too slow: draw points once every three frames
    if (elapsed_time < target_frame_time) {
        printf("sleep for %d\n", target_frame_time - elapsed_time);
        SDL_Delay(target_frame_time - elapsed_time);
    }
    else{
        // printf("drawing takes too long!\n");
    }
    // periodics & timers updates
    if(thing->ms_since_Qdenom_drawn <= 1){
        thing->periodic_counter1++;
        if(thing->periodic_counter1 >= thing->period1) {
            // dying_breath("periodic_counter1 reset\n");
            uint32_t now       = SDL_GetTicks();
            thing->elapsed_1   = now - thing->timestamp_1;
            thing->timestamp_1 = now;
            thing->periodic_counter1 = 0;}
    }
    thing->ms_since_Qdenom_drawn += (elapsed_time < target_frame_time) ? target_frame_time : elapsed_time;
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
    // init SDL font (thing->font)
    if (TTF_Init() != 0) {
        dying_breath("TTF_Init failed");}
    thing->font = TTF_OpenFont("../assets/thefont.ttf", 24);
    if (thing->font == NULL) {
        dying_breath("TTF_OpenFont failed");}
    
    // initialise the screen with black
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

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
    
}