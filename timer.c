//
// Created by Yile Liu on 4/24/20.
//


#include "timer.h"


static char *activity_name[4] = { "step 1", "step 2", "step 3", "total"};

static clock_t current_start_time = 0.0;
static clock_t start_time = 0.0;

void start_activity(activity_t a){
    clock_t new_timer = clock();
    if (a == 0){
        start_time = new_timer;
    }else {
        current_start_time = new_timer;
    }
}

void finish_activity(activity_t a){
    if (a != 0){
        clock_t finish_time = clock();
        double time_elapsed = (finish_time - current_start_time) / CLOCKS_PER_SEC;
        printf("Finishing activity %s. Total time spent %f \n", activity_name[a], time_elapsed);
    }else{
        double finish_time = clock();
        double total_time = (finish_time - start_time) / CLOCKS_PER_SEC;
        printf("Finishing the forward layer. Total time spent %f \n", total_time);
    }

}
