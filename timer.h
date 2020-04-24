//
// Created by Yile Liu on 4/24/20.
//


#ifndef PARALLEL_GRAPH_ATTENTION_NETWORK_FORWARD_PHASE_TIMER_H
#include <stdio.h>
#include <time.h>
#include <stdlib.h>


//#define ACTIVITY_COUNT = 4;

typedef enum {STEP_ONE, STEP_TWO, STEP_THREE, TOTAL} activity_t;

void start_activity(activity_t a);
void finish_activity(activity_t a);


#define PARALLEL_GRAPH_ATTENTION_NETWORK_FORWARD_PHASE_TIMER_H
#endif //PARALLEL_GRAPH_ATTENTION_NETWORK_FORWARD_PHASE_TIMER_H
