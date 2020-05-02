//
// Created by Yile Liu on 4/30/20.
//

//only works for nnode is multiple of 64, in_feature and out_feature is multiple of 32

#include <string.h>
#include <getopt.h>
#include "gat_cuda.h"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <cstdlib>

int main(int argc, char *argv[]) {
    //char c;
    FILE *gfile, *lfile;
    gfile = fopen("/afs/andrew.cmu.edu/usr7/yilel/private/15418/Parallel-Graph-Attention-Network-Forward-Phase/data/graph_2048_800000_2048.txt", "r");
    int check_correctness = 0;

    char optstring[]= "f:l:t:c";
    int c, out, nheads;
    while((c = getopt(argc, argv, optstring)) != -1){
        switch(c){
            case 'f':
                gfile = fopen(optarg, "r");
                break;
            case 'l':
                lfile = fopen(optarg, "r");
            case 'c':
                check_correctness = 1;
                break;
            default:
                abort();
        }
    }

    graph_t *g = read_graph(gfile);

    //read in weights, a, nhead
    if (check_correctness){
        gfile = fopen("/afs/andrew.cmu.edu/usr7/yilel/private/15418/Parallel-Graph-Attention-Network-Forward-Phase/data/graph_64_100_64.txt", "r");
        lfile = fopen("/afs/andrew.cmu.edu/usr7/yilel/private/15418/Parallel-Graph-Attention-Network-Forward-Phase/data/layer_2_64_64.txt", "r");

        graph_t *g = read_graph(gfile);
        layer_t *new_layer = read_layer(lfile, g->nnode, g->nedge);
//        layer_t *new_layer = layer_init(in, out, g->nnode, g->nedge, nheads);
        forward(new_layer, g);

        int out =  g->nfeature;
        FILE *out_file = fopen("/afs/andrew.cmu.edu/usr7/yilel/private/15418/Parallel-Graph-Attention-Network-Forward-Phase/data/c_output.txt", "w+");

        char tmp[50];
        for (int i=0; i<g->nnode; i++){
            for (int j=0; j<out; j++){
                sprintf(tmp, "%lf ", g->features[i*out + j]);
                fputs(tmp, out_file);
            }
            fputs("\n", out_file);
        }

    }else{
        int in = g->nfeature;
        out = in;
        nheads = 2;

        layer_t *new_layer = layer_init(in, out, g->nnode, g->nedge, nheads);

        clock_t startTime = clock();

        forward(new_layer, g);

        clock_t endtime = clock();

        printf("Time difference = %f", (double)(endtime-startTime)/CLOCKS_PER_SEC);


    }

    return 0;
}
