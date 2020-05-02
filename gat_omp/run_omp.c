#include <string.h>
#include <getopt.h>
#include <omp.h>
#include <time.h>
#include "gat_omp.h"

int main(int argc, char *argv[]) {
    //char c;
    FILE *gfile, *lfile;
    gfile = fopen("/afs/andrew.cmu.edu/usr7/yilel/private/15418/Parallel-Graph-Attention-Network-Forward-Phase/data/graph_2048_800000_2048.txt", "r");
    lfile = fopen("/afs/andrew.cmu.edu/usr7/yilel/private/15418/Parallel-Graph-Attention-Network-Forward-Phase/data/simple_1_3_3_layer.txt", "r");

    int thread_count = 8;
    int check_correctness = 0;
    int c;
    int out;
    int nheads;
    char *optstring = "f:l:ct:o:";
    while((c = getopt(argc, argv, optstring)) != -1){
        switch(c){
            case 'f':
                gfile = fopen(optarg, "r");
                break;
            case 'l':
                lfile = fopen(optarg, "r");
            case 't':
                thread_count = atoi(optarg);
                break;
            case 'c':
                check_correctness = 1;
                break;
            case 'o':
                out = atoi(optarg);
                break;
            case 'h':
                nheads = atoi(optarg);
                break;
            default:
                abort();
        }
    }

    omp_set_num_threads(thread_count);


    if (check_correctness){
        gfile = fopen("/afs/andrew.cmu.edu/usr7/yilel/private/15418/Parallel-Graph-Attention-Network-Forward-Phase/data/simple_5_3.txt", "r");
        lfile = fopen("/afs/andrew.cmu.edu/usr7/yilel/private/15418/Parallel-Graph-Attention-Network-Forward-Phase/data/layer_2_3_4.txt", "r");

        graph_t *g = read_graph(gfile);
        layer_t *new_layer = read_layer(lfile, g->nnode);
//        layer_t *new_layer = layer_init(in, out, g->nnode, g->nedge, nheads);
        forward(new_layer, g);

        int out =  g->nfeature;
        FILE *out_file = fopen("/afs/andrew.cmu.edu/usr7/yilel/private/15418/Parallel-Graph-Attention-Network-Forward-Phase/data/c_output.txt", "w+");
        char tmp[50];
        for (int i=0; i<g->nnode; i++){
            for (int j=0; j<out; j++){
                sprintf(tmp, "%lf ", g->features[i][j]);
                fputs(tmp, out_file);
            }
            fputs("\n", out_file);
        }
    }else{
        graph_t *g = read_graph(gfile);
        int in = g->nfeature;
        out = in;
        nheads = 2;

        layer_t *new_layer = layer_init(in, out, g->nnode, nheads);
        double start = omp_get_wtime(), diff;

        forward(new_layer, g);

        diff = omp_get_wtime() - start;

        printf("Total time spent %f seconds\n", diff);
    }

    return 0;
}
