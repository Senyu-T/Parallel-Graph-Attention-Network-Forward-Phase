#include <string.h>
#include <getopt.h>
#include "gat.h"

int main(int argc, char *argv[]) {
    //char c;
    FILE *gfile;


    if (argc >= 2) gfile = fopen(argv[1], "r");
    else gfile = fopen("data/simple_5_3.txt", "r");

    graph_t *g = read_graph(gfile);

    FILE *lfile = fopen("data/simple_1_3_3_layer.txt", "r");
    layer_t *new_layer = read_layer(lfile, g->nnode, g->nedge);
    //read in weights, a, nhead

//      int nheads = 5;
//      int in = 3;
//      int out = 3;
//      layer_t *new_layer = layer_init(in, out, g->nnode, g->nedge, nheads); \\generate random weights and a
    forward(new_layer, g);
    int out = new_layer->params[0]->out_feature;



    FILE *out_file = fopen("data/c_output.txt", "w+");
    char tmp[50];
    for (int i=0; i<g->nnode; i++){
        for (int j=0; j<out; j++){
            sprintf(tmp, "%lf ", g->features[i][j]);
            fputs(tmp, out_file);
        }
        fputs("\n", out_file);
    }


    return 0;
}
