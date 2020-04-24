#include <string.h>
#include <getopt.h>
#include "gat.h"

int main(int argc, char *argv[]) {
    //char c;
    FILE *gfile;

    if (argc >= 2) gfile = fopen(argv[1], "r");
    else gfile = fopen("data/simple_5_3.txt", "r");

    graph_t *g = read_graph(gfile);
    printf("g: %d\n", g->nnode);
    printf("g: %d\n", g->nedge);
    printf("g: %d\n", g->nfeature);
    return 0;
}
