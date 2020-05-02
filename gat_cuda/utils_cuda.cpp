//
// Created by Yile Liu on 4/30/20.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "gat_cuda.h"


/* take input file format :
 *  |  nnode
 *  |  nedge
 *  |  feature vector, one-hot vector for each node for edge |
 */

#define RM(r, c, width) ((r) * (width) + (c))

graph_t *new_graph(int nnode, int nedge, int nfeature) {
    graph_t *g = (graph_t *)malloc(sizeof(graph_t));
    g->nnode = nnode;
    g->nedge = nedge;
    g->nfeature = nfeature;
    g->neighbor = (int *)calloc(sizeof(int), nnode + nedge*2);
    g->neighbor_start = (int *)calloc(sizeof(int), nnode + 1);
    g->features = (double *)calloc(sizeof(double), nnode*nfeature);
    g->adj = (int*)calloc(sizeof(int), nnode * nnode);
//    for (int i = 0; i < nnode; i++)
//        g->features[i] = (double *)calloc(sizeof(double), nfeature);
    return g;
}


static inline bool is_comment(char *s) {
    int i;
    int n = strlen(s);
    for (i = 0; i < n; i++) {
        char c = s[i];
        if (!isspace(c)) return c == '#';
    }
    return false;
}

graph_t *read_graph(FILE *infile) {
    if (!infile) {
        fprintf(stderr, "ERROR opening file\n");
        return NULL;
    }

    char linebuf[MAXLINE];
    int nnode, nedge, nfeature;
    int lineno = 0;

    while (fgets(linebuf, MAXLINE, infile) != NULL) {
        lineno++;
        if (!is_comment(linebuf)) break;
    }
    if (sscanf(linebuf, "%d %d %d", &nnode, &nedge, &nfeature) != 3) {
        fprintf(stderr, "ERROR reading nums\n");
        return NULL;
    }


    graph_t *g = new_graph(nnode, nedge, nfeature);

    // read adj matrix
    int eid = 0;
    for (int i = 0; i < nnode; i++) {
        //       printf("node: %d\n", i);
        int one_hot;
        g->neighbor_start[i] = eid;
        for (int j = 0; j < nnode; j++) {
            if (!fscanf(infile, "%d", &one_hot))
                break;
            g->adj[i*nnode + j] = one_hot;
            if (one_hot)
                g->neighbor[eid++] = j;
            //           printf("%d ", one_hot);
        }
    }

    g->neighbor_start[nnode] = eid;


    if (eid != nnode + 2 * nedge) fprintf(stderr, "sth wrong %d\n", eid);

    // read features
    for (int i = 0; i < nnode; i++) {
        for (int j = 0; j < nfeature; j++) {
            if (!fscanf(infile, "%lf", &g->features[RM(i, j, nfeature)]))
                break;
        }
    }


    return g;
}


layer_t *layer_init(int in, int out, int nnode, int nedge, int num_heads) {
    layer_t *layer = (layer_t *)malloc(sizeof(layer));
    layer->num_heads = num_heads;
    layer->weights = (double *)calloc(sizeof(double), in*out*num_heads);
    for (int i=0; i<in*out*num_heads; i++){
        layer->weights[i] = ((double)rand()) / RAND_MAX;
    }

    layer->a = (double *)calloc(sizeof(double), 2 * out * num_heads);
    for (int i = 0; i < 2 * out * num_heads; i++)
        layer->a[i] = ((double)rand()) / RAND_MAX;

    layer->in_feature = in;
    layer->out_feature = out;

    return layer;
}


layer_t *read_layer(FILE *infile, int nnode, int nedge){
    if (!infile) {
        fprintf(stderr, "ERROR opening file\n");
        return NULL;
    }

    char linebuf[MAXLINE];
    int nheads, in_feature, out_feature;
    double w;
    int lineno = 0;

    while (fgets(linebuf, MAXLINE, infile) != NULL) {
        lineno++;
        if (!is_comment(linebuf)) break;
    }

    if (sscanf(linebuf, "%d %d %d", &nheads, &in_feature, &out_feature) != 3) {
        fprintf(stderr, "ERROR reading nheads\n");
        return NULL;
    }

    layer_t *layer = (layer_t *)malloc(sizeof(layer_t));
    layer -> num_heads = nheads;
    layer->weights = (double *)calloc(sizeof(double), in_feature * out_feature * nheads);
    layer->a = (double *)calloc(sizeof(double), 2 * out_feature * nheads);
    layer->in_feature = in_feature;
    layer->out_feature = out_feature;

    for (int hid = 0; hid < nheads; hid++){
        for (int out = 0; out < 2*out_feature; out++){
            fscanf(infile, "%lf", &w);
            layer->a[hid*2*out_feature + out] = w;
        }
        for (int in = 0; in < in_feature; in++){
            for (int out = 0; out < out_feature; out++){
                fscanf(infile, "%lf", &w);
                layer->weights[in*out_feature*nheads+hid*out_feature+out] = w;
            }
        }

    }

    return layer;
}

