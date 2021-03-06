//
// Created by Yile Liu on 4/30/20.
//

#ifndef PARALLEL_GRAPH_ATTENTION_NETWORK_FORWARD_PHASE_GAT_H
#define PARALLEL_GRAPH_ATTENTION_NETWORK_FORWARD_PHASE_GAT_H

/* C implementation of GAT attention layer forward phase
 * Author: Senyu Tong
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

//#define NUM_HEAD 5      // number of heads = 5
#define ALPHA 0.2         // alpha for leakyRelu
#define MAXLINE 1024      // reading file max length

typedef struct {
    int nnode;
    int nedge;
    int nfeature;

    // TODO: decide which form is better, should we add edge list?
    int *neighbor;       // adjacency lists, length = nedge + nnode
    int *neighbor_start; // starting index for each adjacency list. nnode + 1

    double *features;   // size: nnode * nfeature
    int *adj;
} graph_t;



typedef struct {
    int num_heads;        // K, number of heads in multi-head computation
    double *weights;    // weight parameters (in_feature * out_feature * nheads)
    double *a;
    int in_feature;
    int out_feature;
} layer_t;

void forward(layer_t *L, graph_t *G);
double lrelu(double x, double alpha);
double *concat_weights(double *weight_a, double *weight_b, int size);
layer_t *layer_init(int in_feature, int out_feature, int nnode, int nedge, int nhead);
graph_t *new_graph(int node, int edge, int feat);
graph_t *read_graph(FILE *infile);
layer_t *read_layer(FILE *infile, int nnode, int nedge);


#endif //PARALLEL_GRAPH_ATTENTION_NETWORK_FORWARD_PHASE_GAT_H
