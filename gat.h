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

    double **features;   // size: nnode * nfeature 
} graph_t;


typedef struct {
    int in_feature;
    int out_feature;
    double **weights;    // weight parameters (in_feature * out_feature)
    double **linear;     // featrues after linear transform. size (nnode, out_feat)
    double *a;           // self-attention parameters, size (2 * out_feature)
    double *tmp_attn;   // exp(lrelu(a*(Wh_i||Wh_k))) for edge ik. size nedge+nnode
    double *attentions;  // attention coefficients  nedge + nnode 
} param_t;


typedef struct {
    int num_heads;        // K, number of heads in multi-head computation
    param_t **params;     // K sets of params
} layer_t;


double lrelu(double x, double alpha);
double *concat_weights(double *weight_a, double *weight_b, int size);
param_t *param_init(int in_feature, int out_feature, int nnode, int nedge);
layer_t *layer_init(int in_feature, int out_feature, int nnode, int nedge, int nhead);
graph_t *new_graph(int node, int edge, int feat);
graph_t *read_graph(FILE *infile);
