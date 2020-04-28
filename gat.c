/* C implementation of GAT attention layer forward phase
 * SEQUENTIAL IMPLEMENTATION w/o any optimization
 * in this implementation, we use 3 attention layers and a final output layer
 * the change of features as model update is in-place
 * 
 * Author: Senyu Tong
 * id: senyut
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "gat.h"



// forward for one layer
void forward(layer_t *L, graph_t *g) {
    int nnode = g->nnode;
    int nhead = L->num_heads;
    int *neighbor = g->neighbor;
    int *neighbor_start = g->neighbor_start;
    param_t **params = L->params;
    int out = L->params[0]->out_feature;
    int in = L->params[0]->in_feature;

    // output new embedding with size (nnode, out_feature * nhead)
    double **multi_new_embedding = malloc(sizeof(double *)*nnode);
    for (int i=0; i<nnode; i++){
        multi_new_embedding[i] = malloc(sizeof(double)*out*nhead);
    }


    for (int headid = 0; headid < nhead; headid++) {
        /* STEP 1: Linear Activation */
        // TODO: cache optimization
        double **weights = params[headid]->weights;
        double **linear = params[headid]->linear;

        for (int nid = 0; nid < nnode; nid++) {
            double *orig_feature = g->features[nid];
            for (int out_id = 0; out_id < out; out_id ++) {
                double result = 0;
                for (int in_id = 0; in_id < in; in_id++)
                    result += orig_feature[in_id] * weights[in_id][out_id];
                linear[nid][out_id] = result;
            }
        }
        /* STEP 2: get attention coefficients */
        double *self_attn = params[headid]->a;
        double *tmp_attn = params[headid]->tmp_attn;
        double *attentions = params[headid]->attentions;
        for (int nid = 0; nid < nnode; nid++) {

            /* a * Wh_i */
            double left = 0;
            for (int fid = 0; fid < out; fid++)
                left += self_attn[fid] * linear[nid][fid];

            int nnid_s = neighbor_start[nid];
            int nnid_e = neighbor_start[nid + 1];
            //printf("%d, %d\n", nid, nnid_e);

            /* fill tmp_attn with  exp(lrelu(a * (Wh_i || Wh_k) */
            double down = 0;   // sum of all exp

            for (int nnid = nnid_s; nnid < nnid_e; nnid++) {

                int neighbor_id = neighbor[nnid];
                double right = 0;
                for (int fid = 0; fid < out; fid++)
                    right += self_attn[out + fid] * linear[neighbor_id][fid];
                tmp_attn[nnid] = exp(lrelu(left + right, ALPHA));
                down += tmp_attn[nnid];
            }


            /* get alpha_ij then step 3 */
            for (int nnid = nnid_s; nnid < nnid_e; nnid++) {
                attentions[nnid] = tmp_attn[nnid] / down;
                for (int fid = 0; fid < out; fid++) {
                    int neighbor_id = neighbor[nnid];
                    /* STEP 3: get output embedding */
                    multi_new_embedding[nid][headid * out + fid] += attentions[nnid] * linear[neighbor_id][fid];
                }
            }
        }
    }
    // update g
    g->features = multi_new_embedding;
    g->nfeature = out * nhead;
}






/* utility functions */
double lrelu(double x, double alpha) {
    return x < 0 ? alpha * x : x;
}

// concatenation, a, b or of equal size
double *concat_weights(double *a, double *b, int size) {
    double *concat = calloc(sizeof(double), 2 * size);
    memcpy(concat, a, size * sizeof(double));
    memcpy(concat + size, b, size * sizeof(double));
    return concat;
}



