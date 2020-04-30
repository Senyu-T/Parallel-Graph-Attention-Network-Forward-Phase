
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "gat_omp.h"
#include <omp.h>
#include <time.h>



// forward for one layer
void forward(layer_t *L, graph_t *g) {
    int nnode = g->nnode;
    int nhead = L->num_heads;
    int nedge = g->nedge;
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

    int BLOCKJ_SIZE = (nnode >= 32) ? 32 : nnode;
    int BLOCKI_SIZE = (out >= 32) ? 32 : out;
    int BLOCKK_SIZE = (in >= 32) ? 32 : in;

#pragma omp parallel
    {

        for (int headid = 0; headid < nhead; headid++) {
            /* STEP 1: Linear Activation */
            // TODO: cache optimization
            double **weights = params[headid]->weights;
            double **linear = params[headid]->linear;

#pragma omp for
            for (int jblock=0; jblock<nnode; jblock+=BLOCKJ_SIZE) {
                for (int iblock = 0; iblock < out; iblock += BLOCKI_SIZE)
                    for (int kblock = 0; kblock < in; kblock += BLOCKK_SIZE) {
                        for (int j = 0; j < BLOCKJ_SIZE; j++) {
                            for (int i = 0; i < BLOCKI_SIZE; i++) {
                                for (int k = 0; k < BLOCKK_SIZE; k++) {
                                    linear[jblock + j][iblock + i] +=
                                            g->features[jblock + j][kblock + k] * weights[kblock + k][iblock + i];
                                }
                            }
                        }
                    }
            }
//
            /* STEP 2: get attention coefficients */
            double start = omp_get_wtime(), diff;

            double *self_attn = params[headid]->a;
            //double *tmp_attn = params[headid]->tmp_attn;
            double **tmp_attn = calloc(sizeof(double*), nnode);
            for (int i=0; i<nnode; i++){
                tmp_attn[i] = calloc(sizeof(double), nnode + 2*nedge);
            }
            //double *attentions = params[headid]->attentions;
#pragma omp for
            for (int nid = 0; nid < nnode; nid++) {

                /* a * Wh_i */
                double left = 0;
                for (int fid = 0; fid < out; fid++)
                    left += self_attn[fid] * linear[nid][fid];

                int nnid_s = neighbor_start[nid];
                int nnid_e = neighbor_start[nid + 1];
                /* fill tmp_attn with  exp(lrelu(a * (Wh_i || Wh_k) */
                double down = 0;   // sum of all exp

                for (int nnid = nnid_s; nnid < nnid_e; nnid++) {

                    int neighbor_id = neighbor[nnid];
                    double right = 0;
                    for (int fid = 0; fid < out; fid++)
                        right += self_attn[out + fid] * linear[neighbor_id][fid];
                    tmp_attn[nid][nnid] = exp(lrelu(left + right, ALPHA));
                    down += tmp_attn[nid][nnid];
                }


                /* get alpha_ij then step 3 */
                for (int nnid = nnid_s; nnid < nnid_e; nnid++) {
                    double attention = tmp_attn[nid][nnid] / down;
                    for (int fid = 0; fid < out; fid++) {
                        int neighbor_id = neighbor[nnid];
                        /* STEP 3: get output embedding */
                        multi_new_embedding[nid][headid * out + fid] += attention * linear[neighbor_id][fid];
                    }
                }
            }

            diff = omp_get_wtime() - start;
            printf("Total time jjj spent %f seconds\n", diff);
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



