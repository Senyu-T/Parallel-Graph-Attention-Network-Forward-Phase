#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "gat_omp.h"


/* take input file format :
 *  |  nnode
 *  |  nedge
 *  |  feature vector, one-hot vector for each node for edge |
 */

graph_t *new_graph(int nnode, int nedge, int nfeature) {
    graph_t *g = malloc(sizeof(graph_t));
    g->nnode = nnode;
    g->nedge = nedge;
    g->nfeature = nfeature;
    g->neighbor = calloc(sizeof(int), nnode + nedge*2);
    g->neighbor_start = calloc(sizeof(int), nnode + 1);
    g->features = calloc(sizeof(double *), nnode);
    for (int i = 0; i < nnode; i++)
        g->features[i] = calloc(sizeof(double), nfeature);
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
  //      printf("node: %d\n", i);
        int one_hot;
        g->neighbor_start[i] = eid;
        for (int j = 0; j < nnode; j++) {
            if (!fscanf(infile, "%d", &one_hot))
                break;
            if (one_hot)
                g->neighbor[eid++] = j;
    //        printf("%d ", one_hot);
        }

   //     printf("\n\n\n");
    }
    g->neighbor_start[nnode] = eid;

//    for (int k =0; k<nnode+nedge; k++){
//            printf("%dm", g->neighbor[k]);
//        }


    if (eid != nnode + 2 * nedge) fprintf(stderr, "sth wrong %d\n", eid);
    /*
    for (int i = 0; i < eid; i++) {
        printf("%d ", g->neighbor[i]);
    }
    printf("\n");
    for (int i = 0; i < nnode + 1; i++)
        printf("%d ", g->neighbor_start[i]);
    printf("\n");
    */

    // read features
    for (int i = 0; i < nnode; i++) {
        for (int j = 0; j < nfeature; j++) {
            if (!fscanf(infile, "%lf", &g->features[i][j]))
                break;
        }
    }

    /*
    for (int i = 0; i < nnode; i++) {
        for (int j = 0; j < nfeature; j++) {
            printf("%lf ", g->features[i][j]);
        }
        printf("\n");
    }
    */


    return g;
}

/* init functions */
param_t *param_init(int in, int out, int nnode) {
    param_t *param = malloc(sizeof(param_t));
    param->in_feature = in;
    param->out_feature = out;

    param->weights = calloc(sizeof(double *), in);
    for (int i = 0; i < in; i++) {
        param->weights[i] = calloc(sizeof(double), out);
        for (int j = 0; j < out; j++)
            param->weights[i][j] = ((double)rand()) / RAND_MAX;
    }

    param->linear = calloc(sizeof(double *), nnode);
    for (int i = 0; i < nnode; i++)
        param->linear[i] = calloc(sizeof(double), out);

    param->a = calloc(sizeof(double), 2 * out);
    for (int i = 0; i < 2 * out; i++)
        param->a[i] = ((double)rand()) / RAND_MAX;

//    param->attentions = calloc(sizeof(double), nnode + nedge*2);
//    param->tmp_attn = calloc(sizeof(double), nnode + nedge*2);
    return param;
}

layer_t *layer_init(int in, int out, int nnode, int num_heads) {
    layer_t *layer = malloc(sizeof(layer));
    layer->num_heads = num_heads;
    layer->params = calloc(sizeof(param_t *), num_heads);
    for (int i = 0; i < num_heads; i++)
        layer->params[i] = param_init(in, out, nnode);
    return layer;
}


layer_t *read_layer(FILE *infile, int nnode){
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

    layer_t *layer = malloc(sizeof(layer_t));
    layer -> num_heads = nheads;
    param_t **params = malloc(sizeof(param_t *) * nheads);
    layer -> params = params;

    for (int hid = 0; hid < nheads; hid++){
        param_t *param = param_init(in_feature, out_feature, nnode);
        for (int out = 0; out < 2*out_feature; out++){
            fscanf(infile, "%lf", &w);
            param->a[out] = w;
        }
        for (int in = 0; in < in_feature; in++){
            for (int out = 0; out < out_feature; out++){
                fscanf(infile, "%lf", &w);
                param->weights[in][out] = w;
            }
        }

        layer->params[hid] = param;
    }

    return layer;
}

