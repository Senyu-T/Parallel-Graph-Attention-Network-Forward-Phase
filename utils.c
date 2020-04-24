#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "gat.h"


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
    g->neighbor = calloc(sizeof(int), nnode + nedge);
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
        printf("node: %d\n", i);
        int one_hot;
        g->neighbor_start[i] = eid;
        for (int j = 0; j < nnode; j++) {
            if (!fscanf(infile, "%d", &one_hot))
                break;
            if (one_hot) 
                g->neighbor[eid++] = j;
            printf("%d ", one_hot);
        }
        printf("\n\n\n");
    }
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

