#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "gat_cuda.h"

#define LBLK 32

#define RM(r, c, width) ((r) * (width) + (c))

static inline int updiv(int n, int d) {
    return (n+d-1)/d;
}

__global__ void
cudaBlockKernel(int nnode, int in, int total_out, double *in_features, double *weights, double *out_features){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int bi = threadIdx.y;
    int bj = threadIdx.x;

    __shared__ float subA[LBLK * LBLK];
    __shared__ float subB[LBLK * LBLK];
    float sum = 0;

    for (int k = 0; k < in; k += LBLK) {
        subA[RM(bi,bj,LBLK)] = in_features[RM(i,k+bj,in)];
        subB[RM(bi,bj,LBLK)] = weights[RM(k+bi,j,total_out)];

        __syncthreads();

        for (int bk = 0; bk < LBLK; bk++) {
            sum += subA[RM(bi,bk,LBLK)] * subB[RM(bk,bj,LBLK)];
        }

        __syncthreads();
    }
    out_features[RM(i,j,total_out)] = sum;
}

//out_feature is nheads*out
void cudaMultMatrix(int nnode, int in, int total_out, double *in_features,
        double *weights, double *out_features){

    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(total_out, LBLK), updiv(nnode, LBLK));
    cudaBlockKernel<<<blocks, threadsPerBlock>>>(nnode, in, total_out, in_features, weights, out_features);
}

__global__ void
cudaLinearlrKernel(int nnode, int nheads, int out, double *linear, double *a, double *linear_lr){
    int i = blockIdx.y * blockDim.y + threadIdx.y; //nid
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int hid = blockIdx.x;

    double sum = 0;
    for (int k=0; k<out; k++){
        int l_idx = i * nheads * out + hid * out + k;
        int a_idx = hid * out * 2 + k + threadIdx.x * out;
        sum += linear[l_idx] * a[a_idx];
    }
    linear_lr[i * nheads * 2 + j] = sum;

}

void cudaComputeLR(int nnode, int nheads, int out, double *linear, double *a, double *linear_lr){

    dim3 threadsPerBlock(2, LBLK * 2);
    dim3 blocks(nheads, updiv(nnode, LBLK * 2));
    cudaLinearlrKernel<<<blocks, threadsPerBlock>>>(nnode, nheads, out, linear, a, linear_lr);
}


__global__ void
    cudaLReluKernel(int nnode, int nhead, int out, double *linear_lr, int *adj, double *relu_matrix, double *relu_sum){
    int i = blockIdx.y * blockDim.y + threadIdx.y; //nid
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int hid = blockIdx.x;
    int nei = threadIdx.x;//j in adj matrix

    if (adj[i*nnode + nei]){
        double left = linear_lr[i*2*nhead+hid*2];
        double right = linear_lr[nei*2*nhead+hid*2+1];
        double r = 0;
        if (left+right < 0){
            r = ALPHA * (left+right);
        }else{
            r = left + right;
        }
        relu_matrix[i*nnode*nhead + j] = exp(r);
        relu_sum[i*nnode*nhead + j] = exp(r);
    }

    __syncthreads();

    for (int offset=nnode/2; offset>=1; offset/=2){
        if (nei<offset){
            relu_sum[i*nnode*nhead+j] = relu_sum[i*nnode*nhead+j] + relu_sum[i*nnode*nhead+j+offset];
        }
        __syncthreads();
    }

    relu_matrix[i*nnode*nhead + j] = relu_matrix[i*nnode*nhead + j] / relu_sum[i*nnode*nhead+hid*nnode];
}

void cudaComputeLRelu(int nnode, int nhead, int out, double *linear_lr, int *adj, double *relu_matrix, double *relu_sum){
    //every thread correpsonds to one entry in the adj matrix
    dim3 threadsPerBlock(nnode, 1);
    dim3 blocks(nhead, nnode);
    cudaLReluKernel<<<blocks, threadsPerBlock>>>(nnode, nhead, out, linear_lr, adj, relu_matrix, relu_sum);
}


__global__ void
cudaComputeNewEmbedding(int nnode, int nhead, int out, double *relu_matrix, double *multi_new_embedding, int *neighbor,
                             int *neighbor_start, double *linear){
    int i = blockIdx.y * blockDim.y + threadIdx.y; //nid

    int hid = blockIdx.x;
    int fid = threadIdx.x;

    int nnid_s = neighbor_start[i];
    int nnid_e = neighbor_start[i + 1];

    for (int nnid = nnid_s; nnid < nnid_e; nnid++) {
        int nei = neighbor[nnid];
        multi_new_embedding[i*out*nhead + hid * out + fid] +=
                relu_matrix[i*nnode*nhead+hid*nnode+nei] * linear[nei*nhead*out+hid*out+fid];
    }

}

void cudaNewEmbedding(int nnode, int nhead, int out, double *relu_matrix, double *mult_new_bedding, int *neighbor,
        int *neighbor_start, double *linear){
    //every thread maps to an entry in the new embedding matrix
    dim3 threadsPerBlock(out, 1);

    //every block corresponds to one node in one head
    dim3 blocks(nhead, nnode);
    cudaComputeNewEmbedding<<<blocks, threadsPerBlock>>>(nnode, nhead, out, relu_matrix, mult_new_bedding, neighbor, neighbor_start, linear);


}


// forward for one layer
void forward(layer_t *L, graph_t *g) {
    int nnode = g->nnode;
    int nedge = g->nedge;
    int nhead = L->num_heads;
    int *neighbor = g->neighbor;
    int *neighbor_start = g->neighbor_start;
    int out = L->params[0]->out_feature;
    int in = L->params[0]->in_feature;

//Step 1, compute h*W, all heads are computed together
    double *device_linear;
    double *device_features;
    double *device_weights;

    double *features = g->features;
    double *weights = L->weights;
    cudaMalloc((void**)&device_features, nnode * in * sizeof(double));
    cudaMalloc((void**)&device_weights, in * out * nhead * sizeof(double));
    cudaMalloc((void**)&device_linear, nnode * nhead * out * sizeof(double));

    cudaMemcpy(device_features, features, nnode * in * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weights, weights, in * out * nhead * sizeof(double), cudaMemcpyHostToDevice);

    cudaMultMatrix(nnode, in, out*nhead, device_features, device_weights, device_linear);

    cudaDeviceSynchronize();

    //computer left and right value for every node, every head
    double *a = L->a;
    double *device_a;
    double *device_linear_lr; //nnode * (2 *nhead)

    cudaMalloc((void**)&device_a, 2 * out * nhead * sizeof(double));
    cudaMalloc((void**)&device_linear_lr, 2 * nhead * nnode * sizeof(double));
    cudaMemcpy(device_a, a, 2 * out * nhead * sizeof(double), cudaMemcpyHostToDevice);

    cudaComputeLR(nnode, nhead, out, device_linear, device_a, device_linear_lr);
    cudaDeviceSynchronize();

    //Step 2: for every edge apply leakyRelu, then compyre alpha_{ij}
    int *device_adj;
    double *device_relu_matrix;  //nnode * (nnode * nhead)
    double *device_relu_sum; //nnode * (nnode * nhead)
    cudaMalloc((void**)&device_adj, nnode * nnode  * sizeof(int));
    cudaMemcpy(device_adj, g->adj, nnode * nnode * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&device_relu_matrix, nnode * nnode * nhead * sizeof(double));
    cudaMalloc((void**)&device_relu_sum, nnode * nnode * nhead * sizeof(double));

    cudaComputeLRelu(nnode, nhead, out, device_linear_lr, device_adj, device_relu_matrix, device_relu_sum);

    double *relu_matrix = (double *)calloc(sizeof(double), nnode * nnode * nhead);
    cudaDeviceSynchronize();

    //Step 3: compute new embedding
    double *device_mult_new_embedding;
    int *device_neighbor;
    int *device_neighbor_start;
    cudaMalloc((void**)&device_mult_new_embedding, nnode * nhead * out * sizeof(double));
    cudaMalloc((void**)&device_neighbor, (nnode + 2 * nedge) * sizeof(int));
    cudaMalloc((void**)&device_neighbor_start, (nnode + 1) * sizeof(int));

    cudaMemcpy(device_neighbor, neighbor, (nnode + 2 * nedge) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_neighbor_start, neighbor_start, (nnode + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaNewEmbedding(nnode, nhead, out, device_relu_matrix, device_mult_new_embedding, device_neighbor,
            device_neighbor_start, device_linear);

    double *multi_new_embedding = (double *)calloc(sizeof(double), nnode * nhead * out);

    cudaMemcpy(multi_new_embedding, device_mult_new_embedding, nnode * nhead * out * sizeof(double), cudaMemcpyDeviceToHost);

    g->features = multi_new_embedding;
    g->nfeature = out * nhead;
}

/* utility functions */
double lrelu(double x, double alpha) {
    return x < 0 ? alpha * x : x;
}

// concatenation, a, b or of equal size
double *concat_weights(double *a, double *b, int size) {
    double *concat = (double *)calloc(sizeof(double), 2 * size);
    memcpy(concat, a, size * sizeof(double));
    memcpy(concat + size, b, size * sizeof(double));
    return concat;
}



