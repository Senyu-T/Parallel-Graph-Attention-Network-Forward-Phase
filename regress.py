import numpy as np
import torch

class forward_layer:
    def __init__(self, a, weights, heads):
        self.adj = []
        self.features = []
        self.a = a #length 2*out_features
        self.W = weights

        self.nnode = 0
        self.nnedge = 0
        self.nfeature = 0
        self.out_features = len(self.a) // 2
        self.heads = heads
        self.results = []

        return

    def read_graph(self, file):
        f = open(file, "r")

        while (true):
            cur_line = f.readline()
            if cur_line[0] != '#':
                break;

        l = cur_line.split(" ")
        self.nnode = int(l[0])
        self.nnedge = int(l[1])
        self.nfeature = int(l[2])

        for i in range(self.nnode):
            cur_line = f.readline()
            l_list = cur_line.split(" ")
            l = map(int, l_list)
            self.adj.append(l)

        for i in range(self.nnode):
            cur_line = f.readline()
            l_list = cur_line.split(" ")
            l = map(float, l_list)
            self.features.append(l)

        return

    def forward(self):
        for i in range(heads):

            h = torch.mm(self.features, self.W.t())
            N = h.size()[0]

            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)

            h_prime = torch.matmul(attention, h)
            self.results.append(h_prime)


        return self.results




def check(input_file, c_output_file):
    #the c_output_file should consist of (nheads * nnode) * out_features

    a = []
    weights = []  # where should this be defined
    nheads = 5

    #call the python implementation
    new_forward = forward_layer(nheads, a, weights, nheads)
    new_forward.read_graph(input_file)
    ref_res = new_forward.forward()

    nnode = new_forward.nnode
    out_feature = new_forward.out_features

    output_file = open(c_output_file)

    for hid in range(nheads):
        for nid in range(nnode):
            string_l = output_file.readline().split(" ")
            float_l = map(float, string_l)
            ref = ref_res[hid][nid]

            for fid in range(out_feature):
                if ref[fid] != float_l[fid]:
                    print ("ERROR\n")
                    break












