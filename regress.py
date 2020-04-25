import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class forward_layer:
    def __init__(self, a, weights, heads, alpha=1):
        self.adj = []
        self.features = []

        tmp_a = np.transpose(a)
        tmp_a = tmp_a.tolist()
        self.a = torch.FloatTensor(tmp_a) #length 2*out_features

        tmp_w = np.transpose(weights)
        tmp_w = tmp_w.tolist()
        self.W = torch.FloatTensor(tmp_w)

        self.nnode = 0
        self.nnedge = 0
        self.nfeature = 0
        self.out_features = len(self.a) // 2
        self.nheads = heads
        self.results = []
        self.leakyrelu = nn.LeakyReLU(alpha)

        return

    def read_graph(self, file):
        f = open(file, "r")

        while (True):
            cur_line = f.readline()
            if cur_line[0] != '#':
                break

        l = cur_line.split(" ")
        self.nnode = int(l[0])
        self.nnedge = int(l[1])
        self.nfeature = int(l[2])

        for i in range(self.nnode):
            cur_line = f.readline()
            l_list = cur_line.split(" ")
            new_f = []
            for sf in l_list:
                new_f.append(int(sf))
            self.adj.append(new_f)

        for i in range(self.nnode):
            cur_line = f.readline()
            l_list = cur_line.split(" ")
            new_f = []
            for sf in l_list:
                new_f.append(float(sf))
            self.features.append(new_f)



        self.features = torch.FloatTensor(self.features)
        self.adj = torch.IntTensor(self.adj)
        return

    def forward(self):
        for i in range(self.nheads):

            h = torch.mm(self.features, self.W)
            N = h.size()[0]

            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(self.adj > 0, e, zero_vec)
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
    new_forward = forward_layer(a, weights, nheads)
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





#out_feature = 3
a = [[1,2,3,4,5,6]]
weights = [[0.1, 0.2, 0.3],[0.15, 0.25, 0.4],[0.2, 0.3, 0.12]]
nheads = 1
new_forward = forward_layer(a, weights, nheads)
new_forward.read_graph("data/simple_5_3.txt")
ref_res = new_forward.forward()
print(ref_res)






