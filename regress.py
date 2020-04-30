import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class forward_layer:
    def __init__(self, a, weights, alpha=1):
        self.adj = []
        self.features = []

        tmp_a = np.transpose(a)
        tmp_a = tmp_a.tolist()
        self.a = torch.FloatTensor(tmp_a) #length = 2*out_features

        self.W = torch.FloatTensor(weights) #Weight matrix, here the dimension is R*R'

        self.nnode = 0
        self.nnedge = 0
        self.nfeature = 0
        self.out_features = len(self.a) // 2 #number of out features

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

        #read in the adjcency matrix
        for i in range(self.nnode):
            cur_line = f.readline()
            l_list = cur_line.split(" ")
            new_f = []
            for sf in l_list:
                new_f.append(int(sf))
            self.adj.append(new_f)

        #read in the features
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
        h = torch.mm(self.features, self.W)
        N = h.size()[0]

        # step 2
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(self.adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        # step 3
        h_prime = torch.matmul(attention, h)
        #self.results.append(h_prime)

        return h_prime




def check(input_file, layer_file, c_output_file):
    #the c_output_file should consist of (nheads * nnode) * out_features

    a = []
    weights = []
    lf = open(layer_file, "r")
    cur_line = lf.readline().split()
    nheads = int(cur_line[0])
    in_f = int(cur_line[1])
    out_f = int(cur_line[2])

    for i in range(nheads):
        tmp_a = []
        tmp_w = []
        cur_line = lf.readline().split()
        for j in range(2 * out_f):
            tmp_a.append(float(cur_line[j]))
        a.append([tmp_a])

        for k in range(in_f):
            l = []
            cur_line = lf.readline().split()
            for j in range(out_f):
                l.append(float(cur_line[j]))

            tmp_w.append(l)

        weights.append(tmp_w)

    ref_res = []
    for i in range(nheads):
        #call the python implementation
        new_forward = forward_layer(a[i], weights[i])
        new_forward.read_graph(input_file)
        res = new_forward.forward()
        ref_res.append(res)

    nnode = new_forward.nnode
    out_feature = new_forward.out_features

    output_file = open(c_output_file)
    c_out = []
    for i in range(nnode):
        sl = output_file.readline().split()
        l = []
        for j in range(nheads * out_feature):
            l.append(round(float(sl[j]), 4))
        c_out.append(l)

    for hid in range(nheads):
        for nid in range(nnode):
            for fid in range(out_feature):
                if round(float(ref_res[hid][nid][fid]), 3) != round(c_out[nid][hid * out_feature + fid],3):
                    print("ERROR, python gives value %f c gives value %f" % (
                    ref_res[hid][nid][fid], c_out[nid][hid * out_feature + fid]))





check("data/simple_5_3.txt", "data/layer_2_3_4.txt", "data/c_output.txt")




