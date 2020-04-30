import random
import sys, getopt

def generate(nheads, in_f, out_f):
    a = []
    weights = []
    for i in range(nheads):
        tmp_a = []
        tmp_w = []
        for j in range(2*out_f):
            e = random.uniform(0,2)
            tmp_a.append(round(e,2))
        a.append(tmp_a)

        for i in range(in_f):
            l = []
            for j in range(out_f):
                e = random.uniform(0,5)
                l.append(round(e,2))
            tmp_w.append(l)

        weights.append(tmp_w)

    file_name = "data/layer_%d_%d_%d.txt" % (nheads, in_f, out_f)
    graph_file = open(file_name, "w")

    graph_file.write("%d %d %d\n" % (nheads, in_f, out_f))
    for i in range(nheads):
        for e in a[i]:
            graph_file.write("%f " % e)
        graph_file.write("\n")

        for k in range(in_f):
            for j in range(out_f):
                graph_file.write("%f " % weights[i][k][j])
            graph_file.write("\n")

def main(argv):
    opts, args = getopt.getopt(argv, "h:i:o:")
    nhead = 0
    in_f = 0
    out_f = 0

    for opt, arg in opts:
        if opt == '-h':
            nhead = int(arg)
        if opt == '-i':
            in_f = int(arg)
        if opt == '-o':
            out_f = int(arg)

    generate(nhead, in_f, out_f)

if __name__ == "__main__":
   main(sys.argv[1:])




