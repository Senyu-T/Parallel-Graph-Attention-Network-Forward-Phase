import random
import sys, getopt

def generate(nnode, nedge, nfeature): #graph is undirected
    S = set()
    while (len(S) < nedge):
        a = random.randint(0, nnode-1)
        b = random.randint(0, nnode-1)
        if a == b:
            continue
        if a < b:
            S.add((a, b))
        else:
            S.add((b, a))

    L = [[0 for i in range(nnode)] for j in range(nnode)]
    for s in S:
        (a, b) = s
        L[a][b] = 1
        L[b][a] = 1

    for i in range(nnode):
        L[i][i] = 1

    R = []
    for i in range(nnode):
        l = []
        for j in range(nfeature):
            u = random.uniform(0, 5.1)
            u = round(u, 2)
            l.append(u)
        R.append(l)

    file_name = "data/graph_%d_%d_%d.txt" %(nnode, nedge, nfeature)
    graph_file = open(file_name,"w")

    graph_file.write("%d %d %d\n" %(nnode, nedge, nfeature))
    for i in range(nnode):
        for j in range(nnode):
            graph_file.write("%d "% L[i][j])

        graph_file.write("\n")

    for i in range(nnode):
        for j in range(nfeature):
            graph_file.write("%f "%R[i][j])
        graph_file.write("\n")


    return


def main(argv):
    opts, args = getopt.getopt(argv, "n:e:f:")
    nnode = 0
    nedge = 0
    nfeature = 0

    for opt,arg in opts:
        if opt == '-n':
            nnode = int(arg)
        if opt == '-e':
            nedge = int(arg)
        if opt == '-f':
            nfeature = int(arg)

    generate(nnode, nedge, nfeature)


if __name__ == "__main__":
   main(sys.argv[1:])