import random
import sys, getopt

def generate(nnode, nedge, nfeature): #graph is undirected
    S = set()
    x = nedge*2 // nnode;
    count_lits = [0 for i in range(nnode)]
    remaining_node = set()
    for i in range(nnode):
        remaining_node.add(i)

    while (len(S) < nedge):
        print(len(S))
        remaining_node_list = list(remaining_node)
        remaining_node_count = len(remaining_node_list)
        a = remaining_node_list[random.randint(0, remaining_node_count-1)]
        b = remaining_node_list[random.randint(0, remaining_node_count-1)]
        if count_lits[a]>= x or count_lits[b]>=x:
            continue
        if a == b:
            continue
        if (a,b) in S or (b,a) in S:
            continue
        if a < b:
            S.add((a, b))
        else:
            S.add((b, a))
        count_lits[a] += 1
        count_lits[b] += 1

        if (count_lits[a]==x):
            remaining_node.remove(a)
        if (count_lits[b]==x):
            remaining_node.remove(b)

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
            u = random.uniform(0, 1)
            u = round(u, 2)
            l.append(u)
        R.append(l)

    file_name = "data/regular_graph_%d_%d_%d.txt" %(nnode, nedge, nfeature)
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