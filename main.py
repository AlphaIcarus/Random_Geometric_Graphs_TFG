import sys
import matplotlib.pyplot as plt
import networkx as nx

from graph import Graph, UnionGraph

def drawGraph(G: nx.graph):

    subax1 = plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    subax2 = plt.subplot(122)
    nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')


# First we pass the number of edges, then the size of the boz then the radius
# Default values
n = 100
x = 1
r = 0.1
num_graph = 50

if len(sys.argv) > 1:
    n = int(sys.argv[1])
if len(sys.argv) > 2:
    x = int(sys.argv[2])
if len(sys.argv) > 3:
    r = float(sys.argv[3])
if len(sys.argv) > 4:
    num_graph = int(sys.argv[4])

collection = list()

for i in range(0,num_graph):
    collection.append(Graph(i,n,r,x))

union = UnionGraph(collection)

drawGraph(union.graph)      # Esto no funciona