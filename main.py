# Packages
import sys
import matplotlib.pyplot as plt
import networkx as nx

from graph import Graph, UnionGraph
from config import Config

import random

# Functions

def drawSimpleGraph(G: nx.graph):
    """Donat un graf, imprimeix per pantalla la seva representació gràfica"""
    # TODO: REDO this function
    subax1 = plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    subax2 = plt.subplot(122)
    nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
    plt.show()
    
def drawRandomGeometricGraph(G: nx.graph):
    pos = nx.get_node_attributes(G, "pos")
    
    # find node near center (0.5,0.5)
    dmin = 1
    ncenter = 0
    for n in pos:
        x, y = pos[n]
        d = (x - 0.5) ** 2 + (y - 0.5) ** 2
        if d < dmin:
            ncenter = n
            dmin = d

    # color by path length from node near center
    p = dict(nx.single_source_shortest_path_length(G, ncenter))

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(p.keys()),
        node_size=80,
        node_color=list(p.values()),
        cmap=plt.cm.Reds_r,
    )

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis("off")
    plt.show()
    
def main():
    """Main script"""
    
    # Usage
    if len(sys.argv) > 1 and (sys.argv[1] == "--help" or sys.argv[1] == "-h" or sys.argv[1] == "--usage"):
        """Utilització de l'script"""
        # TODO
        # Usage printing
        exit(1) 
        
    # Parameter parsing and introduction
    conf = Config()

    try:
        if len(sys.argv) > 1:
            conf.n = int(sys.argv[1])
        if len(sys.argv) > 2:
            conf.x = int(sys.argv[2])
        if len(sys.argv) > 3:
            conf.r = float(sys.argv[3])
        if len(sys.argv) > 4:
            conf.num_graph = int(sys.argv[4])
    except:
        print("El format d'algun paràmetre és erroni. Si us plau intenta-ho un altre cop")
        exit(1)
        
    collection = []
    for i in range(0,conf.num_graph):
        collection.append(Graph(i,conf.n,conf.r,conf.x))

    union = UnionGraph(collection)
    drawRandomGeometricGraph(collection[0].graph)        # Esto no funciona
        
    # Script
    
    # Comment zone
    
    """
    
    """
    

# Main script
main()