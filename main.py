# Packages
import sys
import matplotlib.pyplot as plt
import networkx as nx

from graph import Graph, UnionGraph
from config import Config

import random

# Global parameters

conf = Config()

# Functions
    
def drawRandomGeometricGraph(Gph: Graph):
    """TODO: Rehacer el código para imprimir más bonito (IMPORTANTE)
    
    Imprimeix per pantalla el graf aleatori geomètric
    
    Codi importat directament des d'el web de NetworkX: 
    https://networkx.org/documentation/stable/auto_examples/drawing/plot_random_geometric_graph.html"""
    
    G = Gph.graph
    pos = nx.get_node_attributes(G, "pos")

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), alpha=0.4)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=G.nodes(),
        node_size=20,
        cmap=plt.cm.Reds_r,
    )

    plt.xlim(-0.05, Gph.x + 0.05)   
    plt.ylim(-0.05, Gph.x + 0.05)   
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
    try:
        if len(sys.argv) > 1:
            conf.n = int(sys.argv[1])
        if len(sys.argv) > 2:
            conf.x = float(sys.argv[2])
        if len(sys.argv) > 3:
            conf.r = float(sys.argv[3])
        if len(sys.argv) > 4:
            conf.num_graph = int(sys.argv[4])
    except:
        print("El format d'algun paràmetre és erroni. Si us plau intenta-ho un altre cop")
        exit(1)
        
    # Script
    collection = []
    for i in range(0,conf.num_graph):
        collection.append(Graph(i,conf.n,conf.r,conf.x))

    union = UnionGraph(collection)
    drawRandomGeometricGraph(union)
    
    # Comment zone
    """
    
    """

# Main script
main()