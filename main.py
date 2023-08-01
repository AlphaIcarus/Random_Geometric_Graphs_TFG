# Packages
import matplotlib.pyplot as plt
import networkx as nx
import argparse as ap

from graph import Graph, MultilayerGraph
from config import Config

# Global parameters



# Functions

## Utils
    
def drawRandomGeometricGraph(Gph: Graph):
    """TODO: Rehacer el código para imprimir más bonito
    
    Imprimeix per pantalla el graf aleatori geomètric
    
    Codi importat i adaptat des d'el web de NetworkX: 
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
    
## Main utilities

def config():
    """
    Funció per fer configuracions previes a l'execució de l'script:
    
    TODO detallar las funcionalidades exactas
    """
    global conf
    
    parser = ap.ArgumentParser(
        prog="main",
        description="Script principal que computa una sèrie de grafs geomètrics aleatoris, i amb ells en crea un multicapa."
    )
    # Parser for n parameter
    parser.add_argument(
        '-n', 
        help="Numero de vèrtexs que contindrà cada graf",
        type=int,
        default=100,
        dest="n"
    )
    # Parser for x parameter
    parser.add_argument(
        '-x', 
        help="Dimensió del quadrat que conté el graf",
        type=float,
        default=1.0,
        dest="x"
    )
    # Parser for r parameter
    parser.add_argument(
        '-r', 
        help="Radi a partir del qual es generen les circumferències per crear adjacències al graf",
        type=float,
        default=0.1,
        dest="r"
    )
    # Parser for num_graph parameter
    parser.add_argument(
        '-num_graph', 
        help="Número de grafs a generar",
        type=int,
        default=50,
        dest="num_graph"
    )
    
    args = parser.parse_args()
    conf = Config(args)
    
    return
    
def main():
    """Main script"""
       
    # Script
    collection = [Graph(i,conf.n,conf.r,conf.x) for i in range(0,conf.num_graph)]
    union = MultilayerGraph(collection)
    df = union.newGetInfo()
    print("Dataframe for multilayer graph:")
    print(df)
    
    # union.saveXML("Prueba1", True)
    drawRandomGeometricGraph(union)
    
    # Comment zone
    """
    
    """
    return

# Main script
config()
main()