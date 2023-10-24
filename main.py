"""
TODOs generales:
    - TODO Reestructurar el execute.sh con parámetros globales que definan los parámetros del main
    - TODO Pensar en la estructura de ejecución general del main() para que se pueda adaptar al bash
    - TODO Documentar todas las funciones, con sus @param, @return y la función que hacen (mirar online documentación en python completa)
    - TODO Rellenar con todos los paquetes necesarios para pip install en execute.sh
    - TODO Paralelizar la generación de los grafos, así como la obtención de su información
    
    
    - TODO obtener el grafo unión en diferenets fases de la unión (grafo con 2, con 10, con 50...) y ver cómo se conectan (memoria)
    - TODO ver la progresiçon de las propiedades dada la variación de los radios (r), comparando con las que tendría el grafo con una sola capa
        y después dependiendo del número de capas (crecimiento lineal? exponencial?) -> Primero parámetros del grafo (r) y luego del multilayer
        (núm capas)
        
        Vamos de 0.01 a 0.1 en r y vemos los cambios (intervalos de 0.05) y después hacer zoom en una zona si hay un cambio brusco (cambio en el
        crecimiento)
        
    - TODO a partir de los atributos del dataframe, presentarlos de manera de tabla y ver el crecimiento (IMPORTANTE)
    - TODO en multicapa, estamos guardando en cada grafo sencillo los mismos parámetros una y orta vez (n, x, r). Quizá hay una manera de mejorarlo
"""

# Packages
import matplotlib.pyplot as plt
import networkx as nx
import argparse as ap

from graph import Graph, MultilayerGraph
from config import Config

# Functions

## Utils    
def drawRandomGeometricGraph(Gph: Graph) -> None:
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
def config() -> None:
    """
    Funció per fer configuracions previes a l'execució de l'script.\n
    Carrega a l'objecte de la classe Config els paràmetres per executar l'script.\n
    """
    global conf
    
    parser = ap.ArgumentParser(
        prog="main.py",
        description="Script principal que computa una sèrie de grafs geomètrics aleatoris, i amb ells en crea un multicapa."
    )
    # Parser for test parameter
    parser.add_argument(
        '-test', 
        help="Tipus de test que executem. El defecte és 000, que executa el que hi ha al main.",
        type=str,
        default="000",
        dest="test"
    )
    # Parser for n parameter
    parser.add_argument(
        '-n', 
        help="Número de vèrtexs que contindrà cada graf",
        type=int,
        default=100,
        dest="n"
    )
    # Parser for x parameter
    parser.add_argument(
        '-x', 
        help="Dimensió del quadrat que conté el graf",
        type=float,
        default=2.0,
        dest="x"
    )
    # Parser for r_ini parameter
    parser.add_argument(
        '-r_ini', 
        help="Radi a partir del qual es generen les circumferències per crear adjacències al graf",
        type=float,
        default=0.01,
        dest="r_ini"
    )
    # Parser for r_fin parameter
    parser.add_argument(
        '-r_fin', 
        help="Radi final del test",
        type=float,
        default=0.1,
        dest="r_fin"
    )
    # Parser for radius_add parameter
    parser.add_argument(
        '-radius_add', 
        help="Número que defineix la diferència que hi ha entre valor i valor en l'estudi d'un rang al radi",
        type=float,
        default=0.05,
        dest="radius_add"
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

# Tests

def parameterEvolution():
    
    return

def radiusTest():
    
    return

# Main
    
def main() -> None:
    """
    Main script
    """
    # Configuració global
    config()
       
    # Script
    if conf.test == "000":  # Default execution (no test, only random code)
        
        collection = [Graph(i,conf.n,conf.r_ini,conf.x) for i in range(conf.num_graph)]
        union = MultilayerGraph(collection)
        df = union.getInfo()
        print("Dataframe for multilayer graph:")
        print(df)
        
        # union.saveXML("Prueba1", True)
        drawRandomGeometricGraph(union)
        
    elif conf.test == "001":
        parameterEvolution()
    
    elif conf.test == "010":
        radiusTest()
    
    # Comment zone
    """
    
    """
    return

# Main script
main()