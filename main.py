"""
TODOs generales:
    - TODO Reestructurar el execute.sh con parámetros globales que definan los parámetros del main
    - TODO Pensar en la estructura de ejecución general del main() para que se pueda adaptar al bash
    - TODO Documentar todas las funciones, con sus @param, @return y la función que hacen (mirar online documentación en python completa)
    - TODO Rellenar con todos los paquetes necesarios para pip install en execute.sh
    - TODO Paralelizar la generación de los grafos, así como la obtención de su información
    
    
    - TODO [HECHO] obtener el grafo unión en diferenets fases de la unión (grafo con 2, con 10, con 50...) y ver cómo se conectan (memoria)
    - TODO ver la progresiçon de las propiedades dada la variación de los radios (r), comparando con las que tendría el grafo con una sola capa
        y después dependiendo del número de capas (crecimiento lineal? exponencial?) -> Primero parámetros del grafo (r) y luego del multilayer
        (núm capas)
        
        Vamos de 0.01 a 0.1 en r y vemos los cambios (intervalos de 0.05) y después hacer zoom en una zona si hay un cambio brusco (cambio en el
        crecimiento)
        
    - TODO a partir de los atributos del dataframe, presentarlos de manera de tabla y ver el crecimiento (IMPORTANTE)
    - TODO hacer una constructora de multicapa que te devuelva una lista con la progresión de un atributo concreto (o todos)
    
    [MEJORAS EN EL CÓDIGO]
    - TODO en multicapa, estamos guardando en cada grafo sencillo los mismos parámetros una y orta vez (n, x, r). Quizá hay una manera de mejorarlo
    
    [EXTRA]
    - TODO Consultar una manera de guardar los plt.show() de manera local (archivos png)
"""

# Packages
import matplotlib.pyplot as plt
import networkx as nx
import argparse as ap

from graph import Graph, MultilayerGraph
from config import Config

# Functions

## Utils

def drawKCore(kcore) -> plt.Figure:
        """
        Imprimeix per pantalla el k-core donat.
        
        - TODO: NO FUNCIONA HACER DE NUEVOgraphs
        """
        
        return
    
## Main utilities
def config() -> None:
    """
    Funció per fer configuracions previes a l'execució de l'script.\n
    Carrega a l'objecte de la classe Config els paràmetres per executar l'script.
    """
    global conf
    global collection
    global multilayer
    
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
    
    collection = [Graph(i,conf.n,conf.r_ini,conf.x) for i in range(conf.num_graph)]
    multilayer = MultilayerGraph(collection)
    
    return

# Tests

def multilayerEvolution(n: int) -> None:
    """
    Test que, donat un valor enter, obté la progressió del graf unió i imprimeix el graf cada n capes afegides.
    Serveix per veure com evoluciona gràficament, la imatge es va carregant de vèrtexos.s
    """
    plots = multilayer.seeProgression(rang=n)
    return

def parameterEvolution() -> None:   # Funciona menos el k_core
    """
    Test que, donades les condicions d'entrada del programa, imprimeix per pantalla un estudi de com evoluciona 
    el graf multicapa depenent del número de capes.
    
    - TODO retornar també un dataframe amb la informació (una columna per atribut)
    """
    plots = multilayer.getGraphics()
    #k_core = drawRandomGeometricGraph(plots["K-core_graph"])
    return

def radiusEvolution() -> None:
    """
    Test que, donat el rang inicial, final i els intervals donats per la configuració global, obtenim diferents grafs amb els mateixos
    nodes però amb diferents adjacències, determinades pel nou radi
    """
    plots = multilayer.radiusProgression(conf.r_ini, conf.r_fin, conf.radius_add)
    return

# Main
    
def main() -> None:
    """
    Main script
    """
    # Configuració global
    config()
       
    # Script
    opts = [c for c in conf.test]
    if conf.test == "000":  # Default execution (no test, only random code)
        df = Graph.getInfo(multilayer)
        print("Dataframe for multilayer graph:")
        print(df)
        
    if opts[0] == '1':
        n: int = 10
        multilayerEvolution(n)
    
    if opts[1] == '1':
        radiusEvolution()
        
    if opts[2] == '1':
        parameterEvolution()
        
    plt.show()
    return

# Main script
main()