"""
TODOs generales:
    - TODO Reestructurar el execute.sh con parámetros globales que definan los parámetros del main
    - TODO Pensar en la estructura de ejecución general del main() para que se pueda adaptar al bash
    - TODO Documentar todas las funciones, con sus @param, @return y la función que hacen (mirar online documentación en python completa)
    - TODO Rellenar con todos los paquetes necesarios para pip install en execute.sh
    - TODO Paralelizar la generación de los grafos, así como la obtención de su información
    
    
    - TODO [HECHO] obtener el grafo unión en diferenets fases de la unión (grafo con 2, con 10, con 50...) y ver cómo se conectan (memoria)
    - TODO ver la progresión de las propiedades dada la variación de los radios (r), comparando con las que tendría el grafo con una sola capa
        y después dependiendo del número de capas (crecimiento lineal? exponencial?) -> Primero parámetros del grafo (r) y luego del multilayer
        (núm capas)
        
        Vamos de 0.01 a 0.1 en r y vemos los cambios (intervalos de 0.05) y después hacer zoom en una zona si hay un cambio brusco (cambio en el
        crecimiento)
        
    - TODO a partir de los atributos del dataframe, presentarlos de manera de tabla y ver el crecimiento (IMPORTANTE)
    - TODO hacer una constructora de multicapa que te devuelva una lista con la progresión de un atributo concreto (o todos)
    
    REUNIÓN 13 NOV
    
    - TODO Mirar el número de vértices del K-core (con k máxima) y mirar la evolución de la K
    - TODO Hacer varias ejecuciones (2, 5, 10...) y hacer la media en los valores de los dataframes
    
    [MEJORAS EN EL CÓDIGO]
    - TODO en multicapa, estamos guardando en cada grafo sencillo los mismos parámetros una y orta vez (n, x, r). Quizá hay una manera de mejorarlo
        · Hay que cambiar el atributo graphList para guardar únicamente el grafo de NetworkX y no una lista de Graph    
    
    [EXTRA]
    - TODO Consultar una manera de guardar los plt.show() de manera local (archivos png)
"""

# Packages
import matplotlib.pyplot as plt
import networkx as nx
import argparse as ap
import numpy as np
import os
import pandas as pd

from graph import Graph, MultilayerGraph
from config import Config
from collections import defaultdict

from datetime import datetime

# Parameters

now = str(datetime.now()).replace(":",".")      # Necessari pel format de les carpetes

# Auxiliar functions

## Dataframe Utils

def dataframeMean(dfList: [pd.DataFrame]) -> pd.DataFrame:
    """
    Funció auxiliar que, donada una llista de dataframes amb les mateixes columnes i el mateix número de files i columnes,
    retorna un dataframe on cada valor df[j][k] és la mitjana de tots els valors i-èssim df_i[j][k]
    """
    nDataFrames = len(dfList)   # Number of dataframes
    df = dfList[0]              # Final dataframe

    # FIRST METHOD
    for dataframe in dfList[1:]:
        df = df.add(dataframe)
    df = df.div(nDataFrames)

    # ALT METHOD
    """
    # For every dataframe, we sum every i-th-j-th value to the i-th-j-th value of every other dataframe
    for dataframe in dfList[1:]:    # For every dataframe
        for col in dataframe.columns:  # For every column
            for row in dataframe.index:
                df[col][row] += dataframe[col][row] 
    
    # For every value in final dataframe, we divide the value by the number of dataframes
    for col in df.columns:  # For every column
        for row in df.index:
            df[col][row] /= nDataFrames
    """
    return df

## Plot generation and other operations

def drawAndStoreGraphic(xvalues: list, yvalues: list, xlabel: str, ylabel: str, title: str) -> plt.Figure:
    """
    Funció auxiliar que, donats uns valors pels eixos, títols i títol de figura, retorna la figura resultant i la guarda a memòria.
    """
    
    fig, ax = plt.subplots()
    ax.plot(xvalues, yvalues)

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid() # Optional

    dir = ".\\test_output\\" + now
    
    try:
        os.mkdir(dir)
    except(FileExistsError):
        pass
    
    fig.savefig(dir + "\\" + ylabel + ".png") # Save figure
    # fig.savefig(ylabel + ".png") # Save figure

    plt.close()     # Figure closing due to overload
    return fig

def savePlots(fig: plt.Figure, fileName: str) -> None:
    """
    Funció auxiliar que guarda un gràfic en una ubicació determinada per folderPath
    """
    dir = ".\\test_output\\" + now
    try:
        os.mkdir(dir)
    except(FileExistsError):
        pass
    
    fig.savefig(dir + "\\" + fileName + ".png") # Save figure
    return

def kCorePlot(G: nx.Graph) -> plt.Figure:
    """
    Funció auxiliar que retorna el k-core del graf donat. k és maximal per defecte.
    
    Imported code from: https://stackoverflow.com/questions/70297329/visualization-of-k-cores-using-networkx
        - TODO Aún no sé si funciona
    """
    # build a dictionary of k-level with the list of nodes
    kcores = defaultdict(list)
    for n, k in nx.core_number(G).items():
        kcores[k].append(n)

    # compute position of each node with shell layout
    pos = nx.layout.shell_layout(G, list(kcores.values()))
    colors = {1: 'red', 2: 'green'}  # need to be improved, for demo

    # draw nodes, edges and labels
    for kcore, nodes in kcores.items():
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors[kcore])
    nx.draw_networkx_edges(G, pos, width=0.2)
    nx.draw_networkx_labels(G, pos)
    plot = plt.gcf()

    return plot
    
## Main utilities
def config() -> None:
    """
    Funció principal per fer configuracions previes a l'execució de l'script.
    Carrega a l'objecte global de la classe Config els paràmetres per executar l'script.
    """
    global conf
    global collection
    global multilayer
    global name_of_tests
    
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
        default=1.0,
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
    multilayer = MultilayerGraph(collection, default_build=True)
    
    return

### Tests

def multilayerEvolution(n: int) -> None:
    """
    Test que, donat un valor enter, obté la progressió del graf unió i guarda el graf cada n capes afegides.
    Serveix per veure com evoluciona gràficament, la imatge es va carregant de vèrtexos.s
    """
    test: str = f"Multilayer evolution by adding {n} layers"
    df, plots = multilayer.seeProgression(rang=n)       # PArameter dataframe and graphs plotted

    # Parameter plotting
    xlabel = "Layers added to multilayer"
    xvalues = range(1, conf.num_graph, n)
    
    size_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Size"], 
                                         xlabel=xlabel, ylabel="Size of multilayer", title=test)
    connection_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Is_connected"], 
                                         xlabel=xlabel, ylabel="Is connected", title=test)
    number_cc_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Connected_components"], 
                                         xlabel=xlabel, ylabel="Number of Connected components", title=test)
    largest_component_diameter_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Largest_component_diameter"], 
                                         xlabel=xlabel, ylabel="Largest_component_diameter", title=test)
    radius_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Radius"], 
                                         xlabel=xlabel, ylabel="Radius of graph", title=test)
    diameter_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Diameter"], 
                                         xlabel=xlabel, ylabel="Diameter of graph", title=test)
    eulerian_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Is_eulerian"], 
                                         xlabel=xlabel, ylabel="Is eulerian", title=test)
    min_degree_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Min_degree"], 
                                         xlabel=xlabel, ylabel="Minimum degree in graph", title=test)
    max_degree_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Max_degree"], 
                                         xlabel=xlabel, ylabel="Maximum degree in graph", title=test)
    acc_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Average_Clustering_Coefficient"], 
                                         xlabel=xlabel, ylabel="Average clustering coefficient in graph", title=test)
    triangle_number_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Triangle_number"], 
                                         xlabel=xlabel, ylabel="Triangle number of graph", title=test)    
    
    [savePlots(figure, name) for name, figure in zip(map(str,xvalues), plots)]   # Graph figure saving
    return

def parameterEvolution() -> None:   # Funciona menos el k_core
    """
    Test que, donades les condicions d'entrada del programa, imprimeix per pantalla un estudi de com evoluciona 
    el graf multicapa depenent del número de capes.
    
    - TODO retornar també un dataframe amb la informació (una columna per atribut)
    """
    test: str = f"Parameter evolution for ultilayer with {conf.num_graph} layers"
    
    # Plot generation
    plots = multilayer.getParameterProgression()
    k_core = kCorePlot(multilayer.graph)
    
    # Plot saving
    [savePlots(plot, name) for name, plot in zip(plots.keys(), plots.values())]
    savePlots(k_core, "K-core graph of multilayer")

    return

def radiusEvolution() -> None:
    """
    Test que, donat el rang inicial, final i els intervals donats per la configuració global, obtenim el dataframe amb les propietats
    del graf per cada radi i-èssim de l'interval [r_ini, r_fin, +r_add]
    """
    test: str = f"Radius evol in multilayer for i-th radius [{conf.r_ini},{conf.r_fin},+{conf.radius_add}] "
    
    df = multilayer.radiusProgression(conf.r_ini, conf.r_fin, conf.radius_add)
    print(df)
    
    xlabel = "Radius values"
    xvalues = np.arange(conf.r_ini, conf.r_fin, conf.radius_add)
    
    size_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Size"], 
                                         xlabel=xlabel, ylabel="Size of multilayer", title=test)
    connection_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Is_connected"], 
                                         xlabel=xlabel, ylabel="Is connected", title=test)
    number_cc_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Connected_components"], 
                                         xlabel=xlabel, ylabel="Number of Connected components", title=test)
    largest_component_diameter_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Largest_component_diameter"], 
                                         xlabel=xlabel, ylabel="Largest_component_diameter", title=test)
    radius_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Radius"], 
                                         xlabel=xlabel, ylabel="Radius of graph", title=test)
    diameter_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Diameter"], 
                                         xlabel=xlabel, ylabel="Diameter of graph", title=test)
    eulerian_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Is_eulerian"], 
                                         xlabel=xlabel, ylabel="Is eulerian", title=test)
    min_degree_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Min_degree"], 
                                         xlabel=xlabel, ylabel="Minimum degree in graph", title=test)
    max_degree_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Max_degree"], 
                                         xlabel=xlabel, ylabel="Maximum degree in graph", title=test)
    acc_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Average_Clustering_Coefficient"], 
                                         xlabel=xlabel, ylabel="Average clustering coefficient in graph", title=test)
    triangle_number_evol_plot = drawAndStoreGraphic(xvalues=xvalues, yvalues=df["Triangle_number"], 
                                         xlabel=xlabel, ylabel="Triangle number of graph", title=test)
    return

### Main
    
def main() -> None:
    """
    Main script
    """
    config()                                            # Configuració global
    opts = [c for c in conf.test]                       # Script
    
    if conf.test == "000":                              # Default execution (no test, only random code)
        df = Graph.getInfo(multilayer)
        print("Dataframe for multilayer graph:")
        print(df)
        
        plot = multilayer.drawRandomGeometricGraph()
    if opts[0] == '1':
        n: int = 2
        multilayerEvolution(n)
    if opts[1] == '1':
        radiusEvolution()
    if opts[2] == '1':
        parameterEvolution()

    # plt.show()
    return

main()                                                  # Main script