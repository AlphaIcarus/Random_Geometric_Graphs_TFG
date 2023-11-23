"""
TODOs generales:
    - TODO Pensar en la estructura de ejecución general del main() para que se pueda adaptar al bash
    - TODO Documentar todas las funciones, con sus @param, @return y la función que hacen (mirar online documentación en python completa)
    - TODO Rellenar con todos los paquetes necesarios para pip install en execute.sh
    
    - TODO ver la progresión de las propiedades dada la variación de los radios (r), comparando con las que tendría el grafo con una sola capa
        y después dependiendo del número de capas (crecimiento lineal? exponencial?) -> Primero parámetros del grafo (r) y luego del multilayer
        (núm capas)
        
        Vamos de 0.01 a 0.1 en r y vemos los cambios (intervalos de 0.005) y después hacer zoom en una zona si hay un cambio brusco (cambio en el
        crecimiento)
    
    REUNIÓN 13 NOV
    
    - TODO Mirar el número de vértices del K-core (con k máxima) y mirar la evolución de la K
    - TODO Hacer varias ejecuciones (2, 5, 10...) y hacer la media en los valores de los dataframes
    
    [MEJORAS EN EL CÓDIGO]
    - TODO en multicapa, estamos guardando en cada grafo sencillo los mismos parámetros una y orta vez (n, x, r). Quizá hay una manera de mejorarlo
        · Hay que cambiar el atributo graphList para guardar únicamente el grafo de NetworkX y no una lista de Graph    
    
    [EXTRA]
"""

# Packages
import matplotlib.pyplot as plt
import networkx as nx
import argparse as ap
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed

from graph import Graph, MultilayerGraph
from config import Config

from datetime import datetime
from time import time

# Parameters

now = str(datetime.now()).replace(":",".")      # Necessari pel format de les carpetes
n_values = [1000,2000]                          # Valors dels ordres del test 2

# Auxiliar functions

## Dataframe Utils

def dataframeMean(dfList: [pd.DataFrame]) -> pd.DataFrame:
    """
    Funció auxiliar que, donada una llista de dataframes amb les mateixes columnes i el mateix número de files i columnes,
    retorna un dataframe on cada valor df[j][k] és la mitjana de tots els valors i-èssim df_i[j][k]
    """
    nDataFrames = conf.num_copies   # Number of dataframes
    df = dfList[0]                  # Final dataframe

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
    
    - TODO crear un parámetro 'special_case' con valor None por defecto donde especificar casos para crear gráficos específicos
    """
    fig, ax = plt.subplots()
    ax.plot(xvalues, yvalues)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    ax.set_xlim(left=0)
    dir = ".\\test_output\\" + now
    
    try:
        os.mkdir(dir)
    except(FileExistsError):
        pass
    
    fig.savefig(dir + "\\" + ylabel + ".png") # Save figure
    plt.close()     # Figure closing due to overload
    return fig

def drawAndStoreMultipleLinearGraphic(xvalues: list, yvalues: list[list], xlabel: str, ylabel: str, title: str) -> plt.Figure:
    """
    Funció auxiliar que, donats uns valors pels eixos, títols i títol de figura, retorna la figura resultant i la guarda a memòria.
    
    - TODO crear un parámetro 'special_case' con valor None por defecto donde especificar casos para crear gráficos específicos
    """
    fig, ax = plt.subplots()
    
    [ax.plot(xvalues, yvalues[i], label="Order: " + str(n_values[i])) for i in len(yvalues)]
     
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    ax.legend()
    ax.set_xlim(left=0)
    
    dir = ".\\test_output\\" + now
    try:
        os.mkdir(dir)
    except(FileExistsError):
        pass
    fig.savefig(dir + "\\" + ylabel + ".png")   # Save figure
    
    plt.close()                                 # Figure closing due to overload
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
    - TODO Aún no funciona
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
    
    - TODO quizá sea interesante mandar todo a la constructora de Config
    """
    global conf
    global multilayer
        
    parser = ap.ArgumentParser(
        prog="main.py",
        description="Script principal que computa una sèrie de grafs geomètrics aleatoris, i amb ells en crea un multicapa."
    )
    # Parser for test parameter
    parser.add_argument(
        '-test', 
        help="Tipus de test que executem. L'execució per defecte és 'default', o el número de test.",
        type=str,
        default="default",
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
        type=np.double,
        default=1.0,
        dest="x"
    )
    # Parser for r_ini parameter
    parser.add_argument(
        '-r_ini', 
        help="Radi a partir del qual es generen les circumferències per crear adjacències al graf",
        type=np.double,
        default=0.01,
        dest="r_ini"
    )
    # Parser for r_fin parameter
    parser.add_argument(
        '-r_fin', 
        help="Radi final del test",
        type=np.double,
        default=0.1,
        dest="r_fin"
    )
    # Parser for radius_add parameter
    parser.add_argument(
        '-radius_add', 
        help="Número que defineix la diferència que hi ha entre valor i valor en l'estudi d'un rang al radi",
        type=np.double,
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
    # Parser for num_copies parameter
    parser.add_argument(
        '-num_copies', 
        help="Número de grafs multicapa a generar",
        type=int,
        default=1,
        dest="num_copies"
    )
    
    args = parser.parse_args()
    conf = Config(args)
    multilayer = []
    for i in range(conf.num_copies):
        collection = [Graph(i,conf.n,conf.r_ini,conf.x) for i in range(conf.num_graph)]
        multilayer.append(MultilayerGraph(collection, default_build=True))
    
    return

### Tests

def multilayerEvolution(n: int) -> None:
    """
    Test que, donat un valor enter, obté la progressió del graf unió i guarda el graf cada n capes afegides.
    Serveix per veure com evoluciona gràficament, la imatge es va carregant de vèrtexos.s
    
    - TODO cambiar a list de multilayer
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
    
    # [savePlots(figure, name) for name, figure in zip(map(str,xvalues), plots)]   # Graph figure saving (NO ES NECESARIO, YA LOS GUARDAMOS DONDE ANTES)
    return

def parameterEvolution() -> None:   # Funciona menos el k_core
    """
    Test que, donades les condicions d'entrada del programa, imprimeix per pantalla un estudi de com evoluciona 
    el graf multicapa depenent del número de capes.
    
    - TODO retornar també un dataframe amb la informació (una columna per atribut)
    """
    test: str = f"Parameter evolution for multilayer with {conf.num_graph} layers"
    
    # Plot generation
    tags, plots = multilayer.getParameterProgression()
    # k_core = kCorePlot(multilayer.graph)
    
    # Plot saving
    [savePlots(plot, name) for name, plot in zip(tags, plots)]
    # savePlots(k_core, "K-core graph of multilayer")

    return

def radiusEvolution() -> None:
    """
    python3 main.py -test 010 -n 1000 -r_ini 0.1 -r_fin 0.2 -radius_add 0.005 -num_graph 20
    
    Test que, donat el rang inicial, final i els intervals donats per la configuració global, obtenim el dataframe amb les propietats
    del graf per cada radi i-èssim de l'interval [r_ini, r_fin, +r_add]
    
    - TODO cambiar a list de multilayer
    """
    test: str = f"Radius evol in multilayer for i-th radius [{conf.r_ini},{conf.r_fin},+{conf.radius_add}] "
    
    # df = multilayer.radiusProgression(conf.r_ini, conf.r_fin, conf.radius_add)
    # print(df)
    dfs = [multilayer[i].radiusProgression(conf.r_ini, conf.r_fin, conf.radius_add) for i in range(conf.num_copies)]
    df = dataframeMean(dfs)
      
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

### Project Tests

def test1() -> None:
    """
    First test. We want to analyse the changes of multilayer graph's properties changing the radius before generating
    the layers of the multilayer. Then we combine them and get results.
    
    The values of the test are given by conf.r_ini, conf.r_fin and conf.radius_add. It is translated to:
    
    PSEUDOCODE:
    
    xValues = np.arange(conf.r_ini, conf.r_fin, conf.radius_add)
    plots := generate_empty_plots()
    for r in r_values:
        mls := generate_multilayers()
        dataframes += get_properties_datasets(mls)
    end for
    df := dataset_mean(dataframes)
    add_values_to_plots(plot, xValues, df)  # Every plot has xValues for x dimension, and df[key] for y dimension.
    save_plots()
    return
    """
    radiusEvolution()   # Here is the code
    return

def test2() -> None:
    """
    Second test. We want to analyse multilayer's properties changing radius and number of nodes (r, n)
    
    Given default number of nodes config.n, we get the mean of the datasets given by generate config.num_copies datasets 
    of these number of multilayer graphs, and then do the same for some other values of number of nodes (n)
    
    Thus, we create as many graphics as properties of the multilayer graph, but each line with each colour represents
    a nuber of nodes this test has followed to create the multilayers.
    
    PSEUDOCODE:
    
    xValues = np.arange(conf.r_ini, conf.r_fin, conf.radius_add)
    plots := generate_empty_plots()
    for n in n_values:
        for r in r_values:
            mls := generate_multilayers()
            dataframes += get_properties_datasets(mls)
        end for
        df := dataset_mean(dataframes)
        add_values_to_plots(plot, xValues, df) # Every plot has xValues for x dimension, and df[key]s for y dimension (more than one progression)
    end for
    save_plots()
    return
    """
    test: str = f"Radius evolution for different orders of multilayer"
    
    # n_values = [1000,2000,3000,4000,5000,10000,20000]     # Valors de l'ordre del graf
    n_values = [1000,1000]
    
    print(1)
    dfs = []
    
    for n in n_values:
        print(2)
        rawDfs = []
        for _ in range(conf.num_copies):
            collection = [Graph(i,n,conf.r_ini,conf.x) for i in range(conf.num_graph)]
            ml = MultilayerGraph(collection, default_build=True)
            rawDfs.append(ml.radiusProgression(conf.r_ini, conf.r_fin, conf.radius_add))
        dfs.append(dataframeMean(rawDfs))
        print(type(dfs), len(dfs))
        print(4)
        
    xlabel = "Radius values"
    xvalues = np.arange(conf.r_ini, conf.r_fin, conf.radius_add)
        
    print(5)
    
    df = [dfs[i]["Size"] for i in range(len(dfs))]
    size_evol_plot = drawAndStoreMultipleLinearGraphic(xvalues=xvalues, yvalues=df, 
                                         xlabel=xlabel, ylabel="Size of multilayer", title=test)
    df = [dfs[i]["Is_connected"] for i in range(len(dfs))]
    connection_evol_plot = drawAndStoreMultipleLinearGraphic(xvalues=xvalues, yvalues=df, 
                                         xlabel=xlabel, ylabel="Is connected", title=test)
    df = [dfs[i]["Connected_components"] for i in range(len(dfs))]
    number_cc_evol_plot = drawAndStoreMultipleLinearGraphic(xvalues=xvalues, yvalues=df, 
                                         xlabel=xlabel, ylabel="Number of Connected components", title=test)
    df = [dfs[i]["Largest_component_diameter"] for i in range(len(dfs))]
    largest_component_diameter_plot = drawAndStoreMultipleLinearGraphic(xvalues=xvalues, yvalues=df, 
                                         xlabel=xlabel, ylabel="Largest_component_diameter", title=test)
    df = [dfs[i]["Radius"] for i in range(len(dfs))]
    radius_evol_plot = drawAndStoreMultipleLinearGraphic(xvalues=xvalues, yvalues=df, 
                                         xlabel=xlabel, ylabel="Radius of graph", title=test)
    df = [dfs[i]["Diameter"] for i in range(len(dfs))]
    diameter_evol_plot = drawAndStoreMultipleLinearGraphic(xvalues=xvalues, yvalues=df, 
                                         xlabel=xlabel, ylabel="Diameter of graph", title=test)
    df = [dfs[i]["Is_eulerian"] for i in range(len(dfs))]
    eulerian_evol_plot = drawAndStoreMultipleLinearGraphic(xvalues=xvalues, yvalues=df, 
                                         xlabel=xlabel, ylabel="Is eulerian", title=test)
    df = [dfs[i]["Min_degree"] for i in range(len(dfs))]
    min_degree_evol_plot = drawAndStoreMultipleLinearGraphic(xvalues=xvalues, yvalues=df, 
                                         xlabel=xlabel, ylabel="Minimum degree in graph", title=test)
    df = [dfs[i]["Max_degree"] for i in range(len(dfs))]
    max_degree_evol_plot = drawAndStoreMultipleLinearGraphic(xvalues=xvalues, yvalues=df, 
                                         xlabel=xlabel, ylabel="Maximum degree in graph", title=test)
    df = [dfs[i]["Average_Clustering_Coefficient"] for i in range(len(dfs))]
    acc_evol_plot = drawAndStoreMultipleLinearGraphic(xvalues=xvalues, yvalues=df, 
                                         xlabel=xlabel, ylabel="Average clustering coefficient in graph", title=test)
    df = [dfs[i]["Triangle_number"] for i in range(len(dfs))]
    triangle_number_evol_plot = drawAndStoreMultipleLinearGraphic(xvalues=xvalues, yvalues=df, 
                                         xlabel=xlabel, ylabel="Triangle number of graph", title=test)
    
    return

### Main
    
def main() -> None:
    """
    Main script
    """
    ini = time()
    config()                                            # Configuració global    
    if conf.test == "default":                          # Default execution (no test, only random code)
        """
        Execució per defecte. S'executa quan volem fer petites proves sobre el codi
        """
        df = Graph.getInfo(multilayer)
        print("Dataframe for multilayer graph:")
        print(df)
        
        plot = multilayer.drawRandomGeometricGraph()
    if conf.test == '1':
        """
        Test 1. Veiem la progressió dels atributs del multicapa variant valors del radi.
        """
        test1()
    if conf.test == '2':
        """
        Test 2.
        """
        test2()
        #radiusEvolution()
    if conf.test == '3':
        """
        Test 3.
        """
        parameterEvolution()
    # plt.show()
    fin = time()
    print(f"L'script ha trigat {fin-ini} segons")
    return

main()                                                  # Main script