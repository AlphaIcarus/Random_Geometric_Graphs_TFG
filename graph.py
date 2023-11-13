import networkx as nx
import random
import pandas as pd
import math
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from time import time

# Required by networkx.random_geometric_graph
import scipy

# Things to do:
# TODO añadir documentación detallada sobre los parámetros y funciones

# Utils

def drawGraph(xValues: list, yValues: list, title: str):
    """
    Funció per generar un gràfic fent servir Matplotlib: Per cada valor i-èssim de xValues i yValues, es denota un punt
    al gràfic.
    """
    assert(xValues == yValues)
    
    fig = None
        
    return fig

class Graph:

    def __init__(self, id: int, n: int, r: float, x: float, positions = None):
        """
        Constructora de la classe Graph. Es poden aportar les posicions per defecte.
        """
        self.id = id                                            # Identifier of the graph
        self.n = n                                              # Number of nodes of the graph
        self.x = x                                              # Longitude/Amplitude of the map
        self.r = r                                              # Radius of area to get adjacencies  
             
        if positions is not None:
            positions = {i: (random.uniform(0, x), random.uniform(0, x)) for i in range(n)}
        self.graph = nx.random_geometric_graph(n=n, radius=r, pos=positions)   # Graph itself 
    
        return
    
    # Getters
    
    def getInfo(self, index: int = 0) -> pd.DataFrame:
        """
        Mètode per obtenir 
        - TODO Rellenar el dataframme con toda la info de interés
        """
        
        # Degrees
        degrees = [val for (_, val) in self.graph.degree()]
        min_degree = min(degrees)
        max_degree = max(degrees)
        
        # Connected components
        connected_components = sorted(nx.connected_components(self.graph), key=len, reverse=True)   #Sorted from largest to smallest
        cc_sizes = map(len, connected_components)                                                   #Tamanys de les components connexes
        
        largest_component_nodes = max(nx.connected_components(self.graph), key=len)
        largest_component = self.graph.subgraph(largest_component_nodes)
        largest_c_diameter = nx.diameter(largest_component)
        
        dct = {
            "Order":self.graph.order(),
            "Size":self.graph.size(),
            "Is_connected":nx.is_connected(self.graph),
            "Connected_components":nx.number_connected_components(self.graph),
            "Largest_component_diameter":largest_c_diameter,
            "Radius":nx.radius(self.graph) if nx.is_connected(self.graph) else math.inf,
            "Diameter":nx.diameter(self.graph) if nx.is_connected(self.graph) else math.inf,
            "Is_eulerian":nx.is_eulerian(self.graph),
            "Min_degree":min_degree,
            "Max_degree":max_degree,
            "Average_Clustering_Coefficient":nx.average_clustering(self.graph),
            "Triangle_number": nx.triangles(self.graph, 0), # Esto devuelve un mapping, hay que hacerlo de alguna otra forma
            # Tamaños de componentes connexas
            # K-core (grafo donde todos los nodos tienen grado k, k máxima) --> se cosigue con k_core de NetworkX
                # Lo haremos en una función diferente y lo invocamos fuera
        }
        frame = pd.DataFrame(index=[index], data=dct)
        return frame
   
    # Graph printing
    
    def drawRandomGeometricGraph(self) -> plt.Figure:
        """
        Imprimeix per pantalla el graf aleatori geomètric
        
        Codi importat i adaptat des d'el web de NetworkX: 
        https://networkx.org/documentation/stable/auto_examples/drawing/plot_random_geometric_graph.html
        
        - TODO: Rehacer el código para imprimir más bonito
        """
        
        G = self.graph
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

        plt.xlim(-0.05, self.x + 0.05)   
        plt.ylim(-0.05, self.x + 0.05)   
        plt.axis("off")
        
        figure = plt.gcf()
        return figure

class MultilayerGraph(Graph):

    def __init__(self, graphs: list[Graph], default_build: bool = True):
        """
        Constructora de la classe. Construeix, a partir d'un conjunt de grafs aleatoris multicapa,
        un graf aleatori geomètric multicapa.
        """
        self.graphList: list[Graph] = graphs

        self.graph = None                                       # Graph itself
        self.id = graphs[0].id                                  # Identifier of the graph
        self.n = graphs[0].n                                    # Number of nodes of the graph
        self.x = graphs[0].x                                    # Longitude/Amplitude of the map
        self.r = graphs[0].r                                    # Radius of area to get adjacencies
        
        if default_build:
            self.buildMultilayer()
        return
    
    def buildMultilayer(self) -> None:
        """
        Mètode per contruir el graf multicapa. El resultat es guarda a l'atribut self.graph
        """
        self.graph = self.graphList[0].graph.copy()
        
        adjacencies = set()
        for i in range(len(self.graphList)):
            adjacencies = adjacencies.union(set(self.graphList[i].graph.edges())) 

        self.graph.add_edges_from((adjacencies))
        return
        
    def emptyMultilayer(self) -> None:
        """
        Mètode per buidar el paràmetre self.graph
        """
        self.graph = None
        return    
        
    # Getters
    
    def getInfo(self) -> pd.DataFrame:
        """
        Mètode per obtenir totes les dades d'interés del graf unió.
        
        Primer obtenim el dataframe del graf unió, posteriorment obtenim el de la seva col·lecció.
        
        TODO Puede ser interesante guardar la informacón como atributo en la propia clase
        TODO Rellenar el dataframme con toda la info de interés
        """
        df = [Graph.getInfo(self)]
        size = len(self.graphList)
        
        for i in range(size):
            graphDf = Graph.getInfo(self.graphList[i])
            df.append(graphDf)
            
        return pd.concat(df)
    
    # Testers
    
    def seeProgression(self, rang: int) -> (pd.DataFrame, [plt.Figure]):
        """
        Mètode per obtenir un data frame i imatges de com evoluciona el graf multicapa en diferents estats.
        També obtenim un dataframe amb la informació dels grafs intermitjos.
        
        - TODO actualmente se generan uno a uno, hay que obtener todas las imágenes de grafo y luego imprimirlas
        """
        
        inter = rang
        self.graph = self.graphList[0].graph.copy()
        
        plots = [self.drawRandomGeometricGraph()]               # Estat inicial 
        df = [Graph.getInfo(self)]                                # Data frame
        
        for graph in self.graphList:
            # Add new edges
            self.graph.add_edges_from(set(graph.graph.edges()))
            inter -= 1
            if inter == 0:
                df.append(Graph.getInfo(self))
                plots.append(self.drawRandomGeometricGraph())   # Print graph
                inter = rang
        
        df = pd.concat(df)
        return df, plots
    
    def radiusProgression(self, r_ini: float, r_fin: float, r_add: float) -> pd.DataFrame:
        """
        Mètode que, donat un radi inicial, un radi final i un valor, on r_ini + r_add <= r_fin, Va imprimint el graf multicapa fent servir
        els valors intermitjos de la progressió [r_ini, r_ini + r_add, r_ini + r_add*2, ... , r_ini + r_add*N <= r_fin] per N màxima.
        
        - TODO no funciona, tengo que generar los grafos de la colección de nuevo con sus posiciones, con cada radio intermedio,
            y luego generar el multicapa.
        """
        graphs = self.graphList.copy()                                                                   # Còpies dels valors originals per restaurar
        pos = [nx.get_node_attributes(self.graphList[i].graph, "pos") for i in range(len(self.graphList))]  # We get the nodes of every graph
        self.emptyMultilayer()                                                                              # Buidem el multicapa
        
        df = [] # Informació a retornar
        
        for radius in np.arange(r_ini,r_fin,r_add, dtype=float):
            """Aconseguim els grafs nous donats pel radi radius, construim el multicapa nou i el dibuixem"""
            print(radius)
            self.graphList = [Graph(i,graphs[0].n,radius,graphs[0].x, pos[i]) for i in range(len(self.graphList))]
            self.buildMultilayer()
            df.append(Graph.getInfo(self))
            self.emptyMultilayer()
            
        df = pd.concat(df)                         # Construim el dataframe
        # Recuperem les dades anteriors
        self.graphList = graphs
        self.buildMultilayer()
        
        return df
    
    def getGraphics(self) -> dict:
        """
        Mètode per obtenir els gràfics de com varien els atributs del graf multicapa, de manera progressiva.
        
        Els gràfics que s'obtenen venen donats per les propietats que volem obtenir del graf (donat a getInfo)
        
        - TODO: Los gráficos que no funcionan son 6 (radius), 7 (diameter)
        - TODO: Cambiar el funcionamiento para usar emptyMultilayer y buildMultilayer
        """
        # Paràmetres que volem estudiar la seva progressió
        g = self.graphList[0].graph.copy()
        
        layers = [i for i in range(1,len(self.graphList)+1)]
        order = []
        size = []
        is_connected = []
        number_connected_components = []
        largest_component_diameter = []
        radius = []
        diameter = []
        is_eulerian = []
        min_degree = []
        max_degree = []
        average_clustering_coefficient = []
        triangle_number = []
        
        for graph in self.graphList:
            g.add_edges_from((set(graph.graph.edges())))
            
            # We add the new attributes of the i-th step graph
            degrees = [degree[1] for degree in g.degree]
            min_d = min(degrees)
            max_d = max(degrees)
            
            largest_component_nodes = max(nx.connected_components(g), key=len)
            largest_component = g.subgraph(largest_component_nodes)
            largest_c_diameter = nx.diameter(largest_component)
            
            order.append(g.order())
            size.append(g.size())
            is_connected.append(1 if nx.is_connected(g) else 0)
            number_connected_components.append(nx.number_connected_components(g))
            largest_component_diameter.append(largest_c_diameter)
            radius.append(nx.radius(g) if nx.is_connected(g) else math.inf)
            diameter.append(nx.diameter(g) if nx.is_connected(g) else math.inf)
            is_eulerian.append(nx.is_eulerian(g))
            min_degree.append(min_d)
            max_degree.append(max_d)
            average_clustering_coefficient.append(nx.average_clustering(g))
            triangle_number.append(sum(nx.triangles(g).values()))
            
        # Creación de gráficos 
        plot_num: int = 12                                   # Número de gràfics a generar
        axs = [plt.subplots()[1] for _ in range(plot_num)]   # Generem els gràfics
        
        plots = {                                            # Dictionary with everything (Tengo que arreglar muchas cosas)
            "Order": axs[0].plot(layers, order),                                                        # 1
            "Size": axs[1].plot(layers, size),                                                          # 2
            "Is_connected": axs[2].plot(layers, is_connected),                                          # 3
            "Connected_components": axs[3].plot(layers, number_connected_components),                   # 4
            "Largest_component_diameter": axs[4].plot(layers, largest_component_diameter),              # 5
            "Radius": axs[5].plot(layers, radius),                                                      # 6
            "Diameter": axs[6].plot(layers, diameter),                                                  # 7
            "Is_eulerian": axs[7].plot(layers, is_eulerian),                                            # 8
            "Min_degree": axs[8].plot(layers, min_degree),                                              # 9 
            "Max_degree": axs[9].plot(layers, max_degree),                                              # 10
            "Average_Clustering_Coefficient": axs[10].plot(layers, average_clustering_coefficient),     # 11
            "Triangle_number": axs[11].plot(layers, triangle_number),                                   # 12
            #"K-core_graph": nx.k_core(g)                                                                # 13
        }
                
        return plots