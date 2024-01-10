import networkx as nx
import random
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy    # Required by networkx.random_geometric_graph
from datetime import datetime
# Things to do:
# TODO añadir documentación detallada sobre los parámetros y funciones

# Utils

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
        Mètode per obtenir els paràmetres d'estudi dels experiments.
        """
        # Degrees
        degrees = [val for (_, val) in self.graph.degree()]
        min_degree = min(degrees)
        max_degree = max(degrees)
        
        # Connected components
        connected_components = sorted(nx.connected_components(self.graph), key=len, reverse=True)   #Sorted from largest to smallest
        # cc_sizes = map(len, connected_components)                                                   #Tamanys de les components connexes
        
        largest_component_nodes = max(nx.connected_components(self.graph), key=len)
        largest_component = self.graph.subgraph(largest_component_nodes)
        largest_c_diameter = nx.diameter(largest_component) if nx.is_connected(largest_component) else math.inf
                
        dct = {
            "Order":self.graph.order(),
            "Size":self.graph.size(),
            "Is_connected":1 if nx.is_connected(self.graph) else 0,
            "Connected_components":nx.number_connected_components(self.graph),
            "Largest_component_diameter":largest_c_diameter,
            "Radius":nx.radius(self.graph) if nx.is_connected(self.graph) else math.inf,
            "Diameter":nx.diameter(self.graph) if nx.is_connected(self.graph) else math.inf,
            "Is_eulerian":1 if nx.is_eulerian(self.graph) else 0,
            "Min_degree":min_degree,
            "Max_degree":max_degree,
            "Average_Clustering_Coefficient":nx.average_clustering(self.graph),
            "Triangle_number": nx.triangles(self.graph, 0),
            "K_value": max(nx.core_number(self.graph).values()),
            "K_core_order": nx.k_core(self.graph).order()
        }
        frame = pd.DataFrame(index=[index], data=dct)
        return frame
    
    def getDegreeFrequency(self) -> dict:
        """
        Mètode per obtenir la freqüència de cada grau al graf, és a dir el número de vegades que un grau apareix al graf

        Returns:
            dict: Diccionari que contè, per cada valor de grau al graf, el número de vegades que aquest apareix. 
        """
        degrees = [d for (_,d) in list(self.graph.degree)]
        degreeCounting = {d:degrees.count(d) for d in degrees}
        return dict(sorted(degreeCounting.items()))
   
    # Graph printing
    
    def drawRandomGeometricGraph(self) -> plt.Figure:
        """
        Guarda a la carpeta de tests el graf passat per paràmetre, en forma gràfica (format png)
        
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
        
        fig = plt.gcf()
        tdir = '/' if os.name == 'posix' else '\\'
        name = "Random_geometric_graph"
        now = str(datetime.now()).replace(":",".")
        
        dir = f".{tdir}test_output{tdir}" + now
    
        try:
            os.mkdir(dir)
        except(FileExistsError):
            pass
        
        fig.savefig(dir + tdir + name + ".png") # Save figure
        plt.close()     # Figure closing due to overload
        return

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
        
        df = [Graph.getInfo(self)]                              # Data frame
        
        for graph in self.graphList[1:]:
            # Add new edges
            self.graph.add_edges_from(set(graph.graph.edges()))
            inter -= 1
            if inter == 0:
                df.append(Graph.getInfo(self))                  # Get dataframe
                inter = rang
        
        df = pd.concat(df)
        return df
    
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
        
        for radius in np.arange(r_ini,r_fin,r_add):
            """Aconseguim els grafs nous donats pel radi radius, construim el multicapa nou i el dibuixem"""
            self.graphList = [Graph(i,graphs[0].n,radius,graphs[0].x, pos[i]) for i in range(len(self.graphList))]
            self.buildMultilayer()
            df.append(Graph.getInfo(self))
            self.emptyMultilayer()
            
        df = pd.concat(df)                         # Construim el dataframe
        # Recuperem les dades anteriors
        self.graphList = graphs
        self.buildMultilayer()
        
        return df
    
    def getParameterProgression(self) -> pd.DataFrame:
        """
        Mètode per obtenir els gràfics de com varien els atributs del graf multicapa, de manera progressiva.
        
        Els gràfics que s'obtenen venen donats per les propietats que volem obtenir del graf (donat a getInfo)
        
        - TODO: Cambiar el funcionamiento para usar emptyMultilayer y buildMultilayer, y devolver el dataset con la info por capas
        """
        #New mode
        self.emptyMultilayer()
        self.graph = self.graphList[0].graph
        df = []
        
        for graph in self.graphList:
            self.graph.add_edges_from((set(graph.graph.edges())))
            df.append(Graph.getInfo(self))
        
        df = pd.concat(df)
        self.buildMultilayer()     
        return df