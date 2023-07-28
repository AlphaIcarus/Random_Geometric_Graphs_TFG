import networkx as nx
import random
import pandas as pd
import math
import os
import xml.etree.ElementTree as et

class Graph:

    def __init__(self, id: int, n: int, r: float, x: float):
        """Constructora de la classe Graph"""
        self.id = id                                            # Identifier of the graph
        self.n = n                                              # Number of nodes of the graph
        self.x = x                                              # Longitude/Amplitude of the map
        self.r = r                                              # Radius of area to get adjacencies       

        pos = {i: (random.uniform(0, x), random.uniform(0, x)) for i in range(n)}
        self.graph = nx.random_geometric_graph(n=n, radius=r, pos=pos)   # Graph itself 
    
        return
    
    # Basic data
    
    # Tabular information
    
    def getInfo(self):
        """
        TODO doc
        TODO Puede ser interesante guardar la informacón como atributo en la propia clase
        TODO Rellenar el dataframme con toda la info de interés
        """
        dct = {
            "Order":self.graph.order(),
            "Size":self.graph.size(),
            "Radius":nx.radius(self.graph) if nx.is_connected(self.graph) else math.inf,
            "Diameter":nx.diameter(self.graph) if nx.is_connected(self.graph) else math.inf
        }
        frame = pd.DataFrame(index=[0], data=dct)
        
        return frame

    
    def printInfo(self):
        """
        """
        frame = self.getInfo()
        print(frame)
        
        return
    
    # Data save / load
    
    def saveXML(self, fileName: str):
        """
        Mètode per guardar a memòria les dades de l'objecte.
        """
        tree = et.ElementTree()
        
        # TODO meter los datos en el Element Tree
        
        tree.write("./samples/" + fileName + ".xml")
        
        return
    
    def loadXML(self, fileName: str):
        """
        Mètode per carregar de memòria les dades de l'objecte.
        
        TODO Mirar si hay alguna manera de implementar esto como constructora
        """
        tree = et.parse("./samples/" + fileName + ".xml")
        
        # TODO obtener todos los datos del XML y meterlos en Graph
        
        return


class UnionGraph(Graph):

    def __init__(self, graphs: list[Graph]):

        self.graphList: list[Graph] = graphs

        self.graph: nx.Graph = graphs[0].graph                     # Graph itself
        self.id = graphs[0].id                                  # Identifier of the graph
        self.n = graphs[0].n                                    # Number of nodes of the graph
        self.x = graphs[0].x                                    # Longitude/Amplitude of the map
        self.r = graphs[0].r                                    # Radius of area to get adjacencies

        adjacencies = set()
        for i in range(0,len(graphs)):
            s = set(graphs[i].graph.edges())
            adjacencies.union(s)

        self.graph.add_edges_from(list(adjacencies))        # Alomejor no hace falta ni pasarlo a list

        return
    
    # Tabular information
    
    def getInfo(self):
        """
        Mètode per obtenir totes les dades d'interés del graf unió.
        
        Primer obtenim el dataframe del graf unió, posteriorment obtenim el de la seva col·lecció.
        
        TODO doc
        TODO Puede ser interesante guardar la informacón como atributo en la propia clase
        TODO Rellenar el dataframme con toda la info de interés
        """
        dfUnion = Graph.getInfo(self)
        size = len(self.graphList)
        dct = {
            "Order":[self.graphList[i].graph.order() for i in range(0,size)],
            "Size":[self.graphList[i].graph.size() for i in range(0,size)],
            "Radius":[nx.radius(self.graphList[i].graph) if nx.is_connected(self.graphList[i].graph) else math.inf for i in range(0,size)],           
            "Diameter":[nx.diameter(self.graphList[i].graph) if nx.is_connected(self.graphList[i].graph) else math.inf for i in range(0,size)]
        }
        dfCollect = pd.DataFrame(data = dct)
        
        return (dfUnion, dfCollect)
    
    def printInfo(self):
        """
        Imprimeix per pantalla el les dades del graf unió, així com les dades de cadascun dels grafs dels quals parteix.
        """
        dfUnion, dfCollect = self.getInfo()
        
        print(dfUnion)
        print(dfCollect)
        return
    
    # Data save / load
    
    def saveXML(self, fileName: str):
        """
        Mètode per guardar a memòria les dades de l'objecte.
        """
        tree = et.ElementTree()
        
        # TODO meter los datos en el Element Tree
        
        tree.write("./samples/" + fileName + ".xml")
        
        return
    
    def loadXML(self, fileName: str):
        """
        Mètode per carregar de memòria les dades de l'objecte.
        
        TODO Mirar si hay alguna manera de implementar esto como constructora
        """
        tree = et.parse("./samples/" + fileName + ".xml")
        
        # TODO obtener todos los datos del XML y meterlos en Graph
        
        return
    

""" Things to do """

# TODO añadir documentación detallada sobre los parámetros y funciones

""" Script para testing """

"""
U = UnionGraph([Graph(i, 100, 0.2, 1.0) for i in range(0,4)])
U.printInfo()
"""