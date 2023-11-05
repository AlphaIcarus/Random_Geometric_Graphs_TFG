import networkx as nx
import random
import pandas as pd
import math
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import numpy as np

# Required by networkx.random_geometric_graph
import scipy

# Things to do:
# TODO añadir documentación detallada sobre los parámetros y funciones

class Graph:

    def __init__(self, id: int, n: int, r: float, x: float):
        """
        Constructora de la classe Graph
        """
        self.id = id                                            # Identifier of the graph
        self.n = n                                              # Number of nodes of the graph
        self.x = x                                              # Longitude/Amplitude of the map
        self.r = r                                              # Radius of area to get adjacencies       

        pos = {i: (random.uniform(0, x), random.uniform(0, x)) for i in range(n)}
        self.graph = nx.random_geometric_graph(n=n, radius=r, pos=pos)   # Graph itself 
    
        return
    
    # Getters
    
    def getKCore(self) -> nx.graph:
        """
        Mètode per obtenir el graf k-core. Aquest graf és igual que l'original, però cada note té grau k (amb k màxima).
        """
        return nx.k_core(self.graph)
    
    def getInfo(self) -> pd.DataFrame:
        """
        TODO doc
        TODO Puede ser interesante guardar la informacón como atributo en la propia clase
        TODO Rellenar el dataframme con toda la info de interés
        """
        
        # Degrees
        degrees = [val for (_, val) in self.graph.degree()]
        min_degree = min(degrees)
        max_degree = max(degrees)
        
        # Connected components
        connected_components = sorted(nx.connected_components(self.graph), key=len, reverse=True)   #Sorted from largest to smallest
        cc_sizes = map(len, connected_components)                                                   #Sorted from largest to smallest
        
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
            "Triangle_number":None, # Esto devuelve un mapping, hay que hacerlo de alguna otra forma
            # Tamaños de componentes connexas
            # K-core (grafo donde todos los nodos tienen grado k, k máxima) --> se cosigue con k_core de NetworkX
                # Lo haremos en una función diferente y lo invocamos fuera
        }
        frame = pd.DataFrame(index=[0], data=dct)
        return frame

    def printInfo(self) -> None:
        """
        Printa per terminal la informació relacionada amb el graf
        """
        frame = self.getInfo()
        print(frame)
        
        return
    
    # Data save / load
    
    def saveXML(self, fileName: str | None, mode: bool) -> None:
        """
        Mètode per guardar a memòria, o imprimir per terminal, les dades de l'objecte.
        """       
        nodes = [n for n in self.graph.nodes]                       # [ID]
        edges = [e for e in self.graph.edges]                       # [(nodeID, nodeID)]
        nodesPos = nx.get_node_attributes(self.graph, "pos")        # [(coordX, coordY)]
        
        order = self.graph.order()
        size = self.graph.size()
        
        # TODO meter los datos en el Element Tree
        gr = et.Element('graph')
        id = et.SubElement(gr, 'id', attrib={"value":str(self.id)})
        n = et.SubElement(gr, 'n', attrib={"value":str(self.n)})
        r = et.SubElement(gr, 'r', attrib={"value":str(self.r)})
        x = et.SubElement(gr, 'x', attrib={"value":str(self.x)})

        # Node info
        nodeInfo = [et.SubElement(gr, 'node') for i in range(order)]
        nodeIds  = [et.SubElement(nodeInfo[i], 'nodeId', attrib={"value":str(nodes[i])}) for i in range(order)]
        nodeXCoord = [et.SubElement(nodeInfo[i], 'xCoord', attrib={"value":str(nodesPos[i][0])}) for i in range(order)]
        nodeYCoord = [et.SubElement(nodeInfo[i], 'yCoord', attrib={"value":str(nodesPos[i][1])}) for i in range(order)]
        
        # Adjacency info
        adjacencyInfo = [et.SubElement(gr, 'adjacency') for i in range(size)]
        adjacencyUvertex = [et.SubElement(adjacencyInfo[i], 'uVertex', attrib={"value":str(edges[i][0])}) for i in range(size)]
        adjacencyVvertex = [et.SubElement(adjacencyInfo[i], 'vVertex', attrib={"value":str(edges[i][1])}) for i in range(size)]
        
        # Printing
        from xml.dom import minidom
        xmlstr = minidom.parseString(et.tostring(gr)).toprettyxml(indent="   ")
        
        if mode:
            with open("./samples/" + fileName + ".xml", "w") as f:
                f.flush()
                f.write(xmlstr)
                f.close()
        else:
            print(xmlstr)
    
        return
    
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

    def __init__(self, graphs: list[Graph]):
        """
        Constructora de la classe. Construeix, a partir d'un conjunt de grafs aleatoris multicapa,
        un graf aleatori geomètric multicapa.
        
        TODO Cambiar que el multicapa se haga en otro método a parte, no en la constructora
        """
        self.graphList: list[Graph] = graphs

        self.graph = graphs[0].graph.copy()                     # Graph itself
        self.id = graphs[0].id                                  # Identifier of the graph
        self.n = graphs[0].n                                    # Number of nodes of the graph
        self.x = graphs[0].x                                    # Longitude/Amplitude of the map
        self.r = graphs[0].r                                    # Radius of area to get adjacencies

        adjacencies = set()
        for i in range(len(self.graphList)):
            adjacencies = adjacencies.union(set(self.graphList[i].graph.edges())) 

        self.graph.add_edges_from((adjacencies))
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
    
    def seeProgression(self, rang: int) -> None:
        """
        Mètode per obtenir imatges de com evoluciona el graf multicapa en diferents estats.
        També obtenim un dataframe amb la informació dels grafs intermitjos.
        
        - TODO actualmente se generan uno a uno, hay que obtener todas las imágenes de grafo y luego imprimirlas
        """
        
        inter = rang
        self.graph = self.graphList[0].graph.copy()
        
        self.drawRandomGeometricGraph()     # Estat inicial 
        
        for graph in self.graphList:
            # Add new edges
            self.graph.add_edges_from(set(graph.graph.edges()))
            inter -= 1
            if inter == 0:
                # Get dataframe
                
                #Print graph
                self.drawRandomGeometricGraph()
                inter = rang
        
        return
    
    def radiusProgression(self, r_ini: float, r_fin: float, r_add: float):
        """
        Mètode que, donat un radi inicial, un radi final i un valor, on r_ini + r_add <= r_fin, Va imprimint el graf multicapa fent servir
        els valors intermitjos de la progressió [r_ini, r_ini + r_add, r_ini + r_add*2, ... , r_ini + r_add*N <= r_fin] per N màxima.
        
        - TODO no funciona encara
        """
        
        # Còpies dels valors originals per restaurar
        graphs = self.graphList.copy()
        ml = self.graph.copy()
        plots = []
                
        all_nodes = map(nx.get_node_attributes, self.graphList) # We get the nodes of every graph
        for radius in np.arange(r_ini,r_fin,r_add):
            """Aconseguim els grafs nous donats pel radi radius, construim el multicapa nou i el dibuixem"""
            self.graphList = [nx.random_geometric_graph(n=self.n, pos=all_nodes[i]["coords"], radius=radius) for i in range(0,len(self.graphList))]
            self.buildMultilayer()
            plots.append(self.drawRandomGeometricGraph())

        # Recuperem les dades anteriors
        self.graphList = graphs
        self.graph = ml
        
        return plots
    
    def getGraphics(self) -> dict:
        """
        Mètode per obtenir els gràfics de com varien els atributs del graf multicapa, de manera progressiva.
        
        Els gràfics que s'obtenen venen donats per les propietats que volem obtenir del graf (donat a getInfo)
        
        - TODO: Los gráficos que no funcionan son 6 (radius), 7 (diameter)
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
        """
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [1, 2, 0, 0.5])
        plt.show()
        """
        
        axs = [plt.subplots()[1] for _ in range(12)]   # Generamos los plots necesarios (plot por atributo)
        
        # Pruebas
        # axs[0].plot(layers, order)
        
        # Dictionary with everything (Tengo que arreglar muchas cosas)
        plots = {
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
            "K-core_graph": nx.k_core(g)                                                                # 13
        }
                
        return plots
    
    def printInfo(self) -> None:
        """
        Imprimeix per pantalla el les dades del graf unió, així com les dades de cadascun dels grafs dels quals parteix.
        """
        df = self.getInfo()
        print(df)
        return
    
    # Data save / load
    
    def saveXML(self, fileName: str | None, mode: bool) -> None:
        """
        Mètode per guardar a memòria les dades de l'objecte.
        """
        Graph.saveXML(self, fileName, mode)
        for i in range(len(self.graphList)):
            Graph.saveXML(self.graphList[i], fileName + f'_Graph{i}', mode)
        
        return