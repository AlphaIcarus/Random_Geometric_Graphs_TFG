import networkx as nx
import random
import scipy

class Graph:

    def __init__(self, id: int, n: int, r: float, x: int):

        self.id = id                                            # Identifier of the graph
        self.n = n                                              # Number of nodes of the graph
        self.x = x                                              # Longitude/Amplitude of the map
        self.r = r                                              # Radius of area to get adjacencies       

        pos = {i: (random.uniform(0, x), random.uniform(0, x)) for i in range(n)}
        self.graph = nx.random_geometric_graph(n=n, radius=r, pos=pos)   # Graph itself 
    
        return
    
    # Basic data
    
    def getNumberOfNodes(self):
        return self.graph.number_of_nodes()
    
    def getNumberOfEdges(self):
        return self.graph.number_of_edges()
    
    # Connected components 
    
    def getNumberConnectedComponents(self):
        return nx.number_connected_components(self.graph)
    
    def getConnectedComponents(self):
        return nx.connected_components(self.graph)
    


class UnionGraph(Graph):

    def __init__(self, graphs: list[Graph]):

        self.graphList: list[Graph] = graphs

        self.graph: Graph = graphs[0].graph                     # Graph itself
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





