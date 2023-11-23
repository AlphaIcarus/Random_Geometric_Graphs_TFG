import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Script para generar los grafos que enseñamos en el TFG

# FIG 1: Grafo simple de 6 nodos
def fig1() -> None:
    adj = {
        1:[2,5],
        2:[1,3,5],
        3:[2,4],
        4:[3,5,6],
        5:[1,2,4],
        6:[4]
    }
    g = nx.Graph(adj)
    nx.draw_networkx(g)
    return


# FIG 2: Multigraf
def fig2() -> None:
    adj = {
        1:[1,2,2],
        2:[2,3,3],
        3:[3,4,4],
        4:[4,5,5],
        5:[5,6,6],
        6:[6,1,1]
    }
    g = nx.MultiGraph(adj)
    nx.draw_networkx(g)
    return

# Matplotlib Show

fig2()
plt.show()