"""@package Config

Paquet que conté la configuració de paràmetres per l'execució de l'script principal
 
TODO Poner documentción de esta clase
"""

# Packages
from argparse import Namespace
from math import sqrt

# Global parameters / literals / const values 
MINIMUM_GRAPH_ORDER = 0
MAXIMUM_GRAPH_ORDER = 10000
MINIMUM_RADIUS      = 0.0
MINIMUM_NUM_GRAPH   = 1
MAXIMUM_NUM_GRAPH   = 5000

# Class definition
class Config:
    """
    Classe que reuneix els atributs per executar l'script principal
    """

    def __init__(self, args: Namespace | None):
        """Constructora per la classe de configuració.
        
        Defineix els paràmetres necessaris per executar l'script.
        TODO aplicar mejores restricciones para los parámetros
        """
        self.test: str
        """Paràmetre test: defineix el tipus de test que executem. """
        self.n: int
        """Paràmetre n: defineix l'ordre del graf"""
        self.x: float
        """Paràmetre x: defineix l'àrea del quadrat on es projecta el graf"""
        self.r_ini: float
        """
        Paràmetre r_ini: defineix el radi a partir del qual s'extableixen les adjacències.\n
        Definim que el radi màxim és la llargada de la diagonal de l'àrea x*x
        """
        self.r_fin: float
        """
        Paràmetre r_fin: defineix el radi màzim dels tests.\n
        Definim que el radi màxim és la llargada de la diagonal de l'àrea x*x
        """
        self.radius_add: float
        """
        Paràmetre radius_add: defineix els intervals de radi que fa servir els tests per executar les proves.
        """
        self.num_graph: int
        """Paràmetre num_graph: defineix el número de graphs a generar pel graf unió"""
        
        # Value insertion
        self.test = args.test
        self.n = args.n
        self.x = args.x
        self.r_ini = args.r_ini
        self.r_fin = args.r_fin
        self.radius_add = args.radius_add
        self.num_graph = args.num_graph
        
        # Value checks
        try:
            assert(MINIMUM_GRAPH_ORDER <= self.n <= MAXIMUM_GRAPH_ORDER)
        except(AssertionError):
            print(f"El nombre de nodes no es troba en el rang de valors permesos:[{MINIMUM_GRAPH_ORDER},{MAXIMUM_GRAPH_ORDER}]")
            raise ValueError
        try:
            assert(self.x > 0.0)
        except(AssertionError):
            print(f"El valor de x ha de ser positiu major que 0.0")
            raise ValueError
        try:
            assert(MINIMUM_RADIUS <= self.r <= sqrt(self.x ** self.x + self.x ** self.x))
        except(AssertionError):
            print(f"El radi no es troba en el rang de valors permesos:[{MINIMUM_RADIUS},{sqrt(self.x ** self.x + self.x ** self.x)}]")
            raise ValueError
        try:
            assert(MINIMUM_NUM_GRAPH <= self.num_graph <= MAXIMUM_NUM_GRAPH)
        except(AssertionError):
            print(f"El nombre de grafs a generar no es troba en el rang de valors permesos:[{MINIMUM_NUM_GRAPH},{MAXIMUM_NUM_GRAPH}]")
            raise ValueError
        
        # Everything went OK
        return
