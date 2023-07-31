"""@package Config

Paquet que conté la configuració de paràmetres per l'execució de l'script principal
 
TODO Poner documentción de esta clase
"""

from argparse import Namespace

class Config:
    """
    Classe que reuneix els atributs per executar l'script principal
    """

    def __init__(self, args: Namespace | None):
        """Constructora per la classe de configuració.
        
        Defineix els paràmetres necessaris per executar l'script.
        TODO aplicar restricciones en los parámetros (por ejemplo, 0 <= r <= self.x)
        """
        self.n: int
        """Paràmetre n: defineix l'ordre del graf"""
        self.x: float
        """Paràmetre x: defineix l'àrea del quadrat on es projecta el graf"""
        self.r: float
        """Paràmetre r: defineix el radi a partir del qual s'extableixen les adjacències"""
        self.num_graph: int
        """Paràmetre num_graph: defineix el número de graphs a generar pel graf unió"""
        
        # Value insertion
        self.n = args.n
        self.x = args.x
        self.r = args.r
        self.num_graph = args.num_graph

        return
