"""@package docstring
Documentation for this module.
 
More details.
"""

class Config:
    """
    Classe que reuneix els atributs per executar l'script principal
    """

    def __init__(self):
        """Constructora per la classe de configuració.
        
        Defineix els paràmetres necessaris per executar l'script.
        """
        self.n = 100
        """Paràmetre n: defineix l'ordre del graf"""
        self.x = 1
        """Paràmetre x: defineix l'àrea del quadrat on es projecta el graf"""
        self.r = 0.1
        """Paràmetre r: defineix el radi a partir del qual s'extableixen les adjacències"""
        self.num_graph = 50
        """Paràmetre num_graph: defineix el número de graphs a generar pel graf unió"""
