# TFG
Repositori per recollir tot el codi y documentació sobre el Treball de Fi de Grau

Aquest treball consisteix en un estudi sobre els grafs geomètrics aleatoris, en concret els grafs geomètrics aleatoris multicapa. Aquests suposen un bon model per estudiar les xarxes socials, i volem veure les seves propietats modificant els paràmetres generadors d'aquest tipus de grafs.

## Guia d'ús

Per executar totes les proves disponibles, només cal que executeu l'script **execute.sh**. Dins d'aquest trobareu tots els tests llistats, així com els paràmetres d'entrada que necessita el programa.

Per saber quin experiment realitzar, adreceu-vos a la memòria del projecte i sel·leccioneu l'experiment que voleu executar, dels quals en disposem 6 experiments, 4 experiments principals i 2 experiments addicionals.

### Paràmetres d'entrada de l'script

1. Test -test: Defineix el test a realitzar. És de tipus *string* i pot prendre valors "1", "2", "3", "4", "degreeFreq" (estudi de les freqüències dels graus en un graf, emprat al test 1 de la memòria) o "radiusComparison" (estudi de la pendent del grau mínim/màxim segons el paràmetre generador *r*, emprat a les justificacions experimentals). També tenim un valor extra que s'anomena "default" que s'utilitza per fer petites proves al codi, ja que té una secció pròpia dins la funció main del programa principal _main.py_.
3. Ordre del graf -n: Defineix l'ordre del multicapa, així com totes les seves capes. Per valors molt elevats d'ordre (*n* > 3000) els experiments triguen força temps en executar-se. Per valors superiors a 5000 poden arribar a trigar dies.
4. Radi generador o radi inicial -r_ini: Pels experiments on la variable a analitzar és el radi, *r_ini* és el valor inicial de la progressió. Pels casos que sigui un altre paràmetre d'estudi, *r_ini* és el radi generador que es fa servir en tots els casos. Aquest valor ha de complir 0 <= *r_ini* < *x*, on *x* és la dimensió de la capsa contenidora del graf, en el nostre cas *x* = 1.
5. Radi final -r_fin: Pels experiments on la variable a analitzar és el radi, *r_fin* és el valor final de la progressió. Aquest valor compleix els mateixos requisits que *r_ini*.
6. Diferència entre els valors de progressiódel radi -r_add: Pels experiments on la variable a analitzar és el radi, *r_add* és el valor entre valors de la progressió, de manera que tindríem range(*r_ini*,*r_fin*,+*r_add*). El valor de *r_add* ha de complir que *r_add* < (*r_fin*-*r_ini*).
7. Número de capes del graf multicapa -num_graph: Quantitat de capes de les quals disposa el graf multicapa a generar. En cas que l'estudi es realitzi sobre el número de capes, tenim que la progressió és range(1,c+1), és a dir testem a partir d'una capa fins a *num_graphs* capes. Aquest valor compleix *num_graphs* > 0.
8. Número de rèpliques de l'experiment -num_copies: Per estandarditzar els resultats i trobar valors més propers als valors teòrics, realitzem diverses rèpliques del mateix experiment i fem la mitjana aritmètica dels valors resultants, degut al factor aleatori del què disposen els grafs geomètrics aleatoris. El número de rèpliques compleix que *num_copies* > 0.

Un exemple d'execució seria:

python3 main.py -test 1 -n 1000 -r_ini 0.1 -r_fin 0.2 -radius_add 0.005 -num_graph 20 -num_copies 5

On realitzem l'experiment número 1, amb *n* = 1000 nodes, radi generador *r* = 0.1, número de capes *c* = 20 i 5 rèpliques del mateix experiment.

Hem de tenir en compte la dimensió edls experiments que realitzem. Si donem valors molt elevats als paràmetres d'entrada, tindrem que el programa trigarà massa en executar, o potser esgota els recursos disponibles de la màquina, si la mostra a analitzar és excessivament gran.
