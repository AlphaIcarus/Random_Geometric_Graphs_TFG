# Script bash per executar diferents proves amb l'script principal

# Autoconfiguration
# Uncomment to upgrade packages

# !pip install matplotlib
# !pip install networkx
# !pip install argparse
# !pip install pandas
# !pip install random
# !pip install math
# !pip install xml
# !pip install scipy

# Global parameters

# Main script

# Test 1: Parameter evolution, vemos la evolución de los parámetros a estudiar del multicapa 
# python3 main.py -test 001 -n 1000 -x 2.0 -r_ini 0.1 -r_fin 0.1 -radius_add 0.05 -num_graph 50  # Empezamos con dimensión 2 (x = 2.0)

# Test 2: Radius evolution, vemos la evolución del multicapa dados dos radios en un intervalo, con secuencias de radius_add
  python3 main.py -test 010 -n 1000 -x 2.0 -r_ini 0.01 -r_fin 0.1 -radius_add 0.05 -num_graph 50  # Empezamos con dimensión 2 (x = 2.0)

# Test 3: Multilayer evolution, vemos gráficamente el multicapa añadiendo poco a poco las capas, en intervalos
# python3 main.py -test 100 -n 1000 -x 2.0 -r_ini 0.01 -r_fin 0.1 -radius_add 0.05 -num_graph 50  # Empezamos con dimensión 2 (x = 2.0)