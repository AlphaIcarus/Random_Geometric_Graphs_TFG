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

# Test 1: 
python3 main.py -test 000 -n 1000 -x 2.0 -r_ini 0.1 -r_fin 0.1 -radius_add 0.05 -num_graph 50  # Empezamos con dimensión 2 (x = 2.0)

# Test 2: Multilayer evolution, vemos la progresión del multicapa añadiendo poco a poco las capas
# python3 main.py -test 100 -n 1000 -x 2.0 -r_ini 0.01 -r_fin 0.1 -radius_add 0.05 -num_graph 50  # Empezamos con dimensión 2 (x = 2.0)