# Script bash per executar diferents proves amb l'script principal

# Autoconfiguration
# Uncomment to upgrade packages

# !pip install matplotlib
# !pip install networkx
# !pip install argparse
# !pip install pandas
# !pip install random
# !pip install math
# !pip install scipy

# Main script

# Test 1: Evolució del multicapa segons número de nodes n
  python3 main.py -test 1 -n 1000 -r_ini 0.1 -r_fin 0.2 -radius_add 0.005 -num_graph 20 -num_copies 5

# Test 2: Evolució del multicapa segons radi r
  python3 main.py -test 2 -n 1000 -r_ini 0.0 -r_fin 0.1 -radius_add 0.005 -num_graph 20 -num_copies 5

# Test 3: Evolució del multicapa segons radi r, tenint en compte diferents valors de n
  python3 main.py -test 3 -n 1000 -r_ini 0.0 -r_fin 0.1 -radius_add 0.005 -num_graph 20 -num_copies 5

# Test 4: Evolució del multicapa segons número de capes c, tenint en compte diferents valors de n
  python3 main.py -test 4 -n 1000 -r_ini 0.1 -r_fin 0.2 -radius_add 0.005 -num_graph 20 -num_copies 5

# Test 5: Evolució del multicapa segons radi r, tenint en compte diferents valors de n
  python3 main.py -test 3 -n 1000 -r_ini 0.0 -r_fin 0.02 -radius_add 0.0005 -num_graph 20 -num_copies 5
