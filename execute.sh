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
  python3 main.py -test 001 -n 1000 -r_ini 0.1 -r_fin 0.1 -radius_add 0.05 -num_graph 20 

# Test 2: Radius evolution, vemos la evolución del multicapa dados dos radios en un intervalo, con secuencias de radius_add
  python3 main.py -test 010 -n 1000 -r_ini 0.1 -r_fin 0.2 -radius_add 0.005 -num_graph 20

# Test 3: Multilayer evolution, vemos gráficamente el multicapa añadiendo poco a poco las capas, en intervalos
  python3 main.py -test 100 -n 1000 -r_ini 0.1 -r_fin 0.2 -radius_add 0.005 -num_graph 20



# OBERSVERACIONES FINALES

# El parámetro x se queda igual SIEMPRE

# TESTS 1 FORMAL

# Radio: ver valores de 0.1 a 0.5 de radio (en valores de x=1.0) con r_add = 0.1, esto para valores de n = {1000,2000,3000,4000,5000}.
    # Si vemos comportamientos raros hacemos focus en el intervalo raro y hacemos el estudio con r_add menor
    # Hacer el estudio en las zonas que sean interesantes
# Si todo está bien vemos los valores de las propiedades --> Documentamos en la memoria
    # Vemos la progresión dependiendo del valor de n, si vemos que es progresivo no seguimos aumentando el valor de n, pero si hay cambios seguimos el test con valores más altos

# Ver dos bloques: N pequeña y N grande

# Empezar a escribir los resultados de los experimentos y lo que encontremos (todo lo que se me ocurra lo dejo apuntado) --> Qué se observa? (IMPORTANTE)
