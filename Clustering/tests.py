import pandas as pd

file = pd.read_csv('Clustering/Datasets/primeras_ultimas.csv')


groups = file.groupby('cyclenumber')

for value in groups.groups: # Obtención de las keys de los grupos
    print(float((groups.get_group(value).iloc[0])['longitude'])) # Acceso a una fila de un grupo

