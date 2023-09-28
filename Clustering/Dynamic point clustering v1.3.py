import datetime
from datetime import datetime
import tempfile
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import random
import pandas as pd
from tqdm import tqdm
import math
import PySimpleGUI as sg
from os import getcwd
from sklearn import metrics
matplotlib.use("TKAgg")

# Global vars
global_time = None # Tiempo de la aplicación para ver cuando reclusterizar
x = []  # Lista de valores x
y = []  # Lista de valores y
old_x = []  # Lista de valores x
old_y = []  # Lista de valores y
old_x_list = []  # Lista de valores x
old_y_list = []  # Lista de valores y
groups = None # Guarda las rutas que ha realizado cada bicicleta
reasigned_points = []  # Puntos que se reasignaron desde la ultima iteración
# Puntos que se movieron desde que se realizó la última clusterización (Elimina repetidos)
accumulated_moved_points = []
post_reasigned_points = []  # Coordenadas de los puntos q se reasignaron
# Puntos que se reasignaron desde la ultima iteración
last_n_post_reasigned_points_configurations = []
# Puntos que se reasignaron desde la ultima iteración
last_n_reasigned_points_configurations = []
# Guarda las n ultimas configuraciones de clusteres realizadas
last_n_optimal_clusters_configurations = []
# Guarda los centroides de la ejecución anterior
last_n_optimal_centroids_configuration = []
# Guarda las etiquetas de la última clusterización
last_reclusterization_labels = []
# Puntos reasignados desde la ultima iteración
reclustered_points_since_last_iteration = []
input_offset = 3
input_values = []
y_axis_limits = ()
x_axis_limits = ()
plot_columns = 4
plot_rows = 2
max_clusters_to_calculate = 20
# Numero de iteraciones del bucle (simula instantes de tiempo de un día)
max_iterations = 8
point_movement_probability = 0.1  # Probabilidad de que se mueva un punto
# Cantidad de movimiento de un punto respecto a la distancia entre los puntos más lejanos de cada eje.
point_movement_quantity = 0.1
# Porcentaje en valor de 0 a 1 de que cuantos puntos que se tiene que mover para que se recalculen los clústeres
reassigment_coefficient_threshold = 0.1
# Multiplicador de la variación de movimiento
delta_m = 0.1
# Multiplicador de la variación puntos de cluster
delta_c = 0.9
# Coeficiente de reasignación
reassigment_coefficient = 0
#####


plot_index = [0, 0]

# Confifuración de la IU

point_size = 5

sg.theme("Reddit")

frame_layout = [[sg.Multiline("", size=(72, 15), autoscroll=True,
                              reroute_stdout=True, reroute_stderr=False, key='-OUTPUT-')]]

dataset_load_frame = [
    [sg.InputText(key="-FILE-", size=46),
     sg.FileBrowse(file_types=(("CSV", "*.csv"),)),
     sg.Button('Cargar dataset')]
]

dataset_generator_frame = [
    [sg.Text("Número de puntos a generar:"), sg.Input(
        key="-POINTSTOGENERATE-", size=4, default_text=20)],
    [sg.Button('Generar dataset', size=(27,1))]
]

graph_distribution_frame = [
    [sg.Text("Columnas:"), sg.Input(key="-COLUMNS-", size=2, default_text=4),
     sg.VerticalSeparator(),
     sg.Text("Filas"), sg.Input(
        key="-ROWS-", size=2, default_text=2),
     sg.VerticalSeparator(),
     sg.Text("Iteraciones:"), sg.Input(key="-ITERATIONS-", readonly=True, size=3, default_text="f * c")],
    [sg.Text("Tamaño de los puntos del gráfico:"), sg.Input(
        key="-POINTSIZE-", size=2, default_text=5),]
]

movement_variables_frame = [
    [sg.Text("Probabilidad de movimiento (Decimales [0.0, 1.0]:"),
     sg.Input(key="-MOVPROB-", size=4, default_text=0.1),],
    [sg.Text("Cantidad de movimiento (Decimales [0.0, 1.0]):"), sg.Input(
        key="-MOVQUANT-", size=4, default_text=0.1),]
]

reclusterization_variables_frame = [
    [sg.Text("Delta_m (Decimales [0.0, 1.0]):"),
     sg.Input(key="-DELTAM-", size=4,
              default_text=0.1), sg.VerticalSeparator(), sg.Text("Delta_c (Decimales [0.0, 1.0]):"),
     sg.Input(key="-DELTAC-", size=9, default_text="1-delta_m", readonly=True)],
    [sg.Text("Umbral del coeficiente de reasignación para reclusterizar (Decimales [0.0, 1.0]):"),
     sg.Input(key="-RCTHRESHOLD-", size=4, default_text=0.1),
     sg.Text("\n")],
    [sg.Text("Numero de agrupaciones entre las que calcular el óptimo (min. 3):"),
     sg.Input(key="-MAXPOINTCLUST-", size=4, default_text=10)]
]

layout = [[sg.Frame("SELECCION DE DATASET", dataset_load_frame, font="Any 12")],
          [sg.Frame("DISTRIBUCIÓN DEL GRÁFICO",
                    graph_distribution_frame, font="Any 12"), sg.Frame("GENERACIÓN DE DATASETS:", dataset_generator_frame, font="12")],
          [sg.Frame("VARIABLES DE MOVIMIENTO",
                    movement_variables_frame, font="Any 12")],
          [sg.Frame("VARIABlES DE RECLUSTERIZACIÓN",
                    reclusterization_variables_frame, font="Any 12")],
          [[[sg.Button('Distribución de puntos actual',), sg.Button(
              'Mostrar configuración de clústeres actual',), sg.Button('Ejecutar',),], sg.Frame("Salida de consola", frame_layout),],],]

window = sg.Window('Dynamic point clustering', layout,
                   resizable=True, element_justification="l")
####


def setParameters(values):
    global plot_columns
    global point_size
    global plot_rows
    global max_iterations
    global point_movement_probability
    global point_movement_quantity
    global max_clusters_to_calculate
    global reassigment_coefficient_threshold
    global delta_c
    global delta_m
    plot_columns = eval(values["-COLUMNS-"])
    plot_rows = eval(values["-ROWS-"])
    max_iterations = plot_columns * plot_rows
    point_movement_probability = eval(values["-MOVPROB-"])*100
    point_movement_quantity = eval(values["-MOVQUANT-"])
    max_clusters_to_calculate = eval(values["-MAXPOINTCLUST-"])
    point_size = eval(values["-POINTSIZE-"])
    reassigment_coefficient_threshold = eval(
        values["-RCTHRESHOLD-"])
    delta_m = eval(values["-DELTAM-"])
    delta_c = 1-delta_m


def mean(list):  # Función que calcula la media de una lista
    acc = 0
    for i in list:
        acc = acc + i
    return acc/len(list)


def initialize(k="Number of centroids", d="Data points"):
    centroids = []
    # Guarda la distancia de cada punto a cada centroide
    point_distance_to_centroids = [[]]

    # Escojo como primer centroide el valor más cercano a la media
    mean_x_coord = mean(d[0])
    mean_y_coord = mean(d[1])
    for point_index in range(len(d[0])):  # Calculo la distancia
        point_distance_to_centroids[0].append(abs(math.sqrt(
            (mean_x_coord - d[0][point_index]) ** 2) + ((mean_y_coord - d[1][point_index]) ** 2)))

    # Una vez calculadas todas las distancias escogemos el primer centro:
    point_position = point_distance_to_centroids[0].index(
        min(point_distance_to_centroids[0]))
    centroids.append((d[0][point_position], d[1]
                     [point_position]))

    point_distance_to_centroids[0].clear()  # Borro las distancias a la media

    # Calculo la distancia de todos los puntos al primer centroide
    for point_index in range(len(d[0])):
        point_distance_to_centroids[0].append(abs(math.sqrt(
            (centroids[0][0] - d[0][point_index]) ** 2) + ((centroids[0][1] - d[1][point_index]) ** 2)))

    # Captura el índice del centroide al que se asignará cada punto
    assigned_centroid = [0 for i in range(len(d[0]))]

    # Calculo el siguiente centroide, que va a ser el punto más lejano de de entre todos los centroides ya calculados anteriormente
    for n_centroid in range(1, k):
        # Contiene las distancias a los puntos más lejanos de cada centroide
        max_distant_point_distance = []
        furthest_distance = 0
        for centroid_distances in point_distance_to_centroids:
            max_distant_point_distance.append(max(centroid_distances))

        furthest_distance = max(max_distant_point_distance)
        centroid_with_furthest_point = max_distant_point_distance.index(
            furthest_distance)

        # Ahora capturo el punto más lejando del centroide
        furthest_point_index = point_distance_to_centroids[centroid_with_furthest_point].index(
            furthest_distance)
        # Ahora que ya tengo el punto más lejano del centroide, procedo a añadirlo como centroide nuevo

        centroids.append((d[0][furthest_point_index], d[1]
                         [furthest_point_index]))

        # Una vez añadido el centroide, recalculo las distancias y asignaciones de los puntos.
        point_distance_to_centroids.append([])
        # Calculo las distancias al nuevo centroide
        for point_index in range(len(assigned_centroid)):
            point_distance_to_centroids[n_centroid].append(abs(math.sqrt(
                (centroids[n_centroid][0] - d[0][point_index]) ** 2) + ((centroids[n_centroid][1] - d[1][point_index]) ** 2)))

        # Reasignamos los puntos que pertencen a un centroide, a otro, siempre y cuando la distancia mínima sea <= 0,
        # ya que si un centroide tiene una distancia de -1 unidades a un punto, entonces se considera que ese punto no tiene
        # ese centroide como el más cercano y por tanto, está asignado a otro centroide.
        for i in range(len(assigned_centroid)):
            centroids_max_distances = []
            for j in range(len(point_distance_to_centroids)):
                centroids_max_distances.append(
                    point_distance_to_centroids[j][i])
            assigned_centroid[i] = centroids_max_distances.index(min(filter(
                lambda x: x >= 0, centroids_max_distances)))  # Asignamos al punto el centroide más cercano

        # Una vez reasignados los puntos de clúster, reasigno las distancias asignando a los puntos no pertenecientes al centroide valor de distancia -1
        for centroid_index in range(len(point_distance_to_centroids)):
            i = 0
            for point_index in assigned_centroid:
                if (centroid_index != point_index):
                    point_distance_to_centroids[centroid_index][i] = -1
                i = i + 1

    clusters = []  # Reinicializo los clúster iniciales a 0
    old_centroids = []
    for i in range(k):  # Incializo los clústeres iniciales.
        old_centroids.append([[], [], []])
        clusters.append([[], [], []])

    return centroids, old_centroids, clusters


def kmeans(k="Number of centroids", d="Data points"):
    # Calculamos los centroides iniciales y creamos k clusters
    centroids, old_centroids, clusters = initialize(k, d)
    isFirstIteration = True

    # Mientras los centros no sean iguales a los antiguos centros, prosigo con el algoritmo
    while (centroids != old_centroids):
        labels = []  # Guarda el índice del clúster al que está asignado cada punto
        for i in range(k):
            # Borramos el cluster creado para crear uno nuevo
            clusters[i] = [[], [], []]

        for i in range(len(d[0])):
            distance_to_centroids = []
            # Calculo la distancia a cada uno de los centroides
            for centroid in centroids:
                distance_to_centroids.append(
                    abs(math.sqrt(pow(centroid[0] - d[0][i], 2) + (pow(centroid[1] - d[1][i], 2)))))

            # Cojo el índice del cluster que más cerca está de mi punto
            cluster_number = distance_to_centroids.index(
                min(distance_to_centroids))
            labels.append(cluster_number)
            # Asigno al cluster el punto
            clusters[cluster_number][0].append(d[0][i])
            clusters[cluster_number][1].append(d[1][i])

        # Recalculamos los centros calculando el punto más cercano a la media del cluster
        cluster_number = 0  # Usado como índice para recorrer los clusteres
        for ([coord_x, coord_y, point_c]) in clusters:
            # Calculo la media para recomputar el centroide
            mean_x = mean(coord_x)
            mean_y = mean(coord_y)
            # Guardo los centroides que tenía para comprobar más tarde en la condición de parada
            old_centroids[cluster_number] = centroids[cluster_number]
            # Reasigno los centroides
            centroids[cluster_number] = (mean_x, mean_y, -1)
            # Actualizo el índice para acceder al siguiente cluster en la siguiente interación
            cluster_number += 1

    return centroids, clusters, labels


def executeKmeans(max_clusters, reasigned_points):
    n_max_clusteres = max_clusters
    n_clusters_list = []
    clusters_CH_index = []
    centroids_list = []
    labels_list = []
    global old_x
    global old_y

    # Calculamos N clústeres para comprobar cual es la agrupación óptima.
    for n_clusters in range(2, n_max_clusteres):
        # Capturamos los clústeres y las etiquetas
        centroids, clusters, labels = kmeans(n_clusters, (x, y))

        # Calculamos el índice de Calinski-Harabasz utilizando el método proprocionado por la biblioteca scikit.learn.
        clusters_CH_index.append(
            metrics.calinski_harabasz_score(list(zip(x, y)), labels))
        n_clusters_list.append(clusters)
        centroids_list.append(centroids)
        labels_list.append(labels)

    # Escogemos la agrupación de clústeres con el mayor índice como el óptimo relativo
    i = clusters_CH_index.index(max(clusters_CH_index))
    clusters = n_clusters_list[i]
    # Guarda las etiquetas de a que cluster está asignado cada punto
    labels = labels_list[i]
    centroids = centroids_list[i]

    add_clusters_to_plot(clusters, list(zip(*centroids)),
                         reasigned_points, [], (old_x, old_y))
    return centroids, labels, clusters


def hasSignificantVariation(newPoints, centroids, labels, accumulated_moved_points):
    # reasigna los puntos a otro centroide si estos puntos han variado lo suficiente
    global reassigment_coefficient
    global last_reclusterization_labels
    reasigned_points = reasignPoints(centroids, labels, newPoints)

    for point in reasigned_points:
        if (point[0] in reclustered_points_since_last_iteration):
            if (last_reclusterization_labels[point[0]] == point[1]):
                reclustered_points_since_last_iteration.remove(point[0])
        else:
            reclustered_points_since_last_iteration.append(point[0])

    reassigment_coefficient = (1/len(labels)) * (
        (delta_m * len(accumulated_moved_points)) + (delta_c * len(reclustered_points_since_last_iteration)))
    # Si varían el reclustered_points_percentage_to_recalculate% de los puntos a la vez, se recalculan los clusteres
    if (reassigment_coefficient > reassigment_coefficient_threshold):
        return reasigned_points, True
    else:
        return reasigned_points, False

# Implementamos el movimiento aleatorio de los puntos. Este calcula la variación de forma independiente para cada coordenada
# de cada punto.


def movePoints(old_x, old_y, probability=15, movement=0.1):
    # Ajustamos la cantidad de movimiento a escala con el dataset
    global x
    global y
    max_distance_x = (max(old_x) - min(old_x)) * movement
    max_distance_y = (max(old_y) - min(old_y)) * movement
    moved_points = []
    for i in range(0, len(old_x)):
        if (probability > random.randrange(0, 100)):
            moved_distance_x = random.uniform(-max_distance_x, max_distance_x)
            moved_distance_y = random.uniform(-max_distance_y, max_distance_y)
            while (not (x_axis_limits[0] < moved_distance_x + old_x[i] < x_axis_limits[1])):
                moved_distance_x = random.uniform(
                    -max_distance_x, max_distance_x)
            while (not (y_axis_limits[0] < moved_distance_y + old_y[i] < y_axis_limits[1])):
                moved_distance_y = random.uniform(
                    -max_distance_y, max_distance_y)

            x.append(old_x[i] + moved_distance_x)
            y.append(old_y[i] + moved_distance_y)
            moved_points.append(i)
        else:
            x.append(old_x[i])
            y.append(old_y[i])

    return moved_points


# Comprueba si han cambiado puntos de un cluster a otro
def reasignPoints(centroids, labels, newPoints):
    # Guarda el indice de los puntos que se han cambiado de cluster y el cluster al que se asignan
    global reasigned_points
    reasigned_points = []
    for i in range(len(labels)):
        distance_to_centroids = []
        for centroid in centroids:
            distance_to_centroids.append(math.sqrt(pow(
                (centroid[0] - newPoints[0][i]), 2) + pow((centroid[1] - newPoints[1][i]), 2)))
        closest_centroid_index = distance_to_centroids.index(
            min(distance_to_centroids))
        if (closest_centroid_index != labels[i]):
            reasigned_points.append((i, closest_centroid_index))
    return reasigned_points


def add_clusters_to_plot(clusters, centroids, reasigned_points, post_reasigned_points, old_coords):
    global plot_index
    global point_size

    old_x = old_coords[0]
    old_y = old_coords[1]
    ax[plot_index[0]][plot_index[1]].set_ylim(min(y) - 1, max(y) + 1)
    ax[plot_index[0]][plot_index[1]].set_xlim(min(x) - 1, max(x) + 1)
    ax[plot_index[0]][plot_index[1]].set_title(
        "Clusters: " + str(len(clusters)) + " | Reasigned points: " + str(len(reasigned_points)))
    # Puntos que se reasignaron
    reasigned_points_coords = [[], []]
    if (old_x != [] and old_y != []):
        for point in reasigned_points:
            reasigned_points_coords[0].append(old_x[point[0]])
            reasigned_points_coords[1].append(old_y[point[0]])

    for cluster in clusters:
        ax[plot_index[0]][plot_index[1]].plot(
            cluster[0], cluster[1], 'o', markersize=point_size)
        ax[plot_index[0]][plot_index[1]].plot(
            centroids[0], centroids[1], 'X', markersize=point_size, color="red")

    if (len(post_reasigned_points) > 0):
        ax[plot_index[0]][plot_index[1]].plot(
            reasigned_points_coords[0], reasigned_points_coords[1], 'D', markersize=point_size-1, color="black")

    if (post_reasigned_points != [] and old_x != [] and old_y != []):
        for i in range(len(reasigned_points_coords[0])):
            points = [[], []]
            points[0].append(reasigned_points_coords[0][i])
            points[0].append(post_reasigned_points[0][i])
            points[1].append(reasigned_points_coords[1][i])
            points[1].append(post_reasigned_points[1][i])
            ax[plot_index[0]][plot_index[1]].plot(
                points[0], points[1], '-', markersize=point_size+2, color="black")

    plot_index[1] = plot_index[1] + 1
    if (plot_index[1] == plot_columns):
        plot_index[0] = plot_index[0] + 1
        plot_index[1] = 0


def reasign_and_show(reasigned_points, labels, centroids):
    reasigned_points_index = 0  # Indica por que punto de reasignación voy
    clusters = []
    post_reasigned_points = [[], []]
    global x
    global old_x
    global y
    global old_y
    for i in range(max(labels) + 1):
        clusters.append([[], [], []])

    # Copia los clusteres con los puntos reasignados
    for i in range(len(labels)):
        if (reasigned_points_index < len(reasigned_points) and i == reasigned_points[reasigned_points_index][0]):
            clusters[reasigned_points[reasigned_points_index]
                     [1]][0].append(x[i])
            clusters[reasigned_points[reasigned_points_index]
                     [1]][1].append(y[i])
            labels[i] = reasigned_points[reasigned_points_index][1]
            post_reasigned_points[0].append(x[i])
            post_reasigned_points[1].append(y[i])
            reasigned_points_index = reasigned_points_index + 1
        else:
            clusters[labels[i]][0].append(x[i])
            clusters[labels[i]][1].append(y[i])

    add_clusters_to_plot(clusters, list(zip(*centroids)),
                         reasigned_points, post_reasigned_points, (old_x, old_y))
    print()
    return labels, post_reasigned_points


def setDataset(dataset):
    global input_values
    global x
    x = []
    global y
    y = []
    global groups
    global global_time

    try:
        file = pd.read_csv(dataset)
    except Exception as e:
        print("\n", e)
        return False

    groups = file.groupby('id') # Agrupamos el dataset por número de bicicleta

    for value in groups.groups: # Recorre los grupos cogiendo la primera entrada de ellos para la
        if('timestamp' in  file.columns):
            route_start_timestamp = datetime.strptime((groups.get_group(value).iloc[0])['timestamp'], '%Y-%m-%d %H:%M:%S%z')
            if(global_time == None):
                global_time = route_start_timestamp

            elif(route_start_timestamp < global_time):
                global_time = route_start_timestamp
        
        x.append(float((groups.get_group(value).iloc[0])['longitude']))
        y.append(float((groups.get_group(value).iloc[0])['latitude']))

    global y_axis_limits
    y_axis_limits = (min(y), max(y))
    global x_axis_limits
    x_axis_limits = (min(x), max(x))

    return True  # El dataset se pudo cargar con éxito


def generateRandomDataset(number_of_points_to_be_generated=20):
    generated_random_points = [
        [random.uniform(-2, 2), random.uniform(-2, 2), 0] for x in range(number_of_points_to_be_generated)]

    point_id = 0
    with open("random_dataset.csv", "w") as file:
        file.write("id,longitude,latitude,class\n")
        for point in generated_random_points:
            file.write(f"{point_id},{point[0]},{point[1]}\n")
            point_id = point_id + 1
    print(f"Random dataset generated with {number_of_points_to_be_generated}. File path: {getcwd()}/random_dataset.csv")

    

def execute():
    global reasigned_points
    global accumulated_moved_points
    global post_reasigned_points
    # Indica cuándo se debería reclusterizar o reasignar.
    global reassigment_coefficient
    global last_reclusterization_labels
    optimal_centroids = []
    labels = []
    optimal_clusters = []
    global last_n_optimal_centroids_configuration
    global last_n_optimal_clusters_configurations
    global last_n_post_reasigned_points_configurations
    global last_n_reasigned_points_configurations
    global old_x
    global old_y
    global old_x_list
    global old_y_list
    global x
    global y
    last_n_optimal_centroids_configuration.clear()
    last_n_optimal_clusters_configurations.clear()
    last_n_reasigned_points_configurations.clear()
    last_n_post_reasigned_points_configurations.clear()
    old_x_list.clear()
    old_y_list.clear()
    # optimal_centroids Almacena los centroides de la configuración óptima relativa hallada
    # labels Almacena el índice de el clúster al que está asignado cada punto

    for i in tqdm(range(max_iterations-1)):
        if (i == 0):
            optimal_centroids, labels, optimal_clusters = executeKmeans(
                max_clusters_to_calculate, reasigned_points)
            last_n_optimal_clusters_configurations.append(optimal_clusters)
            last_n_optimal_centroids_configuration.append(optimal_centroids)
            last_n_reasigned_points_configurations.append(reasigned_points)
            last_n_post_reasigned_points_configurations.append(
                post_reasigned_points)
            old_x_list.append(old_x.copy())
            old_y_list.append(old_y.copy())
            last_reclusterization_labels = labels.copy()

        old_x = x.copy()
        old_x_list.append(x)
        old_y = y.copy()
        old_y_list.append(y)
        x.clear()
        y.clear()
        # Mueve los puntos con un valor de point_movement_probability% de probabilidad de èxito para cada punto
        moved_points = movePoints(old_x, old_y, point_movement_probability,
                                  point_movement_quantity)
        for point in moved_points:
            if point not in accumulated_moved_points:
                accumulated_moved_points.append(point)

        # Variated guarda el valor booleano acerca de si variaron el número de puntos suficientes
        # Reasigned_points guarda los puntos que variaron para modificarlos en el clúster correspondiente
        # Considero que el dataset ha variado sí hay un x% de puntos que han cambiado su cluster
        reasigned_points, variated = hasSignificantVariation(
            (x, y), optimal_centroids, labels, accumulated_moved_points)
        print(f"\nCoeficiente de reasignacion: {reassigment_coefficient}")

        if (max_iterations > 1):
            # Reconfiguramos los clusters si muchos puntos han cambiado de cluster en un mismo instante, o si llevamos una acumulación de puntos movidos
            # desde la última reconfiguración de al menos el reclustered_points_percentage_to_recalculate% de los puntos del dataset.
            if ((reassigment_coefficient > reassigment_coefficient_threshold) or variated):
                optimal_centroids, labels, optimal_clusters = executeKmeans(
                    max_clusters_to_calculate, reasigned_points)
                print("\nClústeres recalculados\n")
                reassigment_coefficient = 0
                accumulated_moved_points.clear()
                reclustered_points_since_last_iteration.clear()
                last_n_post_reasigned_points_configurations.append([])
            else:  # Si no ha variado lo suficiente, reasignamos los puntos a los clusteres correspodientes y los representamos
                print("\nLos siguientes puntos se reasignaran de cluster:",
                      reasigned_points)
                labels, post_reasigned_points = reasign_and_show(
                    reasigned_points, labels, optimal_centroids)
                last_n_post_reasigned_points_configurations.append(
                    post_reasigned_points)
        print("-------------------------------------")
        last_n_reasigned_points_configurations.append(
            reasigned_points)
        last_n_optimal_clusters_configurations.append(optimal_clusters)
        last_n_optimal_centroids_configuration.append(optimal_centroids)
        window.refresh()


if __name__ == "__main__":
    has_loaded_file = False
    executed_once = False
    plots = None
    ax = None
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_plot:
        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, 'Close'):
                break
            elif event == "Cargar dataset":
                has_loaded_file = setDataset(values["-FILE-"])
                if has_loaded_file:
                    sg.popup("Dataset cargado")
                else:
                    sg.popup(
                        "No se encontró el dataset. Por favor, seleccione uno nuevo.")
            elif event == 'Ejecutar':
                if not has_loaded_file:
                    sg.popup("ERROR. Cargue un archivo antes de ejecutar.")
                elif plot_columns * plot_rows > max_iterations:
                    sg.popup(
                        "\nERROR\nAjuste el número de filas y columnas para que su multiplicación sea mayor o igual que las iteraciones.")
                elif values["-FILE-"][-4:] != ".csv":
                    sg.popup("El archivo introducido no es un .csv")
                elif len(values["-FILE-"]) == 0:
                    sg.popup("\nERROR\nCargue un dataset antes de ejecutar.")
                else:
                    setParameters(values)
                    plots, ax = plt.subplots(
                        plot_rows, plot_columns, figsize=(20, 10))
                    plot_index = [0, 0]
                    execute()
                    executed_once = True
                    plots.savefig(temp_plot.name, dpi=150)
                    plots.show()
            elif event == 'Distribución de puntos actual':
                if has_loaded_file:
                    initial_dataset, ax_init_dataset = plt.subplots(1, 1)
                    ax_init_dataset.scatter(x, y)
                    plt.show()
                else:
                    sg.popup("\nERROR\nCargue un dataset antes de ejecutar.")
            elif event == 'Mostrar configuración de clústeres actual':
                if executed_once:
                    image = Image.open(rf"{temp_plot.name}")
                    plt.imshow(image)
                    plt.show()
                else:
                    sg.popup("Ejecute al menos una vez el algoritmo")
            elif event == 'Generar dataset':
                generateRandomDataset(int(values["-POINTSTOGENERATE-"]))

    window.close()
