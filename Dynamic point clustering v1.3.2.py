import datetime
import json
from datetime import datetime, timedelta, timezone
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import PySimpleGUI as sg
from sklearn import metrics
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Global vars
global_time = None # Tiempo de la aplicación para ver cuando reclusterizar
end_time = None # Tiempo de fin del dataset
time_interval = 3 # Tiempo de incremento del tiempo global en cada ejecución
file = None
x = []  # Lista de valores x
y = []  # Lista de valores y
old_x = []  # Lista de valores x
old_y = []  # Lista de valores y
old_x_list = []  # Lista de valores x
old_y_list = []  # Lista de valores y
groups = [] # Guarda las rutas que ha realizado cada bicicleta
reassigned_points = []  # Puntos que se reasignaron desde la ultima iteración
# Puntos que se movieron desde que se realizó la última clusterización (Elimina repetidos)
accumulated_moved_points = []
post_reassigned_points = []  # Coordenadas de los puntos q se reasignaron
# Puntos que se reasignaron desde la ultima iteración
last_n_post_reassigned_points_configurations = []
# Puntos que se reasignaron desde la ultima iteración
last_n_reassigned_points_configurations = []
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
max_clusters_to_calculate = 20
# Porcentaje en valor de 0 a 1 de que cuantos puntos que se tiene que mover para que se recalculen los clústeres
reassigment_coefficient_threshold = 0.1
# Multiplicador de la variación de movimiento
delta_m = 0.1
# Multiplicador de la variación puntos de cluster
delta_c = 0.9
# Coeficiente de reasignación
reassignment_coefficient = 0
# Velocidad de simulación (Tiempo a dormir entre cada configuración de clusteres)
#####


# Confifuración de la IU
point_size = 5

simulation_speed = 0.1

sg.theme("DarkTanBlue")

frame_layout = [[sg.Multiline("", size=(72, 15), autoscroll=True,
                              reroute_stdout=True, reroute_stderr=False, key='-OUTPUT-', expand_x=True, expand_y=True)]]

dataset_load_frame = [
    [sg.InputText(key="-FILE-", expand_x=True),
     sg.FileBrowse(file_types=(("CSV", "*.csv"),)),
     sg.Button('Cargar dataset', expand_x=True)]
]
simulation_frame = [
    [sg.InputText(key="-JSONFILE-", expand_x=True), 
     sg.FileBrowse(file_types=(("JSON", "*.json"),))], 
    [sg.Button('Mostrar una configuración concreta', expand_x=True), 
     sg.Button('Simular JSON', expand_x=True)],

]

reclusterization_variables_frame = [
    [sg.Text("Delta_m (Decimales [0.0, 1.0]):", expand_x=True),
     sg.Input(key="-DELTAM-", size=4,
              default_text=0.1, expand_x=True)],
    [sg.Text("Delta_c (Decimales [0.0, 1.0]):", expand_x=True),
     sg.Input(key="-DELTAC-", size=9, default_text=f"{str(1-delta_m)}", readonly=True, expand_x=True, text_color="black")],
    [sg.Text("Incremento de tiempo global (En minutos):", expand_x=True),
     sg.Input(key="-DELTATIME-", size=4, default_text=time_interval, expand_x=True)],
    [sg.Text("Umbral del coeficiente para reclusterizar (Decimales [0.0, 1.0]):", expand_x=True),
     sg.Input(key="-RCTHRESHOLD-", size=4, default_text=0.1, expand_x=True),
     sg.Text("\n", expand_x=True)],
    [sg.Text("Numero de agrupaciones entre las que calcular el óptimo (min. 3):", expand_x=True),
     sg.Input(key="-MAXPOINTCLUST-", size=4, default_text=10, expand_x=True), sg.Text("\n", expand_x=True)],
    [sg.Text("Tamaño de los puntos:", expand_x=True),
     sg.Input(key="-POINTSIZE-", size=4, default_text=point_size, expand_x=True), sg.Text("\n", expand_x=True)],
    [sg.Text("Velocidad de simulación (Tiempo en segundos\nque se representa cada gráfico (Decimales [0.0, 1.0])):", expand_x=True),
     sg.Input(key="-SIMSPEED-", size=4, default_text=simulation_speed, expand_x=True), sg.Text("\n", expand_x=True)],
]

layout = [
    [
        sg.Column([
            [sg.Frame("SELECCIÓN DE DATASET", dataset_load_frame, font="Any 12", expand_x=True)],
            [sg.Frame("CONFIGURACIÓN DE PARÁMETROS", reclusterization_variables_frame, font="Any 12", expand_x=True)],
            [sg.Button('Ejecutar', expand_x=True)],
            [sg.Frame("SIMULACIÓN", simulation_frame, font="Any 12", expand_x=True)],
            [sg.Frame("Salida de consola", frame_layout, expand_x=True, expand_y=True),],
        ], expand_x=True, expand_y=True),
        sg.VerticalSeparator(),
        sg.Column([
            [sg.Canvas(key="-CANVAS-", expand_x=True, expand_y=True)],
            [sg.Checkbox('Alternar desplazamiento de puntos reasignados. (Requiere recargar la entrada)', key='-TOGGLETRACE-', default=True)],
            [sg.Button('Cargar anterior', expand_x=True, disabled=True, key="-LOADPREV-", button_color="grey"), sg.Button('Cargar siguiente', expand_x=True, disabled=True, key="-LOADNEXT-", button_color="grey"), sg.Button('Borrar gráfico', expand_x=True, button_color="grey", disabled=True, key="-ERASEPLOT-")],
            [sg.Button('Abortar simulación', key="-ABORT-", expand_x=True, disabled=True, button_color="grey"),]
        ], expand_x=True, expand_y=True, justification="center"),
    ],
]

window = sg.Window('Dynamic point clustering', layout,
                   resizable=True, finalize=True)
fig, ax = plt.subplots()
canvas_elem = window["-CANVAS-"]

canvas = FigureCanvasTkAgg(fig, master=canvas_elem.Widget)
canvas.get_tk_widget().pack(expand=True, fill="both")

####


def setParameters(values):
    global point_size
    global max_clusters_to_calculate
    global reassigment_coefficient_threshold
    global delta_c
    global delta_m
    global time_interval
    global simulation_speed
    max_clusters_to_calculate = eval(values["-MAXPOINTCLUST-"])
    point_size = eval(values["-POINTSIZE-"])
    reassigment_coefficient_threshold = eval(
        values["-RCTHRESHOLD-"])
    delta_m = eval(values["-DELTAM-"])
    time_interval = eval(values["-DELTATIME-"])
    delta_c = 1-delta_m
    simulation_speed = eval(values["-SIMSPEED-"])
    window["-MAXPOINTCLUST-"].update(max_clusters_to_calculate)
    window["-POINTSIZE-"].update(point_size)
    window["-RCTHRESHOLD-"].update(reassigment_coefficient_threshold)
    window["-DELTATIME-"].update(time_interval)
    window["-DELTAM-"].update(delta_m)
    window["-DELTAC-"].update(delta_c)
    window["-SIMSPEED-"].update(simulation_speed)

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


def executeKmeans(max_clusters, reassigned_points, occupied_points=[[], []]):
    n_max_clusteres = max_clusters
    n_clusters_list = []
    clusters_CH_index = []
    centroids_list = []
    labels_list = []
    global old_x
    global old_y

    # Calculamos N clústeres para comprobar cual es la agrupación óptima.
    for n_clusters in range(2, n_max_clusteres):
        try:
            # Capturamos los clústeres y las etiquetas
            centroids, clusters, labels = kmeans(n_clusters, (x, y))
            # Calculamos el índice de Calinski-Harabasz utilizando el método proprocionado por la biblioteca scikit.learn.
            clusters_CH_index.append(
                metrics.calinski_harabasz_score(list(zip(x, y)), labels))
            n_clusters_list.append(clusters)
            centroids_list.append(centroids)
            labels_list.append(labels)

        except Exception:
            print(f"k={n_clusters} - Configuración con clústeres vacíos detectada. Límite establecido en {n_clusters - 1} para esta configuración de puntos.")
            break

    # Escogemos la agrupación de clústeres con el mayor índice como el óptimo relativo
    i = clusters_CH_index.index(max(clusters_CH_index))
    clusters = n_clusters_list[i]
    # Guarda las etiquetas de a que cluster está asignado cada punto
    labels = labels_list[i]
    centroids = centroids_list[i]

    add_clusters_to_plot(clusters, list(zip(*centroids)), reassigned_points, [], (old_x, old_y), occupied_points, reclustered=True)
    return centroids, labels, clusters


def hasSignificantVariation(newPoints, centroids, labels, accumulated_moved_points):
    # reasigna los puntos a otro centroide si estos puntos han variado lo suficiente
    global reassignment_coefficient
    global last_reclusterization_labels
    reassigned_points = has_reassigned_points(centroids, labels, newPoints)

    for point in reassigned_points:
        if (point[0] in reclustered_points_since_last_iteration):
            if (last_reclusterization_labels[point[0]] == point[1]):
                reclustered_points_since_last_iteration.remove(point[0])
        else:
            reclustered_points_since_last_iteration.append(point[0])

    reassignment_coefficient = (1/len(labels)) * (
        (delta_m * len(accumulated_moved_points)) + (delta_c * len(reclustered_points_since_last_iteration)))
    # Si varían el reclustered_points_percentage_to_recalculate% de los puntos a la vez, se recalculan los clusteres
    if (reassignment_coefficient > reassigment_coefficient_threshold):
        return reassigned_points, True
    else:
        return reassigned_points, False

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
def has_reassigned_points(centroids, labels, newPoints):
    # Guarda el indice de los puntos que se han cambiado de cluster y el cluster al que se asignan
    global reassigned_points
    reassigned_points = []
    for i in range(len(labels)):
        distance_to_centroids = []
        for centroid in centroids:
            distance_to_centroids.append(math.sqrt(pow(
                (centroid[0] - newPoints[0][i]), 2) + pow((centroid[1] - newPoints[1][i]), 2)))
        closest_centroid_index = distance_to_centroids.index(
            min(distance_to_centroids))
        if (closest_centroid_index != labels[i]):
            reassigned_points.append((i, closest_centroid_index))
    return reassigned_points


def add_clusters_to_plot(clusters, centroids, reassigned_points, post_reassigned_points, old_coords, occupied_points=[[], []], reclustered=False):
    global point_size
    global ax

    old_x = old_coords[0]
    old_y = old_coords[1]
    ax.clear()
    ax.set_ylim(min(y) - 0.01, max(y) + 0.01)
    ax.set_xlim(min(x) - 0.01, max(x) + 0.01)
    ax.set_title(
        "Cls: " + str(len(clusters)) + " | Rsg pts: " + str(len(reassigned_points)) + " | Occ pts: " + str(len(occupied_points[0])) + " | Rec:" + str(reclustered))
    # Puntos que se reasignaron
    reassigned_points_coords = [[], []]
    if (old_x != [] and old_y != []):
        for point in reassigned_points:
            reassigned_points_coords[0].append(old_x[point[0]])
            reassigned_points_coords[1].append(old_y[point[0]])

    for cluster in clusters:
        ax.plot(
            cluster[0], cluster[1], 'o', markersize=point_size)
        ax.plot(
            centroids[0], centroids[1], 'X', markersize=point_size-2, color="black")
    
    ax.plot(
        occupied_points[0], occupied_points[1], '.', markersize=point_size-0.5, color="white")

    if (post_reassigned_points != [] and old_x != [] and old_y != [] and values["-TOGGLETRACE-"]):
        start_coords = list(zip(reassigned_points_coords[0], reassigned_points_coords[1]))
        end_coords = list(zip(post_reassigned_points[0], post_reassigned_points[1]))
        x_limit = plt.xlim()[1] - plt.xlim()[0]
        y_limit = plt.ylim()[1] - plt.ylim()[0]
        for entry in zip(start_coords, end_coords):
            # Dibuja el recorrido de un punto
            arrow_dx = entry[1][0] - entry[0][0]
            arrow_dy = entry[1][1] - entry[0][1]
            arrow_length = (arrow_dx**2 + arrow_dy**2)**0.5
            head_size = 0.01 * min(x_limit, y_limit)
            tail_width = 0.001 * min(x_limit, y_limit)
            head_length = head_size
            arrow_dx_adjusted = arrow_dx - (head_length / arrow_length) * arrow_dx
            arrow_dy_adjusted = arrow_dy - (head_length / arrow_length) * arrow_dy

            plt.arrow(entry[0][0], entry[0][1], arrow_dx_adjusted, arrow_dy_adjusted,
                    head_width=head_size, head_length=head_length, width=tail_width, fc='grey', ec='grey', zorder=2)

    canvas.draw()



def reassign_points(reassigned_points, labels, centroids, occupied_points):
    reassigned_points_index = 0  # Indica por que punto de reasignación voy
    clusters = []
    post_reassigned_points = [[], []]
    global x
    global old_x
    global y
    global old_y
    for i in range(max(labels) + 1):
        clusters.append([[], [], []])

    # Copia los clusteres con los puntos reasignados
    for i in range(len(labels)):
        if (reassigned_points_index < len(reassigned_points) and i == reassigned_points[reassigned_points_index][0]):
            clusters[reassigned_points[reassigned_points_index]
                     [1]][0].append(x[i])
            clusters[reassigned_points[reassigned_points_index]
                     [1]][1].append(y[i])
            labels[i] = reassigned_points[reassigned_points_index][1]
            post_reassigned_points[0].append(x[i])
            post_reassigned_points[1].append(y[i])
            reassigned_points_index = reassigned_points_index + 1
        else:
            clusters[labels[i]][0].append(x[i])
            clusters[labels[i]][1].append(y[i])

    add_clusters_to_plot(clusters, list(zip(*centroids)), reassigned_points, post_reassigned_points, (old_x, old_y), occupied_points)
    
    return clusters, labels, post_reassigned_points


def setDataset(dataset):
    global input_values
    global x
    x = []
    global y
    y = []
    global global_time
    global end_time
    global file
    global groups
    groups.clear()

    try:
        file = pd.read_csv(dataset)
    except Exception as e:
        print("\n", e)
        return False

    grouped_file = file.groupby('id') # Agrupamos el dataset por número de bicicleta
    for value in grouped_file.groups: # Recorre los grupos cogiendo la primera entrada de ellos para la
        if('timestamp' in  file.columns):
            route_timestamp = datetime.strptime((grouped_file.get_group(value).iloc[0])['timestamp'], '%Y-%m-%d %H:%M:%S%z')
            if(global_time == None):
                global_time = route_timestamp
                end_time = route_timestamp

            elif(route_timestamp < global_time):
                global_time = route_timestamp
        
        x.append(float((grouped_file.get_group(value).iloc[0])['longitude']))
        y.append(float((grouped_file.get_group(value).iloc[0])['latitude']))

    for _, group_data in grouped_file: # Guardamos los grupos en una lista del tipo [ [[(), ()], [(), ()]], ...]
        subgrouped = group_data.groupby('route_code')
        subgroup_list = []
        for _, subgroup_data in subgrouped:
            interval = []
            for _, entry in subgroup_data.iterrows():
                interval.append(entry)
            subgroup_list.append(tuple(interval))
        groups.append(subgroup_list)

    end_time = datetime.strptime(file['timestamp'].max(), '%Y-%m-%d %H:%M:%S%z')

    global y_axis_limits
    y_axis_limits = (min(y), max(y))
    global x_axis_limits
    x_axis_limits = (min(x), max(x))

    ax.clear()
    ax.plot(x, y, "o", color="black")
    ax.set_ylim(min(y) - 0.01, max(y) + 0.01)
    ax.set_xlim(min(x) - 0.01, max(x) + 0.01)
    ax.set_title("Dataset: " + dataset)
    canvas.draw()

    return True  # El dataset se pudo cargar con éxito

def updatePoints(old_x, old_y, group_entry_idx):
    global global_time
    global file
    global x
    global y
    global groups
    occupied_points_idx = [] # Puntos que no pertenecen a ningún clúster
    occupied_points = [[], []] # Puntos ocupados
    moved_points = [] # Puntos con las posiciones actualizadas
    point_idx = 0

    for group in groups: # Iteracion de grupos
        last_read_entry_idx = group_entry_idx[point_idx]
        
        while True:
            trip_entries = group[last_read_entry_idx] # Iteracion de los subgrupos de rutas
            if len(trip_entries) == 2:
                if(global_time >= datetime.strptime((trip_entries[1])['timestamp'], '%Y-%m-%d %H:%M:%S%z')): # Si todos los tiempos son menores que el global
                        last_read_entry_idx = last_read_entry_idx + 1
                        if(last_read_entry_idx == len(group)):
                            x.append(float((trip_entries[1])['longitude']))
                            y.append(float((trip_entries[1])['latitude']))
                            if(x[-1] != old_x[point_idx] or y[-1] != old_y[point_idx]):
                                moved_points.append(point_idx)
                            break

                else: # Si un tiempo es mayor que el global
                    x.append(float((trip_entries[0])['longitude']))
                    y.append(float((trip_entries[0])['latitude']))
                    if(global_time >= datetime.strptime((trip_entries[0])['timestamp'], '%Y-%m-%d %H:%M:%S%z')):
                        occupied_points_idx.append(point_idx)
                        occupied_points[0].append((trip_entries[0])['longitude'])
                        occupied_points[1].append((trip_entries[0])['latitude'])
                    if(x[-1] != old_x[point_idx] or y[-1] != old_y[point_idx]):
                        moved_points.append(point_idx)
                    group_entry_idx[point_idx] = last_read_entry_idx
                    break
            else:
                x.append(float((trip_entries[0])['longitude']))
                y.append(float((trip_entries[0])['latitude']))
                occupied_points_idx.append(point_idx)
                occupied_points[0].append((trip_entries[0])['longitude'])
                occupied_points[1].append((trip_entries[0])['latitude'])
                if(x[-1] != old_x[point_idx] or y[-1] != old_y[point_idx]):
                    moved_points.append(point_idx)
                group_entry_idx[point_idx] = last_read_entry_idx
                break

        point_idx = point_idx + 1

    return moved_points, occupied_points, group_entry_idx
            


def execute(output_file_name):
    global global_time
    global time_interval
    global reassigned_points
    global accumulated_moved_points
    global post_reassigned_points
    # Indica cuándo se debería reclusterizar o reasignar.
    global reassignment_coefficient
    global last_reclusterization_labels
    optimal_centroids = []
    labels = []
    optimal_clusters = []
    global last_n_optimal_centroids_configuration
    global last_n_optimal_clusters_configurations
    global last_n_post_reassigned_points_configurations
    global last_n_reassigned_points_configurations
    global old_x
    global old_y
    global old_x_list
    global old_y_list
    global x
    global y

    last_n_optimal_centroids_configuration.clear()
    last_n_optimal_clusters_configurations.clear()
    last_n_reassigned_points_configurations.clear()
    last_n_post_reassigned_points_configurations.clear()
    old_x_list.clear()
    old_y_list.clear()
    # optimal_centroids Almacena los centroides de la configuración óptima relativa hallada
    # labels Almacena el índice de el clúster al que está asignado cada punto
    optimal_centroids, labels, optimal_clusters = executeKmeans(
        max_clusters_to_calculate, reassigned_points)
    last_n_optimal_clusters_configurations.append(optimal_clusters)
    last_n_optimal_centroids_configuration.append(optimal_centroids)
    last_n_reassigned_points_configurations.append(reassigned_points)
    last_n_post_reassigned_points_configurations.append(
        post_reassigned_points)
    old_x_list.append(old_x.copy())
    old_y_list.append(old_y.copy())
    group_entry_idx = [0,]*len(x) # Almacena el indice de la ultima entrada visitada en cada grupo
    last_reclusterization_labels = labels.copy()
    occupied_points = []
    with open(f"{output_file_name}.json", "w") as file:
        data = [{
            'time': str(global_time),
            'reclustered': True,
            'clusters': optimal_clusters,
            'centroids': optimal_centroids,
            'reassigned': reassigned_points,
            'post_reassigned': [[], []],
            'old_coords': (x, y),
            'occupied': [[], []]
        }]
        json.dump(data, file, indent=4)

    with open(f"{output_file_name}.json", "r") as file:
        json_file = json.load(file)

    dump_it = 0

    while(global_time <= end_time):
        if threading.Event().is_set():
            break
        reclustered = False
        old_x = x.copy()
        old_x_list.append(x)
        old_y = y.copy()
        old_y_list.append(y)
        x.clear()
        y.clear()

        global_time = global_time + timedelta(minutes=time_interval)


        # occupied_points guarda aquellos puntos que no pertencen a ningún clúster
        moved_points, occupied_points, group_entry_idx = updatePoints(old_x, old_y, group_entry_idx)

        for point in moved_points:
            if point not in accumulated_moved_points:
                accumulated_moved_points.append(point)

        # Variated guarda el valor booleano acerca de si variaron el número de puntos suficientes
        # Reasigned_points guarda los puntos que variaron para modificarlos en el clúster correspondiente
        # Considero que el dataset ha variado sí hay un x% de puntos que han cambiado su cluster
        reassigned_points, variated = hasSignificantVariation(
            (x, y), optimal_centroids, labels, accumulated_moved_points)
        print(f"\nCoeficiente de reasignacion: {reassignment_coefficient}")

        # Reconfiguramos los clusters si muchos puntos han cambiado de cluster en un mismo instante, o si llevamos una acumulación de puntos movidos
        # desde la última reconfiguración de al menos el reclustered_points_percentage_to_recalculate% de los puntos del dataset.
        if ((reassignment_coefficient >= reassigment_coefficient_threshold) or variated):
            optimal_centroids, labels, optimal_clusters = executeKmeans(
                max_clusters_to_calculate, reassigned_points, occupied_points)
            reclustered = True
            print("\nClústeres recalculados\n")
            reassignment_coefficient = 0
            accumulated_moved_points.clear()
            reclustered_points_since_last_iteration.clear()
            last_n_post_reassigned_points_configurations.append([])
            data = {
                'time': str(global_time),
                'reclustered': reclustered,
                'clusters': optimal_clusters,
                'labels': labels,
                'centroids': optimal_centroids,
                'reassigned': reassigned_points,
                'post_reassigned': [],
                'old_coords': (old_x, old_y),
                'occupied': occupied_points
            }
        else:  # Si no ha variado lo suficiente, reasignamos los puntos a los clusteres correspodientes y los representamos
            print("\nLos siguientes puntos se reasignaran de cluster:",
                    reassigned_points)
            optimal_clusters, labels, post_reassigned_points = reassign_points(
                reassigned_points, labels, optimal_centroids, occupied_points)
            last_n_post_reassigned_points_configurations.append(
                post_reassigned_points)
            data = {
                'time': str(global_time),
                'reclustered': reclustered,
                'clusters': optimal_clusters,
                'labels': labels,
                'centroids': optimal_centroids,
                'reassigned': reassigned_points,
                'post_reassigned': post_reassigned_points,
                'old_coords': (old_x, old_y),
                'occupied': occupied_points
            }
        print(f"---------------------Hora global actual: {global_time}---------------------")
        last_n_reassigned_points_configurations.append(
            reassigned_points)
        last_n_optimal_clusters_configurations.append(optimal_clusters)
        last_n_optimal_centroids_configuration.append(optimal_centroids)

        json_file.append(data)
        # Write the updated data back to the JSON file
        if(dump_it == 50):
            with open(f"{output_file_name}.json", "w") as file:
                json.dump(json_file, file, indent=4)  # Use indent for formatting
            dump_it = -1
        dump_it = dump_it + 1
        window.refresh()
        
    with open(f"{output_file_name}.json", "w") as file:
        json.dump(json_file, file, indent=4)  # Use indent for formatting
    window["-ABORT-"].update(disabled=True)
    window["-ABORT-"].update(button_color="grey")
    print("--------------------------------\nFIN DE LA EJECUCIÓN\n--------------------------------")
    window["-ERASEPLOT-"].update(disabled=not window["-ERASEPLOT-"].Disabled)
    window["-ERASEPLOT-"].update(button_color="red")

def simulate(jsonFile, sleepTime):
    window["-ABORT-"].update(disabled=True)
    if threading.current_thread().name == "MainThread":
        print("Mainthread exiting")
        window.refresh()
        return  # Skip the function for the main thread
    global x
    global y
    x = jsonFile[0]["old_coords"][0]
    y = jsonFile[0]["old_coords"][1]
    if(sleepTime < 0.1):
        sleepTime = 0.1
        print("Tiempo escogido menor que 0.1. Autocorrigiendo a 0.1.")
    elif(sleepTime > 0.1):
        sleepTime = 1
        print("Tiempo escogido mayor que 2. Autocorrigiendo a 2.")
    for entry in tqdm(jsonFile, unit="item", ascii=True):
        if(threading.Event().is_set()):
            print("Simulación abortada")
            return
        add_clusters_to_plot(entry['clusters'], list(zip(*entry['centroids'])), entry['reassigned'], entry['post_reassigned'], entry['old_coords'], entry['occupied'], entry['reclustered'])
        time.sleep(sleepTime)
        print("\n")
        window.refresh()

    window["-ABORT-"].update(disabled=True)
    window["-ABORT-"].update(button_color="grey")
    window["-ERASEPLOT-"].update(disabled=False)
    window["-ERASEPLOT-"].update(button_color="red")
    window["-TOGGLETRACE-"].update(disabled=False)
    print("--------------------------------\nFIN DE LA SIMULACIÓN\n--------------------------------")
    

def showCertainConfiguration(jsonFile, entryId):
    global x
    global y
    global abort_simulation
    x = jsonFile[0]["old_coords"][0]
    y = jsonFile[0]["old_coords"][1]
    entry = jsonFile[entryId]
    print(f"Fecha de la entrada: {entry['time']}")
    add_clusters_to_plot(entry['clusters'], list(zip(*entry['centroids'])), entry['reassigned'], entry['post_reassigned'], entry['old_coords'], entry['occupied'], entry['reclustered'])



if __name__ == "__main__":
    has_loaded_file = False
    currentConfig = None
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Close'):
            break

        elif event == "Cargar dataset":
            try:
                has_loaded_file = setDataset(values["-FILE-"])
                if has_loaded_file:
                    sg.popup("Dataset cargado")
                else:
                    sg.popup(
                        "No se encontró el dataset. Por favor, seleccione uno nuevo.")
            except Exception as e:
                sg.popup("Puede que la estructura del archivo sea incorrecta. Por favor, asegúrese de que la estructura sea:\n\nroute_code,id,latitude,longitude,timestamp", title="Error al cargar")
                
        elif event == 'Ejecutar':
            if not has_loaded_file:
                sg.popup("ERROR. Cargue un archivo antes de ejecutar.")
            elif values["-FILE-"][-4:] != ".csv":
                sg.popup("El archivo introducido no es un .csv")
            elif len(values["-FILE-"]) == 0:
                sg.popup("\nERROR\nCargue un dataset antes de ejecutar.")
            else:
                setParameters(values)
                text = sg.popup_get_text("Introduzca el nombre del archivo de salida:")
                if text:
                    if not window["-LOADPREV-"].Disabled:
                        window["-LOADPREV-"].update(disabled=True)
                        window["-LOADPREV-"].update(button_color="grey")
                    if not window["-LOADNEXT-"].Disabled:
                        window["-LOADNEXT-"].update(disabled=True)
                        window["-LOADNEXT-"].update(button_color="grey")
                    if not window["-ERASEPLOT-"].Disabled:
                        window["-ERASEPLOT-"].update(disabled=True)
                        window["-ERASEPLOT-"].update(button_color="grey")
                    window["-TOGGLETRACE-"].update(disabled=True)
                    execute(text)
                    window["-TOGGLETRACE-"].update(disabled=False)
        elif event == '-ERASEPLOT-':
            ax.clear()
            canvas.draw()
            window["-ERASEPLOT-"].update(disabled=True)
            window["-ERASEPLOT-"].update(button_color="grey")
            window["-LOADPREV-"].update(disabled=True)
            window["-LOADPREV-"].update(button_color="grey")
            window["-LOADNEXT-"].update(disabled=True)
            window["-LOADNEXT-"].update(button_color="grey")
        
        elif event == '-LOADPREV-':
            try:
                with open(values["-JSONFILE-"], "r") as jsonFile:
                    data = json.load(jsonFile)
                currentConfig = currentConfig - 1
                showCertainConfiguration(data, currentConfig)
                if(currentConfig == 0):
                    window["-LOADPREV-"].update(disabled=True)
                    window["-LOADPREV-"].update(button_color="grey")
                else:
                    window["-LOADPREV-"].update(disabled=False)
                    window["-LOADPREV-"].update(button_color="blue")
                    window["-LOADNEXT-"].update(disabled=False)
                    window["-LOADNEXT-"].update(button_color="blue")
                print(f"Entrada {currentConfig}/{len(data)-1} cargada.\n")

            except Exception as e:
                print(e)

        elif event == '-LOADNEXT-':
            try:
                with open(values["-JSONFILE-"], "r") as jsonFile:
                    data = json.load(jsonFile)
                currentConfig = currentConfig + 1
                showCertainConfiguration(data, currentConfig)
                if(currentConfig == len(data)-1):
                    window["-LOADNEXT-"].update(disabled=True)
                    window["-LOADNEXT-"].update(button_color="grey")
                else:
                    window["-LOADPREV-"].update(disabled=False)
                    window["-LOADPREV-"].update(button_color="blue")
                    window["-LOADNEXT-"].update(disabled=False)
                    window["-LOADNEXT-"].update(button_color="blue")
                print(f"Entrada {currentConfig}/{len(data) - 1} cargada.\n")

            except Exception as e:
                print(e)

        elif event == "Simular JSON":
            threading.Event().clear()
            try:
                with open(values["-JSONFILE-"], "r") as jsonFile:
                    data = json.load(jsonFile)
                if not window["-LOADPREV-"].Disabled:
                    window["-LOADPREV-"].update(disabled=True)
                    window["-LOADPREV-"].update(button_color="grey")
                if not window["-LOADNEXT-"].Disabled:
                    window["-LOADNEXT-"].update(disabled=True)
                    window["-LOADNEXT-"].update(button_color="grey")
                if not window["-ERASEPLOT-"].Disabled:
                    window["-ERASEPLOT-"].update(disabled=True)
                    window["-ERASEPLOT-"].update(button_color="grey")
                window["-TOGGLETRACE-"].update(disabled=True)
                t = threading.Thread(target=simulate, args=(data, simulation_speed))
                window["-ABORT-"].update(button_color="red")
                t.start()
            except Exception as e:
                print(e)

        elif event == 'Mostrar una configuración concreta':
            try:
                with open(values["-JSONFILE-"], "r") as jsonFile:
                    data = json.load(jsonFile)
                currentConfig = int(sg.popup_get_text(f"El dataset seleccionado contiene de 0 a {len(data) - 1} entradas. \nIntroduce el número de la que quieras ver:"))
                if currentConfig < 0:
                    print(f"No existe la entrada {currentConfig}. Cargando la entrada 0")
                    currentConfig = 0
                elif currentConfig >= len(data):
                    print(f"No existe la entrada {currentConfig}. Cargando la entrada {len(data)-1}")
                    currentConfig = len(data) - 1
                showCertainConfiguration(data, currentConfig)
                window["-ERASEPLOT-"].update(disabled=False)
                window["-ERASEPLOT-"].update(button_color="red")
                if(currentConfig != 0):
                    window["-LOADPREV-"].update(disabled=False)
                    window["-LOADPREV-"].update(button_color="blue")
                else:
                    window["-LOADPREV-"].update(disabled=True)
                    window["-LOADPREV-"].update(button_color="grey")
                if(currentConfig != len(data) - 1):
                    window["-LOADNEXT-"].update(disabled=False)
                    window["-LOADNEXT-"].update(button_color="blue")
                else:
                    window["-LOADNEXT-"].update(disabled=True)
                    window["-LOADNEXT-"].update(button_color="grey")

                print(f"Entrada {currentConfig}/{len(data) - 1} cargada\n")

            except Exception as e:
                print(e)
        elif event == "-ABORT-":
            threading.Event().set()
            window["-ABORT-"].update(disabled=True)
            window["-ABORT-"].update(button_color="grey")
            window["-ERASEPLOT-"].update(disabled=False)
            window["-ERASEPLOT-"].update(button_color="red")
            print("Abortando...")
            t.join()
            print("Simulación abortada")
            threading.Event().clear()

    window.close()

