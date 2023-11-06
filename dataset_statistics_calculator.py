import json
#from sklearn import metrics
"""
    'time': str(global_time),
    'reclustered': reclustered,
    'clusters': optimal_clusters,
    'ch_index': ch_index,
    'labels': labels,
    'centroids': optimal_centroids,
    'reassigned': reassigned_points,
    'post_reassigned': [],
    'old_coords': (old_x, old_y),
    'occupied': occupied_points
"""    

def compare_json_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        json1 = json.load(f1)
        json2 = json.load(f2)

    f1_not_needed_reclusterings = 0
    f2_not_needed_reclusterings = 0
    with open(f"{file_path_1}+{file_path_2}_COMPARISON.txt", "w") as f_output:
        previous_f1_entry = None
        previous_f2_entry = None
        for f1_entry, f2_entry in zip(json1, json2):
            f_output.write(str(f1_entry['time']))
            f_output.write(" {\n\t")
            f_output.write("Ambos clusterizaron: ")
            f_output.write(f"{f1_entry['reclustered'] == f2_entry['reclustered']} - (ALWAYS RECLUSTER: {f1_entry['reclustered']}, RECLUSTERING COEFFICIENT: {f2_entry['reclustered']})")
            # Era necesario si:
            #   Los clusters tienen diferentes puntos
            #   El numero de clusteres ha variado
            #   EL CH de la nueva configuración es mejor q la anterior
            f1_clusters_changed = previous_f1_entry != None and previous_f1_entry['clusters'] != f1_entry['clusters']
            f2_clusters_changed = previous_f2_entry != None and previous_f2_entry['clusters'] != f2_entry['clusters']

            f1_clusters_number_changed = previous_f1_entry != None and len(previous_f1_entry['clusters']) != len(f1_entry['clusters'])
            f2_clusters_number_changed = previous_f2_entry != None and len(previous_f2_entry['clusters']) != len(f2_entry['clusters'])

            f1_ch_index_better_than_previous_f2_entry_ch_index = previous_f2_entry != None and f1_entry['ch_index'] > previous_f2_entry['ch_index']

            f1_not_needed_clustering = previous_f1_entry != None and f1_entry['reclustered'] and not f1_clusters_changed and not f1_clusters_number_changed and previous_f1_entry['ch_index'] >= f1_entry['ch_index']
            f2_not_needed_clustering = previous_f2_entry != None and f2_entry['reclustered'] and not f2_clusters_changed and not f2_clusters_number_changed and not f1_ch_index_better_than_previous_f2_entry_ch_index
            # Si se ha reclusterizado cuando no ha habido cambio de puntos de cluster, variación en el número de clusters y el índice CH
            # es peor que si no se hubiese reclusterizado, se considera que no se debería haber reclusterizado.
            if f1_not_needed_clustering:
                f1_not_needed_reclusterings = f1_not_needed_reclusterings + 1

            # Si aplicando el índice, se ha reclusterizado cuando no ha habido cambio de puntos de cluster, variación en el número de clusters y el índice CH
            # es peor que si no se hubiese reclusterizado, se considera que no se debería haber reclusterizado.

            if f2_not_needed_clustering:
                f2_not_needed_reclusterings = f2_not_needed_reclusterings + 1
                
            f_output.write("\n\t¿Era necesario?\n\t\t(ALWAYS RECLUSTER): ")
            f_output.write(str(not f1_not_needed_clustering))
            f_output.write("\n\t\t(RECLUSTERING COEFFICIENT): ")
            f_output.write(str(not f2_not_needed_clustering))
            if f1_not_needed_clustering == "No" and f1_entry['reclustered']:
                f1_not_needed_reclusterings = f1_not_needed_reclusterings + 1
            if f2_not_needed_clustering == "No" and f2_entry['reclustered']:
                f2_not_needed_reclusterings = f2_not_needed_reclusterings + 1
            f_output.write("\n\t")
            f_output.write(f"\tNumero de clusters (ALWAYS RECLUSTER): {len(f1_entry['clusters'])}\n\t")
            f_output.write(f"\tNumero de clusters (RECLUSTERING COEFFICIENT): {len(f2_entry['clusters'])}\n")
            f_output.write("}\n")
            previous_f1_entry = f1_entry
            previous_f2_entry = f2_entry
        f_output.write("Estadísticas:\n\tReclusters innecesarios:\n")
        f_output.write(f"\t\t(ALWAYS RECLUSTER): {f1_not_needed_reclusterings}\n")
        f_output.write(f"\t\t(RECLUSTERING COEFFICIENT): {f2_not_needed_reclusterings}")
        

# Provide the file paths to the JSON files you want to compare
pre_path = "/home/hugo/Documents/TFGs/"
file_path_1 = 'bicis_3mins_URcl0_Cls10_Dm0.1_Dc0.9_time109.81.json'
file_path_2 = 'bicis_3mins_URcl0.1_Cls10_Dm0.1_Dc0.9_time65.94.json'

compare_json_files(file_path_1, file_path_2)
