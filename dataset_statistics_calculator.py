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

    f1_times_reclustered = 0
    f2_times_reclustered = 0
    f1_not_needed_reclusterings = 0
    f1_not_needed_reclusterings = 0
    f2_not_needed_reclusterings = 0
    f2_not_detected_needed_reclusterings = 0
    times_f1_ch_index_is_better_than_f2_ch_index = 0
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
            #   El CH de la nueva configuración es mejor q la anterior
            if previous_f1_entry == None or previous_f2_entry == None:
               f1_should_cluster = True 
               f2_should_cluster = True 
            else:
                f1_clusters_changed = previous_f1_entry['clusters'] != f1_entry['clusters']
                f2_clusters_changed = previous_f2_entry['clusters'] != f2_entry['clusters']

                f1_clusters_number_changed = len(previous_f1_entry['clusters']) != len(f1_entry['clusters'])
                f2_clusters_number_changed = len(previous_f2_entry['clusters']) != len(f2_entry['clusters'])

                f1_should_cluster = f1_clusters_changed and f1_clusters_number_changed and previous_f1_entry['ch_index'] < f1_entry['ch_index']
                f2_should_cluster = f2_clusters_changed and f2_clusters_number_changed and previous_f2_entry['ch_index'] < f2_entry['ch_index']
                
            # Si aplicando el índice, se ha reclusterizado cuando no ha habido cambio de puntos de cluster, variación en el número de clusters y el índice CH
            # es peor que si no se hubiese reclusterizado, se considera que no se debería haber reclusterizado.

            if f1_entry['reclustered']: 
                f1_times_reclustered = f1_times_reclustered + 1
                if not f1_should_cluster:
                    f1_not_needed_reclusterings = f1_not_needed_reclusterings + 1

            # Si aplicando el índice, se ha reclusterizado cuando no ha habido cambio de puntos de cluster, variación en el número de clusters y el índice CH
            # es peor que si no se hubiese reclusterizado, se considera que no se debería haber reclusterizado.
            if f2_entry['reclustered']:
                f2_times_reclustered = f2_times_reclustered + 1
                if not f2_should_cluster:
                    f2_not_needed_reclusterings = f2_not_needed_reclusterings + 1
            
            if not f2_entry['reclustered'] and f1_entry['ch_index'] > f2_entry['ch_index'] and len(f1_entry['clusters']) != len(f2_entry['clusters']) and f1_entry['clusters'] != f2_entry['clusters']:
                f2_not_detected_needed_reclusterings = f2_not_detected_needed_reclusterings + 1
                
            if f1_entry['ch_index'] > f2_entry['ch_index']:
                times_f1_ch_index_is_better_than_f2_ch_index = times_f1_ch_index_is_better_than_f2_ch_index + 1

            f_output.write("\n\t¿Era necesario?\n\t\t(ALWAYS RECLUSTER): ")
            f_output.write(str(f1_should_cluster))
            f_output.write("\n\t\t(RECLUSTERING COEFFICIENT): ")
            f_output.write(str(f2_should_cluster))
            f_output.write("\n\t")
            f_output.write(f"\tNumero de clusters (ALWAYS RECLUSTER): {len(f1_entry['clusters'])}\n\t")
            f_output.write(f"\tNumero de clusters (RECLUSTERING COEFFICIENT): {len(f2_entry['clusters'])}\n")
            f_output.write("}\n")
            previous_f1_entry = f1_entry
            previous_f2_entry = f2_entry
        f_output.write("Estadísticas:\n")
        f_output.write(f"\tVeces reclusterizado:\n")
        f_output.write(f"\t\t(ALWAYS RECLUSTER): {f1_times_reclustered}\n")
        f_output.write(f"\t\t(RECLUSTERING COEFFICIENT): {f2_times_reclustered}")
        f_output.write("\n\t\tReclusters innecesarios:\n")
        f_output.write(f"\t\t\t(ALWAYS RECLUSTER): {f1_not_needed_reclusterings}\n")
        f_output.write(f"\t\t\t(RECLUSTERING COEFFICIENT): {f2_not_needed_reclusterings}")
        f_output.write("\n\t\tReclusters necesarios no detectados:\n")
        f_output.write(f"\t\t\t(RECLUSTERING COEFFICIENT): {f2_not_detected_needed_reclusterings}")
        f_output.write(f"\n\n\tVeces que ALWAYS RCL CH_IDX > RCL COEFF CH_IDX: {times_f1_ch_index_is_better_than_f2_ch_index}\n")
        

# Provide the file paths to the JSON files you want to compare
pre_path = "/home/hugo/Documents/TFGs/"
file_path_1 = 'bicis_60mins_URcl0_Cls20_Dm0.1_Dc0.9_time20.58.json'
file_path_2 = 'bicis_60mins_URcl0.1_Cls20_Dm0.1_Dc0.9_time14.3.json'

compare_json_files(file_path_1, file_path_2)
