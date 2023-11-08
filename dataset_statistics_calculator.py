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
    f2_not_detected_but_needed_reclusterings = 0
    times_f1_ch_index_is_better_than_f2_ch_index = 0
    with open(f"{file1}+{file2}_COMPARISON.txt", "w") as f_output:
        previous_f1_entry = None
        previous_f2_entry = None
        for non_idx_entry, idx_entry in zip(json1, json2):
            f_output.write(str(non_idx_entry['time']))
            f_output.write(" {\n\t")
            f_output.write("Ambos clusterizaron: ")
            f_output.write(f"{non_idx_entry['reclustered'] == idx_entry['reclustered']} - (ALWAYS RECLUSTER: {non_idx_entry['reclustered']}, RECLUSTERING COEFFICIENT: {idx_entry['reclustered']})")
            # Era necesario si:
            #   Los clusters tienen diferentes puntos
            #   El numero de clusteres ha variado
            #   El CH de la nueva configuración es mejor q la anterior
            if previous_f1_entry == None or previous_f2_entry == None:
               f1_should_cluster = True 
               f2_should_cluster = True 
            else:
                f1_clusters_changed = previous_f1_entry['clusters'] != non_idx_entry['clusters']
                #f2_clusters_changed = previous_f2_entry['clusters'] != f2_entry['clusters']

                f1_clusters_number_changed = len(previous_f1_entry['clusters']) != len(non_idx_entry['clusters'])
                #f2_clusters_number_changed = len(previous_f2_entry['clusters']) != len(f2_entry['clusters'])

                f1_should_cluster = f1_clusters_changed and f1_clusters_number_changed and previous_f1_entry['ch_index'] < non_idx_entry['ch_index']
                #f2_should_cluster = f2_clusters_changed and f2_clusters_number_changed and previous_f2_entry['ch_index'] < f2_entry['ch_index']
                
            # Si aplicando el índice, se ha reclusterizado cuando no ha habido cambio de puntos de cluster, variación en el número de clusters y el índice CH
            # es peor que si no se hubiese reclusterizado, se considera que no se debería haber reclusterizado.

            if non_idx_entry['reclustered']: 
                f1_times_reclustered = f1_times_reclustered + 1
                if not f1_should_cluster:
                    f1_not_needed_reclusterings = f1_not_needed_reclusterings + 1

            # Si se ha hecho clustering, pero no se debería haber hecho, se cuenta como innecesario
            if idx_entry['reclustered']:
                f2_times_reclustered = f2_times_reclustered + 1
                if not f1_should_cluster:
                    f2_not_needed_reclusterings = f2_not_needed_reclusterings + 1
            else:
                if f1_should_cluster:
                    f2_not_detected_but_needed_reclusterings = f2_not_detected_but_needed_reclusterings + 1


            if non_idx_entry['ch_index'] > idx_entry['ch_index']:
                times_f1_ch_index_is_better_than_f2_ch_index = times_f1_ch_index_is_better_than_f2_ch_index + 1

            f_output.write("\n\t¿Era necesario?\n\t\t(ALWAYS RECLUSTER): ")
            f_output.write(str(f1_should_cluster))
            f_output.write("\n\t\t(RECLUSTERING COEFFICIENT): ")
            f_output.write(str(f2_should_cluster))
            f_output.write("\n\t")
            f_output.write(f"\tNumero de clusters (ALWAYS RECLUSTER): {len(non_idx_entry['clusters'])}\n\t")
            f_output.write(f"\tNumero de clusters (RECLUSTERING COEFFICIENT): {len(idx_entry['clusters'])}\n")
            f_output.write("}\n")
            previous_f1_entry = non_idx_entry
            previous_f2_entry = idx_entry
        f_output.write("Estadísticas:\n")
        f_output.write(f"\tVeces reclusterizado:\n")
        f_output.write(f"\t\t(ALWAYS RECLUSTER): {f1_times_reclustered}\n")
        f_output.write(f"\t\t(RECLUSTERING COEFFICIENT): {f2_times_reclustered}")
        f_output.write("\n\t\tReclusters innecesarios:\n")
        f_output.write(f"\t\t\t(ALWAYS RECLUSTER): {f1_not_needed_reclusterings}\n")
        f_output.write(f"\t\t\t(RECLUSTERING COEFFICIENT): {f2_not_needed_reclusterings}")
        f_output.write("\n\t\tReclusters necesarios no realizados:\n")
        f_output.write(f"\t\t\t(RECLUSTERING COEFFICIENT): {f2_not_detected_but_needed_reclusterings}")
        f_output.write(f"\n\n\tVeces que ALWAYS RCL CH_IDX > RCL COEFF CH_IDX: {times_f1_ch_index_is_better_than_f2_ch_index}\n")
        

# Provide the file paths to the JSON files you want to compare
pre_path = "/home/hugo/Documents/TFGs/"
file_path_1 = ['bicis_60mins_URcl0_Cls20_Dm0.1_Dc0.9_time20.58.json']# ['bicis_3mins_URcl0_Cls10_Dm0.1_Dc0.9_time112.34.json', 'bicis_3mins_URcl0_Cls20_Dm0.1_Dc0.9_time308.0.json', 'bicis_10mins_URcl0_Cls10_Dm0.1_Dc0.9_time40.16.json', 'bicis_10mins_URcl0_Cls20_Dm0.1_Dc0.9_time96.09.json', 'bicis_30mins_URcl0_Cls10_Dm0.1_Dc0.9_time15.29.json', 'bicis_30mins_URcl0_Cls20_Dm0.1_Dc0.9_time41.71.json', 'bicis_60mins_URcl0_Cls10_Dm0.1_Dc0.9_time6.84.json','bicis_60mins_URcl0_Cls20_Dm0.1_Dc0.9_time20.58.json']
file_path_2 = ['bicis_60mins_URcl0.5_Cls20_Dm0.1_Dc0.9_time6.97.json']# ['bicis_3mins_URcl0.1_Cls10_Dm0.1_Dc0.9_time67.75.json', 'bicis_3mins_URcl0.1_Cls20_Dm0.1_Dc0.9_time101.21.json', 'bicis_10mins_URcl0.1_Cls10_Dm0.1_Dc0.9_time26.32.json', 'bicis_10mins_URcl0.1_Cls20_Dm0.1_Dc0.9_time46.17.json', 'bicis_30mins_URcl0.1_Cls10_Dm0.1_Dc0.9_time12.3.json', 'bicis_30mins_URcl0.1_Cls20_Dm0.1_Dc0.9_time23.84.json', 'bicis_60mins_URcl0.1_Cls10_Dm0.1_Dc0.9_time4.83.json','bicis_60mins_URcl0.1_Cls20_Dm0.1_Dc0.9_time14.3.json']


for file_1, file_2 in list(zip(file_path_1, file_path_2)):
    compare_json_files(file_1, file_2)
