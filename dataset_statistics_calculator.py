import json
"""
    'time': str(global_time),
    'reclustered': True,
    'clusters': optimal_clusters,
    'centroids': optimal_centroids,
    'reassigned': reassigned_points,
    'post_reassigned': [[], []],
    'old_coords': (x, y),
    'occupied': [[], []]
"""
def compare_json_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        json1 = json.load(f1)
        json2 = json.load(f2)

    both_reclustered = lambda x, y: "Sí" if x and y else "No"

    with open(f"bicis_3mins_URcl0_Cls10_Dm0.1_Dc0.9+bicis_3mins_URcl0.1_Cls10_Dm0.1_Dc0.9_COMPARISON.txt", "w") as f_output:
        for f1_entry, f2_entry in zip(json1, json2):
            f_output.write(str(f1_entry['time']))
            f_output.write(" {\n\t")
            f_output.write("Ambos clusterizaron: ")
            f_output.write(both_reclustered(f1_entry["reclustered"], f2_entry["reclustered"]))
            f_output.write("\n\t")
            f_output.write(f"\tNumero de clusters (ALWAYS RECLUSTER): {len(f1_entry['clusters'])}\n\t")
            f_output.write(f"\tNumero de clusters (RECLUSTERING COEFICIENT): {len(f2_entry['clusters'])}\n")
            f_output.write("}\n")

# Provide the file paths to the JSON files you want to compare
file_path_1 = '/home/hugo/Documents/TFGs/bicis_3mins_URcl0_Cls10_Dm0.1_Dc0.9_time136.34.json'
file_path_2 = '/home/hugo/Documents/TFGs/bicis_3mins_URcl0.1_Cls10_Dm0.1_Dc0.9_time65.05.json'

compare_json_files(file_path_1, file_path_2)
