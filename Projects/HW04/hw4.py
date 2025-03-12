import csv
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster import hierarchy


def load_data(filepath):
    output_list = []  # Initialize an empty list 
    

    with open(filepath, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader) #The first row: Descriptions
        
        for row in reader:
            row_dict = {}
            
            for i, value in enumerate(row):  #Ok, so python here provides pairs of (index, value)
                row_dict[headers[i]] = value
            
            output_list.append(row_dict)    
    return output_list     





def calc_features(row):
    keys = ['child_mort', 'exports', 'health', 'imports', 'income', 
            'inflation', 'life_expec', 'total_fer', 'gdpp']
    
    feature_vector = np.array([float(row[key]) for key in keys], dtype=np.float64)
    return feature_vector




def hac(features):
    num_countries = len(features)
    linkage_output = np.zeros((num_countries - 1, 4))


    #Initially, every country is its own cluster, n ones, but I add n - 1 itrations
    #to it, so it is 2n - 1
    cluster_sizes = np.ones(2 * num_countries - 1) 


    #Initialize the distance matrix: (2n-1)*(2n-1) instead of n*n right?
    distance_matrix  = np.full((2 * num_countries - 1 , 2 * num_countries - 1), np.inf)

    for i in range(num_countries):
        for j in range(i + 1, num_countries):
            distance_matrix[i, j] = distance_matrix[j, i] = np.linalg.norm(features[i] - features[j])



    # n -1 here means the total number of iterations (max depth discussed in lecture)
    for iteration in range(num_countries - 1):

        min_distance = np.inf
        min_i, min_j = -1, -1  #Indices of closest clusters

        for i in range( 2 * num_countries - 1):
            if cluster_sizes[i] == 0:    
                continue    #Already merged, skip

            for j in range(i + 1, 2 * num_countries - 1):
              if cluster_sizes[j] == 0:   #Already merged, skip
                continue
              #print("type: ", distance_matrix[i, j], "max: ", max(distance_matrix[i,j]))
              current = np.max(distance_matrix[i, j])
              if current < min_distance:
                min_distance = current
                min_i, min_j = i, j      


        linkage_output[iteration, 0] = min_i
        linkage_output[iteration, 1] = min_j
        linkage_output[iteration, 2] = min_distance
        linkage_output[iteration, 3] = cluster_sizes[min_i] + cluster_sizes[min_j]   



        #Also need to set back to 0 to those clusters:
        cluster_sizes[min_i] = 0
        cluster_sizes[min_j] = 0
        cluster_sizes[num_countries + iteration] = linkage_output[iteration, 3]

        # Update distance matrix
        for k in range(num_countries + iteration):
            if cluster_sizes[k] == 0:
                continue
            distance_matrix[k, num_countries + iteration] = max(distance_matrix[k, min_i], distance_matrix[k, min_j])
            distance_matrix[num_countries + iteration, k] = distance_matrix[k, num_countries + iteration]


    return linkage_output





def fig_hac(Z, names):
    fig = plt.figure()
    hierarchy.dendrogram(
        Z,
        labels=names,  # Use country names as labels
        leaf_rotation=90,  # Rotate labels for better readability
    )

    plt.tight_layout()
    return fig

     

def normalize_features(features):

    features_array = np.array(features)
    mean = np.mean(features_array, axis=0)  
    std = np.std(features_array, axis=0)  

    normalized = (features_array - mean) / std
    normalized = [np.array(row, dtype=np.float64) for row in normalized]

    return normalized











# # Run the main method if this script is executed
if __name__ == "__main__":
    data = load_data("Country-data.csv")
    features = [calc_features(row) for row in data]
    names = [row["country"] for row in data]
    features_normalized = normalize_features(features)
    np.savetxt("output.txt", features_normalized)
    n = 20
    Z = hac(features[:n])
    fig = fig_hac(Z, names[:n])
    plt.show()





