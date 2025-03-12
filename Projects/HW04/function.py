import geopandas
import numpy as np
from matplotlib import pyplot as plt

def world_map(Z, names, K_clusters):
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    world['name'] = world['name'].str.strip()
    names = [name.strip() for name in names]

    world['cluster'] = np.nan

    n = len(names)
    clusters = {j: [j] for j in range(n)}

    for step in range(n-K_clusters):
        cluster1 = Z[step][0]
        cluster2 = Z[step][1]

        # Create new cluster id as n + step
        new_cluster_id = n + step

        # Merge clusters
        clusters[new_cluster_id] = clusters.pop(cluster1) + clusters.pop(cluster2)

    # Assign cluster labels to countries in the world dataset
    for i, value in enumerate(clusters.values()):
        for val in value:
            world.loc[world['name'] == names[val], 'cluster'] = i

    # Plot the map
    world.plot(column='cluster', legend=True, figsize=(15, 10), missing_kwds={
        "color": "lightgrey",  # Set the color of countries without clusters
        "label": "Other countries"
    })

    # Show the plot
    plt.show()