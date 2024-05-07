import pandas as pd
import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu, gaussian
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import regionprops
from skimage.measure import label


from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
import networkx as nx


def extract_connected_components(parameters, key_file):

    output_folder = parameters["output_folder"]
    distance_px = parameters["distance_px"]
    nuclei_data = pd.read_csv(output_folder + "results_nuclei.csv")

    # create empty dataframe to store results

    print(nuclei_data.head())

    for filename in nuclei_data["filename"].unique():

        print(filename)
        data = nuclei_data[nuclei_data["filename"] == filename]
        #img_nuc = io.imread(output_folder + filename)
        #img_nuc = normalize(img_nuc, 1, 99.8, axis=(0, 1))

        # Create an empty graph
        G = nx.Graph()

        # Add nodes to the graph
        for index, row in data.iterrows():
            G.add_node(index, pos=(row['x'], row['y']))

        # Add edges based on some condition (e.g., distance between nodes)
        for node1 in G.nodes(data=True):
            #print(node1)
            for node2 in G.nodes(data=True):
                if node1 != node2:
                    # Calculate Euclidean distance
                    dist = np.sqrt((node1[1]['pos'][0] - node2[1]['pos'][0])**2 + (node1[1]['pos'][1] - node2[1]['pos'][1])**2)
                    # Add edge if distance is less than some threshold
                    if dist < distance_px:
                        G.add_edge(node1[0], node2[0])


        # Compute the number of nodes
        num_nodes = G.number_of_nodes()

        # Compute the number of edges
        num_edges = G.number_of_edges()

        print(f"The graph has {num_nodes} nodes and {num_edges} edges.")

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

        #ax.imshow(img_nuc, cmap="gray")
        # Get node positions from the graph
        pos = nx.get_node_attributes(G, 'pos')

        # Draw the graph
        nx.draw(G, pos, node_size=10, node_color = 'r', edge_color='k', with_labels=False, ax=ax)

        plt.savefig(output_folder + filename + "_connected_components.png")
        # Show the plot
        plt.show()


    return nuclei_data

