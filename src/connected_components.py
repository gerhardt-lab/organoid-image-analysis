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

from griottes.graphmaker import graph_generation_func
from griottes import get_cell_properties, generate_delaunay_graph, generate_geometric_graph, plot_2D
from griottes.graphplotter import graph_plot
import networkx as nx


def generate_graph(parameters, key_file):

    min_size_component = parameters["min_size_component"]

    output_folder = parameters["output_folder"]
    distance_px = parameters["distance_px"]
    nuclei_data = pd.read_csv(output_folder + "results_nuclei.csv")
    

    #descriptors = ["label", "x", "y"]
    descriptors = nuclei_data.columns
    data_connected = pd.DataFrame(columns = descriptors)
    index = 0

    for filename in nuclei_data["filename"].unique():

        print("*"*50)
        print(filename)

        img_path = parameters["input_folder"] + filename
        img = io.imread(img_path)
        img_vecad = img[:,:,parameters["channel_EC_junction"]]
        img_nuclei = img[:,:,parameters["channel_nuclei"]]

        data = nuclei_data[nuclei_data["filename"] == filename]
        G = generate_delaunay_graph(data[descriptors],
                                descriptors = descriptors,
                                distance=distance_px,
                                image_is_2D = True)
    
        #graph_plot.network_plot_2D(G,
        #        figsize = (15,15),
        #        alpha_line = 1,
        #        scatterpoint_size = 2,
        #        #background_image = img_vecad,
        #        #weights = False,
        #        edge_color = 'yellow',
        #        line_factor = 0.15)

        plt.show()

        for component in list(nx.connected_components(G)):
            if len(component)< min_size_component:
                for node in component:
                    G.remove_node(node)

        #graph_plot.network_plot_2D(G,
        #        figsize = (15,15),
        #        alpha_line = 1,
        #        scatterpoint_size = 2,
        #        #background_image = img_vecad,
        #        #weights = False,
        #        edge_color = 'yellow',
        #        line_factor = 0.15)
        

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))


        #graph_plot.network_plot_2D(G,
        #        figsize = (15,15),
        #        alpha_line = 1,
        #        scatterpoint_size = 2,
                #background_image = img_vecad,
        #        #weights = False,
        #        edge_color = 'k',
        #        line_factor = 0.15)

        #ax.imshow(img_vecad, cmap="gray")
        ax.imshow(img_nuclei.T, cmap="gray")
        ax.invert_yaxis()
        start_x = data["monolayer_end_px"].values[0] - parameters["monolayer_width"]*parameters["pixel_to_micron_ratio"]

        ax.set_ylim(start_x, img_vecad.shape[1])

        ax.axhline(y=data["monolayer_end_px"].values[0], color='r', linestyle='--')
        #ax.axhline(y=0, color='r', linestyle='--')

        #ax.set_xlim(0, img_vecad.shape[0])
        #ax.set_ylim(0, img_vecad.shape[1])

        # Get node positions from the graph
        pos = nx.get_node_attributes(G, 'pos')
        # Draw the graph
        nx.draw(G, pos, node_size=2,edge_color='w')
        for node in G.nodes(data=True):
            x = node[1]["x"]
            y = node[1]["y"]
            color = node[1]["color"]
            if color == "orange":
                color = "violet"
            if color == "green":
                color = "limegreen"
            print(f"Node {node} has neighbors:")

            tip_cell = False
            monolayer_cell = True
            device_region = "monolayer" 
            if x > node[1]["monolayer_end_px"]:
                device_region = "sprout"
                tip_cell = True
                monolayer_cell = False
                neighbors = G[node[0]]
                # Count the neighbors with color red
                for neighbor in neighbors:
                    #print(f"  Neighbor: {neighbor}")
                    x_neighbor = G.nodes[neighbor]["x"]
                    y_neighbor = G.nodes[neighbor]["y"]
                    if x <= x_neighbor:
                        print("not a tip cell")
                        tip_cell = False
                        #device_region = "sprout"
            
            for descriptor in descriptors:
                data_connected.at[index, descriptor] = node[1][descriptor]
            data_connected.at[index, "tip_position"] = tip_cell
            data_connected.at[index, "monolayer_cell"] = monolayer_cell        
            #data_connected.at[index, "filename"] = filename
            if tip_cell:
                device_region = "tip"
            data_connected.at[index, "device_region"] = device_region   

            index += 1

                
            #red_count = sum(1 for neighbor in neighbors if G_delaunay.nodes[neighbor].get('color') == 'red')
            #orange_count = sum(1 for neighbor in neighbors if G_delaunay.nodes[neighbor].get('color') == 'orange')
            #green_count = sum(1 for neighbor in neighbors if G_delaunay.nodes[neighbor].get('color') == 'green')
        
            #for neighbor in G.neighbors(node):
            #    print(f"  Neighbor: {neighbor}")

            ax.plot(y, x, color = color, marker='o', markersize=3)
            if tip_cell:
                ax.plot(y, x, color = "red", marker='x', markersize=10)
            #ax.plot(x, y, color = color, marker='o', markersize=2)
        
        data_connected.to_csv(output_folder + "results_connected.csv", index = False)
        plt.savefig(output_folder + filename + "_connected_components.png")
        plt.savefig(output_folder + filename + "_connected_components.pdf")
        plt.close()        


#plt.tight_layout()
#        graph_plot(G_delaunay, output_folder + "delaunay_graph.png")
#    
#        plt.savefig(output_folder + filename + "_connected_components.png")
#        # Show the plot
#        plt.show()


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

