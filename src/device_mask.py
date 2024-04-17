import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import regionprops
import skimage.measure as measure
from skimage import io


def compute_device_regions(parameters,key_file):


    for index, row in key_file.iterrows():

        filepath = parameters["input_folder"] + row["device_mask"]
        mask = io.imread(filepath)
        print(mask.shape)
        label_mask = measure.label(mask)
        
        #np.unique(mask)
        print("Number of regions:")
        print(np.unique(label_mask))

        left_post_x_centroids = []
        left_post_x_left_border = []
        left_post_x_right_border = []
        
        right_post_x_centroids = []
        right_post_x_right_border = []
        right_post_x_left_border = [] 

        for label in np.unique(label_mask):

            if label == 0:
                continue
                
            single_mask = np.where(label_mask== label, 1, 0)
            fig, ax = plt.subplots(figsize= (5,5))
            ax.imshow(single_mask)

            region = regionprops(single_mask)[0]
            
            #print(len(props))
            print(region.bbox)
            if region.centroid[1] < label_mask.shape[1]/2 and region.centroid[1] > label_mask.shape[1]*0.1:
                left_post_x_centroids.append(region.centroid[1])
                left_post_x_left_border.append(region.bbox[1])
                left_post_x_right_border.append(region.bbox[3])
            
                # to determine end of sprouting region:
                ## use centroids of objects that are between 50% and 85% of the image
            elif region.centroid[1] > label_mask.shape[1]/2: #and region.centroid[1] < label_mask.shape[1]*0.85:
                right_post_x_centroids.append(region.centroid[1])
                right_post_x_left_border.append(region.bbox[1])
                right_post_x_right_border.append(region.bbox[3])


        #print(left_post_x_centroids)
        #print(left_post_x_left_border)
        #print(right_post_x_centroids)
        #print(left_post_x_right_border)

        #sprouting_start = np.mean(left_post_x_centroids)
        #sprouting_end = np.mean(right_post_x_centroids)
        monolayer_end = np.mean(left_post_x_left_border)
        open_space_start = np.mean(left_post_x_right_border)
        open_space_end = np.mean(right_post_x_left_border)

        fig, ax = plt.subplots(figsize= (5,5))
        ax.imshow(label_mask)
        #ax.vlines(sprouting_start, label_mask.shape[0]*0.05, label_mask.shape[0]*0.95, color='White')
        #ax.vlines(sprouting_end, label_mask.shape[0]*0.05, label_mask.shape[0]*0.95, color='White')
        ax.vlines(monolayer_end, label_mask.shape[0]*0.05, label_mask.shape[0]*0.95, color='White')
        ax.vlines(open_space_start, label_mask.shape[0]*0.05, label_mask.shape[0]*0.95, color='White')
        ax.vlines(open_space_end, label_mask.shape[0]*0.05, label_mask.shape[0]*0.95, color='White')

        ax.text(monolayer_end*0.5,label_mask.shape[0]*0.5, "monolayer",color="white", rotation = "vertical")
        ax.text(open_space_start + monolayer_end*0.5,label_mask.shape[0]*0.5, "open space",color="white", rotation = "vertical")

        #ax.vlines(open_space_start + monolayer_end, label_mask.shape[0]*0.05, label_mask.shape[0]*0.95, color='White')
        #ax.text(open_space_start + monolayer_end*1.5,label_mask.shape[0]*0.5, "region B",color="white", rotation = "vertical")
        #ax.vlines(open_space_start + monolayer_end*2.0, label_mask.shape[0]*0.05, label_mask.shape[0]*0.95, color='White')

        plt.savefig(parameters["output_folder"] + row["filename"] + "device_regions.png")
        plt.close()

        key_file.at[index,"monolayer_end_px"] = monolayer_end
        key_file.at[index,"open_space_start_px"] = open_space_start
        key_file.at[index,"open_space_end_px"] = open_space_end

    return key_file

