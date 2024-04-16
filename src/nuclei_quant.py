import pandas as pd
import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu, gaussian
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.measure import label


from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize




def nuclei_quant(parameters):
    
    print("Nuclei segmentation")
    # load key file
    key_file = pd.read_csv(parameters["key_file"])
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    results_df = pd.DataFrame()

    counter = 0

    for index, row in key_file.iterrows():

        filepath = parameters["input_folder"] + row["filename"]
        img = io.imread(filepath)
        print(img.shape)

        vertical_px = img.shape[0]
        vertical_px_tile = vertical_px // parameters["number_of_vertical_tiles"]

        for k in range(parameters["number_of_vertical_tiles"]):
            start_px = k*vertical_px_tile 
            end_px = start_px + vertical_px_tile
            img_tile = img[start_px:end_px,:,:]
            #img_tile = img_tile[start_px:,:,:]
            print(k, img_tile.shape)

            # plot original image and stardist normalized image -------------------

            img_nuclei = img_tile[:,:,0]
            fig, ax = plt.subplots()
            ax.imshow(img_nuclei, cmap="Grays")
            ax.axis("off")
            filepath_out = parameters["output_folder"] + row["filename"] + "-" + str(k) + ".png"
            plt.savefig(filepath_out)
            plt.close()

            fig, ax = plt.subplots()
            ax.imshow(normalize(img_nuclei), cmap="Grays")
            ax.axis("off")
            filepath_out = parameters["output_folder"] + row["filename"] + "-" + str(k) + "-normalized.png"
            plt.savefig(filepath_out)
            plt.close()

            # nuclei segmentation ------------------------------------------------

            img_labels, details = model.predict_instances(normalize(img_nuclei), n_tiles = model._guess_n_tiles(img_nuclei))
            
            fig, ax = plt.subplots(figsize= (10,10))
            ax.imshow(img_labels)
            ax.axis("off")
            filepath_out = parameters["output_folder"] + row["filename"] + "-" + str(k) + "-labels.png"
            plt.savefig(filepath_out)
            plt.close()

            img_EC_vecad = gaussian(img_tile[:,:,parameters["channel_EC_junction"]], sigma = parameters["gaussian_sigma"])
            thresh_vecad = threshold_otsu(img_EC_vecad)
            binary_vecad = img_EC_vecad > thresh_vecad

            fig, ax = plt.subplots(figsize= (10,10))
            ax.imshow(binary_vecad, cmap="Greens")
            ax.axis("off")
            ax.set_title("%s cells" % row["green"])
            fig.savefig(parameters["output_folder"] + row["filename"] + "-" + str(k) + "-ve_cad.png")
            plt.close()

            img_siRNA = gaussian(img_tile[:,:,parameters["channel_siRNA"]], sigma = parameters["gaussian_sigma"])
            thresh_orange = threshold_otsu(img_siRNA)
            binary_orange = img_siRNA > thresh_orange

            fig, ax = plt.subplots(figsize= (10,10))
            ax.imshow(binary_orange, cmap="Oranges")
            ax.axis("off")
            ax.set_title("%s cells" % row["orange"])
            fig.savefig(parameters["output_folder"] + row["filename"] + "-" + str(k) + "-orange.png")
            plt.close()

            binary_green = np.logical_and(np.logical_not(binary_orange), binary_vecad)
            print(binary_green.shape)
            EC_nuclei_labels_siRNA_green = np.unique(binary_green*img_labels) # all labels ECs in the image
            EC_nuclei_labels_siRNA_orange = np.unique(binary_orange*img_labels) # all labels ECs in the image
            EC_nuclei_labels = np.unique(binary_vecad*img_labels) # all labels ECs in the image

            EC_nuclei = np.zeros_like(img_labels)
            EC_nuclei_siRNA_green = np.zeros_like(img_labels)
            EC_nuclei_siRNA_orange = np.zeros_like(img_labels)

            for label in EC_nuclei_labels:
                if label == 0:
                    continue
                single_nucleus = np.where(img_labels == label, 1, 0)
                EC_nuclei += single_nucleus

                regions = regionprops(single_nucleus)
                for props in regions:
                    y0, x0 = props.centroid
                    print(x0,y0)
            
                results_df.at[counter,"label"] = label
                results_df.at[counter,"x"] = x0
                results_df.at[counter,"y"] = y0
                results_df.at[counter,"x_mum"] = x0/parameters["pixel_to_micron_ratio"]
                results_df.at[counter,"y_mum"] = y0/parameters["pixel_to_micron_ratio"]
                if label in np.unique(EC_nuclei_labels_siRNA_orange):
                    results_df.at[counter,"color"] = "orange"
                    results_df.at[counter,"siRNA"] = row["orange"]
                    EC_nuclei_siRNA_orange += single_nucleus
                else:
                    results_df.at[counter,"color"] = "green"
                    results_df.at[counter,"siRNA"] = row["green"]
                    EC_nuclei_siRNA_green += single_nucleus
                results_df.at[counter,"filename"] = row["filename"]
                results_df.at[counter,"condition"] = row["condition"]
                
                results_df.at[counter,"tile"] = k 
                counter += 1

            
            fig, ax = plt.subplots(figsize= (10,10))
            ax.imshow(EC_nuclei , cmap="Greens")
            ax.axis("off")
            ax.set_title("EC nuclei")
            fig.savefig(parameters["output_folder"] + row["filename"] + "-" + str(k) + "-nuclei_vecad.png")
            plt.close()

            fig, ax = plt.subplots(figsize= (10,10))
            ax.imshow(EC_nuclei_siRNA_green, cmap="Greens")
            ax.axis("off")
            ax.set_title("nuclei , %s cells" % row["green"])
            fig.savefig(parameters["output_folder"] + row["filename"] + "-" + str(k) + "-nuclei_green.png")
            plt.close()

            fig, ax = plt.subplots(figsize= (10,10))
            ax.imshow(EC_nuclei_siRNA_orange , cmap="Oranges")
            ax.axis("off")
            ax.set_title("nuclei , %s cells" % row["orange"])
            fig.savefig(parameters["output_folder"] + row["filename"] + "-" + str(k) + "-nuclei_orange.png")
            plt.close()

            results_df.to_csv(parameters["output_folder"] + "results_nuclei.csv", index = False)
    
    return results_df

    