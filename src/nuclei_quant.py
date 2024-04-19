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




def nuclei_quant(parameters, key_file):
    
    print("Nuclei segmentation")
    # load key file
    #key_file = pd.read_csv(parameters["key_file"])
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    results_df = pd.DataFrame()

    counter = 0

    for index, row in key_file.iterrows():

        monolayer_end_px = row["monolayer_end_px"]
        open_space_start_px = row["open_space_start_px"]
        open_space_end_px = row["open_space_end_px"]
        monolayer_end_um = int(monolayer_end_px/parameters["pixel_to_micron_ratio"])
        open_space_start_um = int((open_space_start_px - monolayer_end_px)/parameters["pixel_to_micron_ratio"])
        open_space_end_um = int((open_space_end_px - monolayer_end_px)/parameters["pixel_to_micron_ratio"])    

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

            img_nuclei = img_tile[:,:,parameters["channel_nuclei"]]
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

            img_EC_vecad = gaussian(img_tile[:,:,parameters["channel_EC_junction"]], 
                                    sigma = parameters["gaussian_sigma"]["EC_junction"])
            thresh_vecad = threshold_otsu(img_EC_vecad)
            binary_vecad = img_EC_vecad > thresh_vecad

            fig, ax = plt.subplots(figsize= (10,10))
            ax.imshow(binary_vecad, cmap="Greens")
            ax.axis("off")
            ax.set_title("junction channel")
            fig.savefig(parameters["output_folder"] + row["filename"] + "-" + str(k) + "-ve_cad.png")
            plt.close()

            img_siRNA = gaussian(img_tile[:,:,parameters["channel_siRNA"]], 
                                 sigma = parameters["gaussian_sigma"]["siRNA"])
            thresh_orange = threshold_otsu(img_siRNA)
            binary_orange = img_siRNA > thresh_orange

            fig, ax = plt.subplots(figsize= (10,10))
            ax.imshow(binary_orange, cmap="Oranges")
            ax.axis("off")
            ax.set_title("mosaic label for %s" % row["orange"])
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

                regions = regionprops(single_nucleus, intensity_image=img_EC_vecad)
                for props in regions:
                    y0, x0 = props.centroid
                    area = props.area
                    y0 += start_px 
                    mean_intensity_vecad = props.intensity_mean
                    print("x: ",x0, " y: ", y0, " area: ", area, " mean intensity: ", mean_intensity_vecad)

                if parameters["filter_nuclei"]:
                    if area < parameters["nuclei_min_area"]:
                        print("Nuclei label mask below min area threshold")
                        continue
                    if area > parameters["nuclei_max_area"]:
                        print("Nuclei label mask above max area threshold")
                        continue
                    if mean_intensity_vecad < parameters["nuclei_min_intensity"]:
                        print("Nuclei label mask below min intensity threshold")
                        continue

                x_adjusted = x0 - row["monolayer_end_px"]
                results_df.at[counter,"label"] = label
                results_df.at[counter,"x"] = x0
                results_df.at[counter,"x_adjusted"] = x_adjusted
                results_df.at[counter,"y"] = y0
                results_df.at[counter,"x_adjusted_mum"] = x_adjusted/parameters["pixel_to_micron_ratio"]
                results_df.at[counter,"y_mum"] = y0/parameters["pixel_to_micron_ratio"]
                results_df.at[counter,"nuclei_area"] = area
                results_df.at[counter,"mean_intensity_vecad"] = mean_intensity_vecad
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
    
        plot_df = results_df[results_df["filename"] == row["filename"]]
        fig, ax = plt.subplots(figsize = (9,6))

        sns.histplot(data=plot_df, x="x_adjusted_mum", hue="color", common_norm=False, binwidth = 100.0,
        fill = True, ax = ax, palette = {"orange" :"orange", "green": "green"})
            
        monolayer_end_um = 0.0 # row["monolayer_end_px"]/parameters["pixel_to_micron_ratio"]
        open_space_start_um = (row["open_space_start_px"] - row["monolayer_end_px"])/parameters["pixel_to_micron_ratio"]
        open_space_end_um = (row["open_space_end_px"]- row["monolayer_end_px"])/parameters["pixel_to_micron_ratio"]

        ax.axvline(monolayer_end_um,  color='black')
        ax.axvline(open_space_start_um, color='black')
        ax.axvline(open_space_end_um, color='black')
            
        #plt.savefig(parameters["output_folder"] + row["filename"] + "-kde.pdf")
        plt.savefig(parameters["output_folder"] + row["filename"] + "-nuclei-kde-with-monolayer.png")
        plt.close()

        plot_df = plot_df[plot_df["x_adjusted_mum"] > 0.0]
        plot_df = plot_df[plot_df["x_adjusted_mum"] < open_space_end_um]       
        fig, ax = plt.subplots(figsize = (9,6))

        sns.histplot(data=plot_df, x="x_adjusted_mum", hue="color", common_norm=False, binwidth = 100.0,
                fill = True, ax = ax, palette = {"orange" :"orange", "green": "green"})
        #ax.axvline(monolayer_end_um,  color='black')
        ax.axvline(open_space_start_um, color='black')
        #ax.axvline(open_space_end_um, color='black')
            
        #plt.savefig(parameters["output_folder"] + row["filename"] + "-kde.pdf")
        plt.savefig(parameters["output_folder"] + row["filename"] + "-nuclei-kde.png")
        plt.close()

    return results_df

    
