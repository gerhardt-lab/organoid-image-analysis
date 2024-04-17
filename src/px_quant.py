import pandas as pd
import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import seaborn as sns


def pixel_quant(parameters, key_file):
    """
    Function to compute the pixel localization of the siRNA and vecad channels in the device.
    """

    # create empty dataframe to store results
    results_px_df = pd.DataFrame()

    counter_px = 0
    for index, row in key_file.iterrows():
        
        print(row)
        filepath = parameters["input_folder"] + row["filename"]
        img = io.imread(filepath)
        print(img.shape)

        thresh_vecad = threshold_otsu(img[:,:,parameters["channel_EC_junction"]])
        binary_vecad = img[:,:,parameters["channel_EC_junction"]] > thresh_vecad

        thresh_siRNA = threshold_otsu(img[:,:,parameters["channel_siRNA"]])
        binary_siRNA = img[:,:,parameters["channel_siRNA"]] > thresh_siRNA

        n_sample = parameters["n_sample"]   # number of pixel samples to take per image
        sample_counter = 0                  # counter to keep track of the number of pixel samples taken
        while sample_counter < n_sample:
            x = np.random.randint(0, img.shape[1])
            y = np.random.randint(0, img.shape[0])
            x_adjusted = x - row["monolayer_end_px"]

            if binary_siRNA[y,x]:
                # align measurements with respect to the device mask with origin at the left border 
                # of the device meaing the end of the cell monolayer
                
                results_px_df.at[counter_px,"filename"] = row["filename"]
                results_px_df.at[counter_px,"condition"] = row["condition"]
                results_px_df.at[counter_px,"x"] = x
                results_px_df.at[counter_px,"x_adjusted"] = x_adjusted
                results_px_df.at[counter_px,"y"] = y
                results_px_df.at[counter_px,"x_adjusted_mum"] = x_adjusted/parameters["pixel_to_micron_ratio"]
                results_px_df.at[counter_px,"y_mum"] = y/parameters["pixel_to_micron_ratio"]
                results_px_df.at[counter_px,"color"] = "orange"
                results_px_df.at[counter_px,"siRNA"] = row["orange"]
                
                sample_counter += 1
                counter_px += 1

            elif binary_vecad[y,x]:
                # pixel that is not siRNA but is vecad
                results_px_df.at[counter_px,"filename"] = row["filename"]
                results_px_df.at[counter_px,"condition"] = row["condition"]
                results_px_df.at[counter_px,"x"] = x
                results_px_df.at[counter_px,"x_adjusted"] = x_adjusted
                results_px_df.at[counter_px,"y"] = y
                results_px_df.at[counter_px,"x_adjusted_mum"] = x_adjusted/parameters["pixel_to_micron_ratio"]
                results_px_df.at[counter_px,"y_mum"] = y/parameters["pixel_to_micron_ratio"]
                results_px_df.at[counter_px,"color"] = "green"
                results_px_df.at[counter_px,"siRNA"] = row["green"]

                sample_counter += 1
                counter_px += 1

            else:
                continue     

        plot_df = results_px_df[results_px_df["filename"] == row["filename"]]
        fig, ax = plt.subplots(figsize = (9,6))

        sns.kdeplot(data=plot_df, x="x_adjusted_mum", hue="color", common_norm=False, 
            fill = True, ax = ax, palette = {"orange" :"orange", "green": "green"})
        
        monolayer_end_um = 0.0 # row["monolayer_end_px"]/parameters["pixel_to_micron_ratio"]
        open_space_start_um = (row["open_space_start_px"] - row["monolayer_end_px"])/parameters["pixel_to_micron_ratio"]
        open_space_end_um = (row["open_space_end_px"]- row["monolayer_end_px"])/parameters["pixel_to_micron_ratio"]

        ax.axvline(monolayer_end_um,  color='black')
        ax.axvline(open_space_start_um, color='black')
        ax.axvline(open_space_end_um, color='black')
        
        #plt.savefig(parameters["output_folder"] + row["filename"] + "-px-kde.pdf")
        plt.savefig(parameters["output_folder"] + row["filename"] + "-px-kde.png")
        plt.close()

    return results_px_df
