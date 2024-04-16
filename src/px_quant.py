import pandas as pd
import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu


def pixel_quant(parameters):
    """
    Function to compute the pixel localization of the siRNA and vecad channels in the device.
    """

    # load key file
    key_file = pd.read_csv(parameters["key_file"])
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

            if binary_siRNA[y,x]:
                #pixe
                results_px_df.at[counter_px,"filename"] = row["filename"]
                results_px_df.at[counter_px,"condition"] = row["condition"]
                results_px_df.at[counter_px,"x"] = x
                results_px_df.at[counter_px,"y"] = y
                results_px_df.at[counter_px,"x_mum"] = x/parameters["pixel_to_micron_ratio"]
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
                results_px_df.at[counter_px,"y"] = y
                results_px_df.at[counter_px,"x_mum"] = x/parameters["pixel_to_micron_ratio"]
                results_px_df.at[counter_px,"y_mum"] = y/parameters["pixel_to_micron_ratio"]
                results_px_df.at[counter_px,"color"] = "green"
                results_px_df.at[counter_px,"siRNA"] = row["green"]

                sample_counter += 1
                counter_px += 1
            else:
                continue     


    return results_px_df
