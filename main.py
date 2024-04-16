from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize

import skimage.io as io
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, regionprops_table

import matplotlib.pyplot as plt
import argparse
import sys
import yaml
import pandas as pd
import numpy as np

from utils.io import read_parameters
from src.px_quant import pixel_quant


#########################
# read parameters
#########################

parser = argparse.ArgumentParser(
    description='This script runs ec movement trajectory analysis. '
                'You need to provide a parameter file and optionally '
                'an input folder, a key file, and an output folder.',
    epilog='Example usage: python run.py parameters.yaml --input_folder /path/to/input '
           '--key_file /path/to/key --output_folder /path/to/output',
)
#parser.add_argument('param', type=str, help='Path to the parameter file.')
parser.add_argument('param', type=str, help='Path to the parameter file.')
parser.add_argument('--input_folder', type=str, help='Path to the input folder.')
parser.add_argument('--key_file', type=str, help='Path to the key file.')
parser.add_argument('--output_folder', type=str, help='Path to the output folder.')

if len(sys.argv) < 4:
    #parser.print_usage()
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
print("-------")
print("reading parameters from: ", args.param)
print("-------")

parameter_file  = args.param
parameters = read_parameters(parameter_file)               

# Override parameters with command line arguments if provided
if args.input_folder:
    parameters["input_folder"] = args.input_folder
if args.output_folder:
    parameters["output_folder"] = args.output_folder
if args.key_file:
    parameters["key_file"] = args.key_file

print("-------")
print("used parameter values: ")
print(parameters)
print("-------")

output_folder = parameters["output_folder"]
# save parameter file to output
with open(output_folder + "/parameters.yml", 'w') as outfile:
    yaml.dump(parameters, outfile)


### stardist segmentation
    

model = StarDist2D.from_pretrained('2D_versatile_fluo')


key_file = pd.read_csv(parameters["key_file"])
key_file.to_csv(output_folder + "key_file.csv")
results_df = pd.DataFrame()
results_px_df = pd.DataFrame()


results_px_df = pixel_quant(parameters)
results_px_df.to_csv(parameters["output_folder"] + "results_px.csv", index = False)

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

        img_labels, details = model.predict_instances(normalize(img_nuclei), n_tiles = model._guess_n_tiles(img_nuclei))
        
        fig, ax = plt.subplots(figsize= (10,10))
        ax.imshow(img_labels)
        ax.axis("off")
        filepath_out = parameters["output_folder"] + row["filename"] + "-" + str(k) + "-lables.png"
        plt.savefig(filepath_out)
        plt.close()

        thresh_vecad = threshold_otsu(img_tile[:,:,parameters["channel_EC_junction"]])
        binary_vecad = img_tile[:,:,parameters["channel_EC_junction"]] > thresh_vecad

        fig, ax = plt.subplots(figsize= (10,10))
        ax.imshow(binary_vecad, cmap="Greens")
        ax.axis("off")
        ax.set_title("%s cells" % row["green"])
        fig.savefig(parameters["output_folder"] + row["filename"] + "-" + str(k) + "-ve_cad.png")
        plt.close()

        thresh_orange = threshold_otsu(img_tile[:,:,parameters["channel_siRNA"]])
        binary_orange = img_tile[:,:,parameters["channel_siRNA"]] > thresh_orange

        fig, ax = plt.subplots(figsize= (10,10))
        ax.imshow(binary_orange, cmap="Oranges")
        ax.axis("off")
        ax.set_title("%s cells" % row["orange"])
        fig.savefig(parameters["output_folder"] + row["filename"] + "-" + str(k) + "-orange.png")
        plt.close()

        binary_green = np.logical_and(np.logical_not(binary_orange), binary_vecad)
        print(binary_green.shape)
        siScr_green = binary_green*img_labels

        binary_green = np.logical_and(np.logical_not(binary_orange), binary_vecad)
        print(binary_green.shape)
        siScr_orange = binary_orange*img_labels

        all_ECs = binary_vecad*img_labels
        
        fig, ax = plt.subplots(figsize= (10,10))
        ax.imshow(siScr_green , cmap="Greens")
        ax.axis("off")
        ax.set_title("nuclei , %s cells" % row["green"])
        fig.savefig(parameters["output_folder"] + row["filename"] + "-" + str(k) + "-nuclei_green.png")
        plt.close()

        fig, ax = plt.subplots(figsize= (10,10))
        ax.imshow(siScr_orange , cmap="Oranges")
        ax.axis("off")
        ax.set_title("nuclei , %s cells" % row["orange"])
        fig.savefig(parameters["output_folder"] + row["filename"] + "-" + str(k) + "-nuclei_orange.png")
        plt.close()

        for label in np.unique(all_ECs):
            single_nucleus = np.where(img_labels == label, 1, 0)
            regions = regionprops(single_nucleus)
            for props in regions:
                y0, x0 = props.centroid
                print(x0,y0)
        
            results_df.at[counter,"label"] = label
            results_df.at[counter,"x"] = x0
            results_df.at[counter,"y"] = y0
            results_df.at[counter,"x_mum"] = x0/parameters["pixel_to_micron_ratio"]
            results_df.at[counter,"y_mum"] = y0/parameters["pixel_to_micron_ratio"]
            if label in np.unique(siScr_orange):
                results_df.at[counter,"color"] = "orange"
                results_df.at[counter,"siRNA"] = row["orange"]
            elif label in np.unique(siScr_green):
                results_df.at[counter,"color"] = "green"
                results_df.at[counter,"siRNA"] = row["green"]
            results_df.at[counter,"filename"] = row["filename"]
            results_df.at[counter,"condition"] = row["condition"]
            
            results_df.at[counter,"tile"] = k 
            counter += 1

        results_df.to_csv(parameters["output_folder"] + "results.csv", index = False)



img = test_image_nuclei_2d()
labels, _ = model.predict_instances(normalize(img))
