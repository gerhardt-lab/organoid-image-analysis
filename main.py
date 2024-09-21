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
from src.nuclei_quant import nuclei_quant
from src.device_mask import compute_device_regions
from src.connected_components import extract_connected_components


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
    
key_file = pd.read_csv(parameters["key_file"])
key_file.to_csv(output_folder + "key_file.csv")

key_file = compute_device_regions(parameters, key_file)
key_file.to_csv(output_folder + "key_file.csv")

results_px_df = pixel_quant(parameters, key_file)
results_px_df.to_csv(parameters["output_folder"] + "results_px.csv", index = False)

results_nuclei_df = pd.DataFrame()
results_nuclei_df = nuclei_quant(parameters, key_file)

extract_connected_components(parameters=parameters, key_file=key_file)


