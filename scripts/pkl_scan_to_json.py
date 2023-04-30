import json
import pickle
import argparse
import pandas as pd

# First convert 0_10.pkl to 0_10_processed.pkl, then run this script
# Sample usage: 
# python scripts/process_data.py --input data/0_10.pkl
# python scripts/pkl_scan_to_json.py --input data/0_10_processed.pkl

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Input pickle file')
args = parser.parse_args()

# Load the pickle file into a DataFrame
with open(args.input, 'rb') as f:
    data = pickle.load(f)
    
pointcloud_data = data['scan']

# Convert the pointcloud data into a JSON array
json_data = []
for timestep in pointcloud_data:
    json_timestep = []
    for point in timestep:
        json_timestep.append(list(point))
    json_data.append(json_timestep)

# Save the JSON object to a file
output_file = args.input.replace('.pkl', '.json')
with open(output_file, 'w') as f:
    json.dump(json_data, f)
