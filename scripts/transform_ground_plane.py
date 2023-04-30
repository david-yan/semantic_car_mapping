import numpy as np
import cv2
import pickle
import argparse
import pandas as pd

# Define the ground plane coefficients
a, b, c, d = -0.0908798, -0.137995, 0.961871, 1.53492

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Input pickle file')
args = parser.parse_args()

# Load the pickle file into a DataFrame
with open(args.input, 'rb') as f:
    data = pickle.load(f)
    
all_pts = data['scan']
all_cuboids = data['cuboids']

# TODO: go thru all_pts, apply the ground plane transform properly
# the coefficients above come from 0_10.pkl

# Compute the normal vector of the ground plane
normal = np.array([a, b, c])

# Compute the distance of the ground plane from the origin
distance = abs(d) / np.linalg.norm(normal)

# Create a translation matrix to move the point cloud back to z=0
translation = np.identity(4)
translation[2, 3] = distance

# Create a rotation matrix to align the ground plane with the x-y plane
x_axis = np.array([1, 0, 0])
rotation_axis = np.cross(normal, x_axis)
rotation_axis /= np.linalg.norm(rotation_axis)
angle = np.arccos(np.dot(normal, x_axis) / (np.linalg.norm(normal) * np.linalg.norm(x_axis)))
rotation = np.identity(4)
rotation[:3, :3] = cv2.Rodrigues(rotation_axis * angle)[0]

# Combine the translation and rotation matrices
transform = translation.dot(rotation)

# Apply the transformation to the point cloud
pts_transformed = np.hstack([pts, np.ones((pts.shape[0], 1))]).dot(transform.T)[:, :3]
