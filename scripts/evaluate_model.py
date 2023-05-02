import argparse
import rospy
import ros_numpy
import json
import os
import tf
import numpy as np
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import pandas as pd
import pickle

from sensor_msgs.msg import PointCloud2
from interactive_boxes import InteractiveCuboids

from sklearn.metrics.pairwise import cosine_similarity
import pdb

# Parse input arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--idx', type=int, required=True, help='Frame number')
# parser.add_argument('--input', type=str, required=False, help='Input pickle file')
parser.add_argument('--path', type=str, required=True, help='Directory path of pickle files')
args = parser.parse_args()

files = []
for file in os.listdir(args.path):
    if file.endswith(".pkl"):
        files.append(os.path.join(args.path, file))

total_pred = 0
total_true = 0
total_gt = 0
for file in files:

    # Load the pickle file into a DataFrame
    with open(file, 'rb') as f:
        df = pickle.load(f)

    # import pdb; pdb.set_trace()
    for idx in range(len(df)):

        # get the cuboids from the model
        gt_cuboids_cam = df['gt_cuboids'][idx]
        pred_cuboids_cam = df['cuboids'][idx]


        # print out evaluation
        gt = gt_cuboids_cam[:,[0,1,2,-1]]
        pred = pred_cuboids_cam[:,[0,1,2,-1]]
        pred[:,-1] = np.where(pred[:, -1] > 0, pred[:, -1] - np.pi, pred[:, -1])

        # use cosine similarity to figure out which is closest gt cuboid for each pred cuboid
        cos_sim = cosine_similarity(pred, gt)
        most_similar_rows = np.argmax(cos_sim, axis=1)
        diff = gt[most_similar_rows]-pred

        # Pass criteria xyz_sse < 0.5, yaw < 10 degs
        xyz_threshold = 0.5**2
        yaw_threshold = np.deg2rad(10)

        xyz_sse = np.sum(diff[:, :3]**2, axis=1)
        delta_yaw = diff[:, 3] 
        arr = np.hstack((xyz_sse.reshape(-1, 1), delta_yaw.reshape(-1, 1)))
        eval = np.logical_and(arr[:, 0] < xyz_threshold, np.abs(arr[:, 1]) < yaw_threshold) 

        num_pred = len(eval)
        num_true = np.count_nonzero(eval)
        percent = np.round((num_true/num_pred)*100.0, 2)
        print("{} row {}, {} gt cuboids, {}/{} pred cuboids {}%".format(file, idx, len(gt_cuboids_cam), num_true, num_pred, percent))

        total_pred += num_pred
        total_true += num_true
        total_gt += len(gt_cuboids_cam)

percent = np.round((total_true/total_pred)*100.0, 2)
print("Overall stats: {} gt cuboids, {}/{} pred cuboids {}%".format(total_gt, total_true, total_pred, percent))

TP = total_true
FP = total_pred - total_true
FN = total_gt - total_pred
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print("Precision {}, Recall {}".format(precision, recall))