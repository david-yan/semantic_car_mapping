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

rospy.init_node('model_visualizer')

# Parse input arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--idx', type=int, required=True, help='Frame number')
# parser.add_argument('--input', type=str, required=False, help='Input pickle file')
parser.add_argument('--model', type=str, required=True, help='Model output pickle file')
args = parser.parse_args()

# Load the pickle file into a DataFrame
with open(args.model, 'rb') as f:
    df = pickle.load(f)
    # idx = args.idx
    idx = 0 # pickle files just have 1 scan for now

# convert numpy array to structured array with named fields
cloud = df['scan'][idx]
cloud_arr = np.zeros_like(cloud)
cloud_arr[:, 0] = cloud[:, 2]
cloud_arr[:, 1] = -cloud[:, 0]
cloud_arr[:, 2] = -cloud[:, 1]
data = np.array([(x, y, z) for x, y, z in cloud_arr], dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
pcd_msg = ros_numpy.point_cloud2.array_to_pointcloud2(data, frame_id='odom')
pub_pcd = rospy.Publisher('/pointcloud', PointCloud2, queue_size=10)

# get the predicted cuboids from the model
gt_cuboids_cam = df['gt_cuboids'][idx]
gt_cuboids = []
print("there are {} gt_cuboids".format(len(gt_cuboids_cam)))
for cuboid in gt_cuboids_cam:
    # transform into quadrotor frame by switching coordinate; x rightward, y downward, z forward
    gt_cuboid = {}
    gt_cuboid['x'] = cuboid[2]
    gt_cuboid['y'] = -cuboid[0]
    gt_cuboid['z'] = -cuboid[1]

    # Calculate the qx,qy,qz,w by creating a unit quaternion, rotate in z-axis by yaw equal to negative yaw cuboid['ry']
    q = R.from_euler('z', cuboid[-1])
    gt_cuboid['qx'], gt_cuboid['qy'], gt_cuboid['qz'], gt_cuboid['w'] = q.as_quat()
    gt_cuboids.append(gt_cuboid)

interactive_gt_cuboids = InteractiveCuboids(cuboids_data=gt_cuboids, frame_id="gt_cuboids", server_name="gt_cuboids_server", color=(0.5, 0.5, 1, 0.6))
interactive_gt_cuboids.visualize()


# get the predicted cuboids from the model
pred_cuboids_cam = df['cuboids'][idx]
pred_cuboids = []
print("there are {} pred_cuboids".format(len(pred_cuboids_cam)))
for cuboid in pred_cuboids_cam:
    # transform into quadrotor frame by switching coordinate; x rightward, y downward, z forward
    pred_cuboid = {}
    pred_cuboid['x'] = cuboid[2]
    pred_cuboid['y'] = -cuboid[0]
    pred_cuboid['z'] = -cuboid[1]

    # Calculate the qx,qy,qz,w by creating a unit quaternion, rotate in z-axis by yaw equal to negative yaw cuboid['ry']
    q = R.from_euler('z', cuboid[-1])
    pred_cuboid['qx'], pred_cuboid['qy'], pred_cuboid['qz'], pred_cuboid['w'] = q.as_quat()
    pred_cuboids.append(pred_cuboid)

interactive_pred_cuboids = InteractiveCuboids(cuboids_data=pred_cuboids, frame_id="pred_cuboids", server_name="pred_cuboids_server", color=(1, 0.5, 0.5, 0.8))
interactive_pred_cuboids.visualize()

# print out evaluation
gt = gt_cuboids_cam[:,[0,1,2,-1]]
pred = pred_cuboids_cam[:,[0,1,2,-1]]
pred[:,-1] = np.where(pred[:, -1] > 0, pred[:, -1] - np.pi, pred[:, -1])

# use cosine similarity to figure out which is closest gt cuboid for each pred cuboid
cos_sim = cosine_similarity(pred, gt)
most_similar_rows = np.argmax(cos_sim, axis=1)
diff = gt[most_similar_rows]-pred

# Pass criteria xyz_sse < 0.5, yaw < 10 degs
xyz_threshold = 0.5
yaw_threshold = np.deg2rad(10)

xyz_sse = np.sum(diff[:, :3]**2, axis=1)
delta_yaw = diff[:, 3] 
arr = np.hstack((xyz_sse.reshape(-1, 1), delta_yaw.reshape(-1, 1)))
eval = np.logical_and(arr[:, 0] < xyz_threshold, np.abs(arr[:, 1]) < yaw_threshold) 
np.set_printoptions(suppress=True, precision=6)
print(arr)
print(eval)

# Create rate object with desired frequency of 1 Hz
rate = rospy.Rate(1)

# Loop to publish the point cloud at the desired rate
while not rospy.is_shutdown():
    pub_pcd.publish(pcd_msg)
    rate.sleep()