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

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

class PointCloudReader:
    def __init__(self, args):
        self.args = args
        with open(args.boxes, 'r') as f:
            self.cuboids_data = json.load(f)

        self.box_dim_x = 2
        self.box_dim_y = 4.5
        self.box_dim_z = 2

        self.pc_ = rospy.Subscriber('/cloud_registered_body', PointCloud2, callback=self.pc_cb, queue_size=1)
        self.odom_ = rospy.Subscriber('/Odometry', Odometry, callback=self.odom_cb, queue_size=1)
        self.tf2_ = tf.TransformListener()

        self.plot = args.plot

        if self.plot:
            self.fig = plt.figure(1)
            self.fig.clear()
            self.ax = Axes3D(self.fig)
            self.x, self.y, self.z = [], [], []
            self.plt = self.ax.scatter(self.x, self.y, self.z, s=0.5)
            self.title = self.ax.set_title('3D Test, time={}'.format(0))

        self.output = args.output
        self.write_counter = 0
        self.write_size = 10
        self.df = pd.DataFrame(data={'scan': [], 'cuboids': [], 'T': []})

    def pc_cb(self, msg: PointCloud2):
        H = np.zeros((4, 4))

        try:
            t, q = self.tf2_.lookupTransform(msg.header.frame_id, 'quadrotor/odom', msg.header.stamp)
            r = R.from_quat(q)
            H_r = r.as_matrix()
            H_t = np.array(t)

            H[:3, :3] = H_r
            H[:3, 3] = H_t
            H[3, 3] = 1

            xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
            xyz_homo = np.vstack((xyz.T, np.ones(xyz.shape[0])))
            tf_xyz_homo = (np.linalg.pinv(H) @ xyz_homo).T
            factor_stacked = np.repeat(
                tf_xyz_homo[:, 3].reshape(-1, 1), 3, axis=1)
            # normalize
            tf_xyz = np.divide(
                tf_xyz_homo[:, :3], factor_stacked)
            # print(tf_xyz.shape)
            self.x, self.y, self.z = tf_xyz.T

            # print('calculating intersecting cuboids')
            intersecting_cuboids = {}
            for cuboid in self.cuboids_data:
                p_w = tf_xyz
                cp_w = np.array([cuboid['x'], cuboid['y'], cuboid['z']]).reshape((1, 3))
                p_c = p_w - cp_w
                q = R.from_quat([cuboid['qx'], cuboid['qy'], cuboid['qz'], cuboid['w']])
                p_c_rot = q.apply(p_c)
                # print(p_c_rot.shape)
                num_intersecting = np.sum((np.abs(p_c_rot[:, 0]) <= self.box_dim_x / 2) & (np.abs(p_c_rot[:, 1]) <= self.box_dim_y / 2) & (np.abs(p_c_rot[:, 2]) <= self.box_dim_z / 2))
                if num_intersecting > 70:
                    intersecting_cuboids[str(cuboid)] = cuboid
                    print('found intersecting cuboid with %d points'%(num_intersecting))
                    # print('cuboid center:', cp_w)
                    # idxs = np.where((np.abs(p_c_rot[:, 0]) <= self.box_dim_x / 2) & (np.abs(p_c_rot[:, 1]) <= self.box_dim_y / 2) & (np.abs(p_c_rot[:, 2]) <= self.box_dim_z / 2))
                    # print('intersecting points:', tf_xyz[idxs, :])
                    # print('projected intersecting points:', p_c_rot[idxs, :])
            # print('intercepting cuboids:', intersecting_cuboids)
            self.intersect_c = intersecting_cuboids.keys()

            self.df.loc[len(self.df.index)] = [xyz, list(intersecting_cuboids.values()), np.linalg.pinv(H)]
            self.write_counter += 1
            if self.write_counter % self.write_size == 0:
                print('Writing batch to file.')
                fname = os.path.join(self.output, "%d_%d.pkl"%(self.write_counter - self.write_size, self.write_counter))
                self.df.to_pickle(fname)
                self.df = self.df.iloc[0:0]

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Could not find tf2 at ", msg.header.stamp)
        
    def odom_cb(self, msg: Odometry):
        pass

    def draw_cuboid(self, cuboid, intersecting=False):
        q = R.from_quat([cuboid['qx'], cuboid['qy'], cuboid['qz'], cuboid['w']])
        x = np.array([1, 0, 0])
        r90z = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
        r90y = R.from_quat([0, np.sin(np.pi/4), 0, np.cos(np.pi/4)])
        x_rot = q.apply(x) * self.box_dim_x / 2
        y_rot = r90z.apply(x_rot) * self.box_dim_y / 2
        z_rot = r90y.apply(x_rot) * self.box_dim_z / 2

        xx_plot = [cuboid['x'] - x_rot[0], cuboid['x'] + x_rot[0]]
        xy_plot = [cuboid['y'] - x_rot[1], cuboid['y'] + x_rot[1]]
        xz_plot = [cuboid['z'] - x_rot[2], cuboid['z'] + x_rot[2]]

        yx_plot = [cuboid['x'] - y_rot[0], cuboid['x'] + y_rot[0]]
        yy_plot = [cuboid['y'] - y_rot[1], cuboid['y'] + y_rot[1]]
        yz_plot = [cuboid['z'] - y_rot[2], cuboid['z'] + y_rot[2]]

        zx_plot = [cuboid['x'] - z_rot[0], cuboid['x'] + z_rot[0]]
        zy_plot = [cuboid['y'] - z_rot[1], cuboid['y'] + z_rot[1]]
        zz_plot = [cuboid['z'] - z_rot[2], cuboid['z'] + z_rot[2]]

        c = 'r' if not intersecting else 'b'

        cuboid_plot = [self.ax.plot(xx_plot, xy_plot, xz_plot, c)[0], self.ax.plot(yx_plot, yy_plot, yz_plot, c)[0], self.ax.plot(zx_plot, zy_plot, zz_plot, c)[0]]
        return cuboid_plot

    def plot_init(self):
        self.ax.set_xlim(-100, 25)
        self.ax.set_ylim(-100, 50)
        self.ax.set_zlim(-100, 100)

        self.c_plots = {}
        self.intersect_c = set()

        for cuboid in self.cuboids_data:
            self.c_plots[str(cuboid)] = self.draw_cuboid(cuboid)
            
        return self.plt

    def update_plot(self, frame):
        self.plt._offsets3d = (self.x, self.y, self.z)
        for key, c_plot in self.c_plots.items():
            # print('searching for key:', key)
            if key in self.intersect_c:
                # print('setting cuboid to be intersecting:', key)
                # print(c_plot)
                for plot in c_plot:
                    plot.set(color='b')
            else:
                # print(c_plot)
                for plot in c_plot:
                    plot.set(color='r')

        self.title.set_text('3D Test, time={}'.format(frame))
        return self.plt

# Initialize the interactive marker server
def main():
    # Create an argparse argument parser
    parser = argparse.ArgumentParser(description='Convert a raw text file to JSON')

    # Add arguments
    parser.add_argument('--boxes', help='Path to the boxes file')
    parser.add_argument('--output', type=str, help='Path to output directory', default='.')
    parser.add_argument('--plot', action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    rospy.init_node('data_reader')
    pcr = PointCloudReader(args)
    if args.plot:
        ani = FuncAnimation(pcr.fig, pcr.update_plot, init_func=pcr.plot_init)
        plt.show(block=True) 

    rospy.spin()

    

if __name__ == '__main__':
    # rospy.init_node("interactive_marker_node")

    main()
