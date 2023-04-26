import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import pandas as pd

def draw_cuboid(ax, cuboid):
    box_dim_x = 2
    box_dim_y = 4.5
    box_dim_z = 2

    q = R.from_quat([cuboid['qx'], cuboid['qy'], cuboid['qz'], cuboid['w']])
    x = np.array([1, 0, 0])
    r90z = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    r90y = R.from_quat([0, np.sin(np.pi/4), 0, np.cos(np.pi/4)])
    x_rot = q.apply(x) * box_dim_x / 2
    y_rot = r90z.apply(x_rot) * box_dim_y / 2
    z_rot = r90y.apply(x_rot) * box_dim_z / 2

    xx_plot = [cuboid['x'] - x_rot[0], cuboid['x'] + x_rot[0]]
    xy_plot = [cuboid['y'] - x_rot[1], cuboid['y'] + x_rot[1]]
    xz_plot = [cuboid['z'] - x_rot[2], cuboid['z'] + x_rot[2]]

    yx_plot = [cuboid['x'] - y_rot[0], cuboid['x'] + y_rot[0]]
    yy_plot = [cuboid['y'] - y_rot[1], cuboid['y'] + y_rot[1]]
    yz_plot = [cuboid['z'] - y_rot[2], cuboid['z'] + y_rot[2]]

    zx_plot = [cuboid['x'] - z_rot[0], cuboid['x'] + z_rot[0]]
    zy_plot = [cuboid['y'] - z_rot[1], cuboid['y'] + z_rot[1]]
    zz_plot = [cuboid['z'] - z_rot[2], cuboid['z'] + z_rot[2]]

    c = 'b'

    cuboid_plot = [ax.plot(xx_plot, xy_plot, xz_plot, c)[0], ax.plot(yx_plot, yy_plot, yz_plot, c)[0], ax.plot(zx_plot, zy_plot, zz_plot, c)[0]]
    return cuboid_plot

# Initialize the interactive marker server
def main():
    # Create an argparse argument parser
    parser = argparse.ArgumentParser(description='Convert a raw text file to JSON')

    # Add arguments
    parser.add_argument('--input', help='Path to the data file')

    # Parse the arguments
    args = parser.parse_args()

    df = pd.read_pickle(args.input)
    df.reset_index()

    for i, row in df.iterrows():
        fig = plt.figure()
        fig.clear()

        ax = Axes3D(fig)
        ax.set_xlim(-100, 25)
        ax.set_ylim(-100, 50)
        ax.set_zlim(-100, 100)

        xyz, cuboids, T = row['scan'], row['cuboids'], row['T']
        xyz_homo = np.vstack((xyz.T, np.ones(xyz.shape[0])))
        tf_xyz_homo = (T @ xyz_homo).T
        factor_stacked = np.repeat(
            tf_xyz_homo[:, 3].reshape(-1, 1), 3, axis=1)
        # normalize
        tf_xyz = np.divide(
            tf_xyz_homo[:, :3], factor_stacked)
        # print(tf_xyz.shape)
        x, y, z = tf_xyz.T
        ax.scatter(x, y, z, s=0.5)
        ax.set_title('3D Test, i={}'.format(i))
        for cuboid in cuboids:
            draw_cuboid(ax, cuboid)

        plt.show()

if __name__ == '__main__':
    main()
