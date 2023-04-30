import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import rotate_cuboid
from visualize_data import draw_cuboid
from process_data import ransac_plane

def main():
    # Create an argparse argument parser
    parser = argparse.ArgumentParser(description='Convert a raw text file to JSON')

    # Add arguments
    parser.add_argument('--input', help='Path to the data file')

    # Parse the arguments
    args = parser.parse_args()

    df = pd.read_pickle(args.input)
    df.reset_index()

    df_out = pd.DataFrame(data={'scan': [], 'cuboids': [], 'T': []})

    for _, row in df.iterrows():
        xyz, cuboids, T = row['scan'], row['cuboids'], row['T']
        filtered_xyz = xyz[np.where(xyz[:, 2] <= 1)]
        # print('pre unique check', len(filtered_xyz))
        filtered_xyz = np.unique(filtered_xyz, axis=0)
        # print('post unique check', len(filtered_xyz))
        w_plane, idxs_inlier, idx_sample = ransac_plane(filtered_xyz)
        # print('num inliers:', len(np.unique(idxs_inlier)))

        # fig = plt.figure()
        # fig.clear()

        # ax = Axes3D(fig)
        # ax.set_xlim(-50, 50)
        # ax.set_ylim(-50, 50)
        # ax.set_zlim(-30, 30)

        # x, y, z = filtered_xyz[idxs_inlier].T
        # ax.scatter(x, y, z, s=1, color='r')

        # x, y, z = np.delete(filtered_xyz, idxs_inlier, 0).T
        # ax.scatter(x, y, z, s=0.5, color='b')

        # x, y, z = filtered_xyz[idx_sample].T
        # ax.scatter(x, y, z, s=20, color='g')

        # ax.set_title('Before rotation')
        
        # ax.plot([0, w_plane[0]], [0, w_plane[1]], [0, w_plane[2]], 'r')

        # plt.show()

        w_des = np.array([0, 0, 1])
        r_axis = np.cross(w_des, w_plane)
        r_axis /= np.linalg.norm(r_axis)
        r_theta = np.arccos(w_plane.dot(w_des))
        # print(r_axis, r_theta)

        flip_z = None
        if r_theta > 3/4 * np.pi:
            flip_z = R.from_quat([np.sin(np.pi/2), 0, 0, np.cos(np.pi/2)])

        # print(np.hstack((r_axis * np.sin(r_theta/2), np.cos(r_theta/2))))
        q_rot = R.from_quat(np.hstack((r_axis * np.sin(r_theta/2), np.cos(r_theta/2)))).inv()
        if flip_z is not None:
            q_rot = flip_z * q_rot
        # print('q_rot:', q_rot.as_quat())

        T_rot = np.zeros((4, 4))
        T_rot[:3, :3] = q_rot.as_matrix()
        T_rot[3, 3] = 1

        rotated_xyz = q_rot.apply(xyz)
        rotated_cuboids = [rotate_cuboid(cuboid, q_rot) for cuboid in cuboids]
        rotated_T = T_rot @ T

        # fig = plt.figure()
        # fig.clear()

        # ax = Axes3D(fig)
        # ax.set_xlim(-50, 50)
        # ax.set_ylim(-50, 50)
        # ax.set_zlim(-30, 30)

        # x, y, z = rotated_xyz.T
        # ax.scatter(x, y, z, s=0.5, color='b')

        # ax.set_title('After rotation')
        # for cuboid in rotated_cuboids:
        #     draw_cuboid(ax, cuboid)
        
        # plt.show()
        
        df_out.loc[len(df_out.index)] = [rotated_xyz, rotated_cuboids, rotated_T]

    
        
    fout = args.input.replace('.pkl', '_corrected.pkl')
    df_out.to_pickle(fout)

if __name__ == '__main__':
    main()
