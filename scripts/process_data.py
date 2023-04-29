import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

def clip_fov(xyz, cuboids, lower=-np.pi / 6, upper=np.pi / 6):
    # print('lower:', lower, 'upper:', upper)
    x, y = xyz[:, :2].T
    theta = np.arctan2(y, x)
    if lower < upper:
        filtered_idx = np.where((theta > lower) & (theta < upper))
    else:
        filtered_idx = np.where((theta > lower) | (theta < upper))
    filtered_xyz = xyz[filtered_idx]

    box_dim_x = 2
    box_dim_y = 4.5
    box_dim_z = 2

    i_cuboids = {}
    for cuboid in cuboids:
        c_r = np.array([cuboid['x'], cuboid['y'], cuboid['z']]).reshape((1, 3))
        q = R.from_quat([cuboid['qx'], cuboid['qy'], cuboid['qz'], cuboid['w']])

        p_r = filtered_xyz
        p_c = p_r - c_r
        p_c_rot = q.apply(p_c)

        # print(p_c_rot.shape)
        num_intersecting = np.sum((np.abs(p_c_rot[:, 0]) <= box_dim_x / 2) & (np.abs(p_c_rot[:, 1]) <= box_dim_y / 2) & (np.abs(p_c_rot[:, 2]) <= box_dim_z / 2))
        if num_intersecting > 70:
            i_cuboids[str(cuboid)] = cuboid

    return filtered_xyz, list(i_cuboids.values())

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

    d_theta = np.pi / 6

    for _, row in df.iterrows():
        xyz, cuboids, T = row['scan'], row['cuboids'], row['T']

        for i in range(6):
            clipped_xyz, clipped_cuboids = clip_fov(xyz, cuboids, lower=(i*d_theta)-d_theta, upper=(i*d_theta)+d_theta)
            df_out.loc[len(df_out.index)] = [clipped_xyz, clipped_cuboids, T]

        for i in range(5):
            clipped_xyz, clipped_cuboids = clip_fov(xyz, cuboids, lower=-i*d_theta-np.pi/3, upper=-i*d_theta)
            df_out.loc[len(df_out.index)] = [clipped_xyz, clipped_cuboids, T]

        clipped_xyz, clipped_cuboids = clip_fov(xyz, cuboids, lower=5*d_theta, upper=-5*d_theta)
        df_out.loc[len(df_out.index)] = [clipped_xyz, clipped_cuboids, T]

    fout = args.input.replace('.pkl', '_processed.pkl')
    df_out.to_pickle(fout)

if __name__ == '__main__':
    main()
