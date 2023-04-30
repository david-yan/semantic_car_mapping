import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

from utils import rotate_cuboid

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


def ransac_plane(xyz, threshold=0.2, iterations=50000):
    best_sample = None
    best_w = None
    idxs_inlier=[]
    n_points=len(xyz)
    for i in range(iterations):
        idx_samples = np.random.choice(n_points, 3)
        pts = xyz[idx_samples]
        vecA = pts[1] - pts[0]
        vecB = pts[2] - pts[0]
        normal = np.cross(vecA, vecB)
        w = normal / np.linalg.norm(normal)
        a,b,c = w
        d=-np.sum(normal*pts[1])
        distance = (a * xyz[:,0] + b * xyz[:,1] + c * xyz[:,2] + d
                    ) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

        idx_candidates = np.where(np.abs(distance) <= threshold)[0]
        if len(idx_candidates) > len(idxs_inlier):
            idxs_inlier = idx_candidates
            best_sample = idx_samples
            best_w = w
            
    return best_w, idxs_inlier, best_sample

def main():
    # Create an argparse argument parser
    parser = argparse.ArgumentParser(description='Convert a raw text file to JSON')

    # Add arguments
    parser.add_argument('--input', help='Path to the data file')
    parser.add_argument('--augment-factor', help='What factor to augment the data by', default=0, type=int)

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

        corrected_xyz = q_rot.apply(xyz)
        corrected_cuboids = [rotate_cuboid(cuboid, q_rot) for cuboid in cuboids]
        corrected_T = T_rot @ T

        # Clip FOV
        # d_theta = np.pi / 6
        # for i in range(6):
        #     clipped_xyz, clipped_cuboids = clip_fov(xyz, cuboids, lower=(i*d_theta)-d_theta, upper=(i*d_theta)+d_theta)
        #     df_out.loc[len(df_out.index)] = [clipped_xyz, clipped_cuboids, T]

        # for i in range(5):
        #     clipped_xyz, clipped_cuboids = clip_fov(xyz, cuboids, lower=-i*d_theta-np.pi/3, upper=-i*d_theta)
        #     df_out.loc[len(df_out.index)] = [clipped_xyz, clipped_cuboids, T]

        # clipped_xyz, clipped_cuboids = clip_fov(xyz, cuboids, lower=5*d_theta, upper=-5*d_theta)
        # df_out.loc[len(df_out.index)] = [clipped_xyz, clipped_cuboids, T]

        for _ in range(args.augment_factor):
            theta_ran = np.random.uniform(0, 2 * np.pi)
            q_ran = R.from_quat([0, 0, np.sin(theta_ran / 2), np.cos(theta_ran / 2)])
            T_ran = np.zeros((4, 4))
            T_ran[:3, :3] = q_ran.as_matrix()
            T_ran[3, 3] = 1

            rotated_xyz = q_ran.apply(corrected_xyz)
            rotated_cuboids = [rotate_cuboid(cuboid, q_ran) for cuboid in corrected_cuboids]
            rotated_T = T_ran @ corrected_T
            df_out.loc[len(df_out.index)] = [rotated_xyz, rotated_cuboids, rotated_T]

    fout = args.input.replace('.pkl', '_processed.pkl')
    df_out.to_pickle(fout)

if __name__ == '__main__':
    main()
