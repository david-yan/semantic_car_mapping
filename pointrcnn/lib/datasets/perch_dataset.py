######## Derived Dataset class for PERCH Dataset #########
# Author: Beiming Li

import os
import numpy as np
import pickle
import torch.utils.data as torch_data
import pandas as pd
from scipy.spatial.transform import Rotation as R
from lib.config import cfg
import lib.utils.kitti_utils as kitti_utils

class PerchDataset(torch_data.Dataset):
    def __init__(self, root_dir, mode):
        self.mode = mode
        # TODO: change the hardcoding
        self.input_dirs = [os.path.join(root_dir, 'PENN', 'PERCH', 'very-important-3_processed.pkl'),
                          os.path.join(root_dir, 'PENN', 'PERCH', 'falcon_processed.pkl'),
                          os.path.join(root_dir, 'PENN', 'PERCH', 'open-field_processed.pkl')]
        
        # for old model
        # if mode == 'EVAL':
        #     self.input_dirs = [os.path.join(root_dir, 'PENN', 'PERCH', 'very-important-3-preprocessed.pkl')]
        self.load_pickle(self.input_dirs)

    def load_pickle(self, input_dirs):
        self.pts_lidar, self.gt_boxes3d, self.T = [], [], []
        for input_dir in input_dirs:
            df = pd.read_pickle(input_dir)
            
            for i, row in df.iterrows():
                pts_scan, cuboids, T = row['scan'], row['cuboids'], row['T']

                if len(self.T) != 0:
                    t_last = self.T[-1][0:2, 3]
                    t_cur = T[0:2, 3]
                    if np.linalg.norm(t_cur - t_last) < 1:  # if drone hasn't move 1 meter in x, y direction, drop this sample
                        continue
                
                if len(cuboids) == 0: continue # no gt boxes

                self.T.append(T)

                # switch coordinate, x rightward, y downward, z forward
                pts_lidar = np.zeros_like(np.array(pts_scan))
                pts_lidar[..., 0] = -pts_scan[..., 1]
                pts_lidar[..., 1] = -pts_scan[..., 2]
                pts_lidar[..., 2] = pts_scan[..., 0]
                self.pts_lidar.append(pts_lidar)
                if pts_lidar.shape[0] != 65536:
                    print(pts_lidar.shape)

                # process ground truth bounding boxes
                gt_boxes3d = np.empty((len(cuboids), 7))
                for i in range(len(cuboids)):
                    cuboid = cuboids[i]
                    q = R.from_quat([cuboid['qx'], cuboid['qy'], cuboid['qz'], cuboid['w']])
                    rot_euler = q.as_euler('zyx', degrees=False)
                    yaw = rot_euler[0]
                    gt_boxes3d[i] = np.array([-cuboid['y'], -cuboid['z'], cuboid['x'], 1.52563191462, 1.62856739989, 3.88311640418, yaw])
                self.gt_boxes3d.append(gt_boxes3d)

        print("Loaded %d samples from datasets" % len(self.gt_boxes3d))

    def __len__(self):
        return len(self.pts_lidar)

    def __getitem__(self, index):
        sample_info = {}

        aug_pts_lidar, aug_gt_boxes3d = self.pts_lidar[index], self.gt_boxes3d[index]
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            aug_pts_lidar, aug_gt_boxes3d = self.data_augmentation(self.pts_lidar[index], self.gt_boxes3d[index])

        sample_info['pts_input'] = aug_pts_lidar
        sample_info['pts_rect'] = aug_pts_lidar
        sample_info['pts_features'] = aug_pts_lidar
        sample_info['gt_boxes3d'] = aug_gt_boxes3d

        if cfg.RPN.FIXED:
            return sample_info

        rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(aug_pts_lidar, aug_gt_boxes3d)
        sample_info['rpn_cls_label'] = rpn_cls_label
        sample_info['rpn_reg_label'] = rpn_reg_label
        return sample_info

    def data_augmentation(self, pts_input, gt_boxes3d):
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)

        aug_pts_input, aug_gt_boxes3d = pts_input, gt_boxes3d
        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            angle = np.random.uniform(0, 2 * np.pi)
            aug_pts_input = kitti_utils.rotate_pc_along_y(pts_input, rot_angle=angle)
            aug_gt_boxes3d = kitti_utils.rotate_pc_along_y(gt_boxes3d, rot_angle=angle)

            # TODO: calculate the ry after rotation
            aug_gt_boxes3d[:, 6] = (gt_boxes3d[:, 6] + angle) % np.pi

        return aug_pts_input, aug_gt_boxes3d
    
    @staticmethod
    def generate_rpn_training_labels(pts_rect, gt_boxes3d):
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        reg_label = np.zeros((pts_rect.shape[0], 7), dtype=np.float32)  # dx, dy, dz, ry, h, w, l
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
        extend_gt_corners = kitti_utils.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = kitti_utils.in_hull(pts_rect, box_corners)
            fg_pts_rect = pts_rect[fg_pt_flag]
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = kitti_utils.in_hull(pts_rect, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

            # pixel offset of object center
            center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)
            center3d[1] -= gt_boxes3d[k][3] / 2
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_rect  # Now y is the true center of 3d box 20180928

            # size and angle encoding
            reg_label[fg_pt_flag, 3] = gt_boxes3d[k][3]  # h
            reg_label[fg_pt_flag, 4] = gt_boxes3d[k][4]  # w
            reg_label[fg_pt_flag, 5] = gt_boxes3d[k][5]  # l
            reg_label[fg_pt_flag, 6] = gt_boxes3d[k][6]  # ry

        return cls_label, reg_label

    def collate_batch(self, batch):
        if self.mode != 'TRAIN' and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            assert batch.__len__() == 1
            return batch[0]

        batch_size = batch.__len__()
        ans_dict = {}

        for key in batch[0].keys():
            # make sure each sample in the mini batch has the same dimension for bounding boxes
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, batch[k][key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, :batch[i][key].__len__(), :] = batch[i][key]
                ans_dict[key] = batch_gt_boxes3d
                continue

            if isinstance(batch[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)
            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict