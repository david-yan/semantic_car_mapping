######## Evaluation pipeline for perch data on pointrcnn model #########
# Author: Beiming Li

import _init_path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.datasets.perch_dataset import PerchDataset
import argparse

from lib.net.point_rcnn import PointRCNN
import tools.train_utils.train_utils as train_utils
from lib.utils.bbox_transform import decode_bbox_target
from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate

from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
from datetime import datetime
import logging
import re
import glob
import time
from tensorboardX import SummaryWriter
import tqdm

import matplotlib.pyplot as plt

np.random.seed(1024)  # set the same seed

parser = argparse.ArgumentParser(description="arg parser")

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--cfg_file', type=str, default='cfgs/default.yml', help='specify the config for evaluation')
parser.add_argument("--eval_mode", type=str, default='rpn', required=True, help="specify the evaluation mode")

parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
parser.add_argument('--test', action='store_true', default=False, help='evaluate without ground truth')
parser.add_argument("--ckpt", type=str, default=None, help="specify a checkpoint to be evaluated")
parser.add_argument("--rpn_ckpt", type=str, default=None, help="specify the checkpoint of rpn if trained separated")
parser.add_argument("--rcnn_ckpt", type=str, default=None, help="specify the checkpoint of rcnn if trained separated")

parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--extra_tag", type=str, default='default', help="extra tag for multiple evaluation")
parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
parser.add_argument("--ckpt_dir", type=str, default=None, help="specify a ckpt directory to be evaluated if needed")

parser.add_argument('--save_result', action='store_true', default=False, help='save evaluation results to files')
parser.add_argument('--save_rpn_feature', action='store_true', default=False,
                    help='save features for separately rcnn training and evaluation')

parser.add_argument('--random_select', action='store_true', default=True, help='sample to the same number of points')
parser.add_argument('--start_epoch', default=0, type=int, help='ignore the checkpoint smaller than this epoch')
parser.add_argument("--rcnn_eval_roi_dir", type=str, default=None,
                    help='specify the saved rois for rcnn evaluation when using rcnn_offline mode')
parser.add_argument("--rcnn_eval_feature_dir", type=str, default=None,
                    help='specify the saved features for rcnn evaluation when using rcnn_offline mode')
parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                    help='set extra config keys if needed')
args = parser.parse_args()


def create_dataloader():
    mode = 'TEST' if args.test else 'EVAL'
    DATA_PATH = os.path.join('..', 'data')

    # create dataloader
    test_set = PerchDataset(root_dir=DATA_PATH, mode=mode)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.workers, collate_fn=test_set.collate_batch)

    return test_loader


def load_ckpt_based_on_args(model):
    train_utils.load_checkpoint(model, filename=args.ckpt)


def eval_one_epoch(model, dataloader):
    if cfg.RPN.ENABLED and not cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_rpn(model, dataloader)
    elif not cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_rcnn(model, dataloader)
    elif cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_joint(model, dataloader)
    else:
        raise NotImplementedError
    return ret_dict


def eval_one_epoch_rpn(model, dataloader):
    np.random.seed(1024)
    mode = 'TEST' if args.test else 'EVAL'

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0

    for data in dataloader:
        pts_rect, pts_features, pts_input, gt_boxes3d = data['pts_rect'], data['pts_features'], data['pts_input'], data['gt_boxes3d']
        inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
        input_data = {'pts_input': inputs}

        # model inference
        ret_dict = model(input_data)
        rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']
        backbone_xyz, backbone_features = ret_dict['backbone_xyz'], ret_dict['backbone_features']

        rpn_scores_raw = rpn_cls[:, :, 0]
        rpn_scores = torch.sigmoid(rpn_scores_raw)
        seg_result = (rpn_scores > cfg.RPN.SCORE_THRESH).long()

        # proposal layer
        rois, roi_scores_raw = model.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)
        batch_size = rois.shape[0]

        # calculate recall and save results to file
        for bs_idx in range(batch_size):
            cur_scores_raw = roi_scores_raw[bs_idx]  # (N)
            cur_boxes3d = rois[bs_idx]  # (N, 7)
            cur_seg_result = seg_result[bs_idx]
            cur_pts_rect = pts_rect[bs_idx]
            visualize_data(pts_input[bs_idx], gt_boxes3d[bs_idx], cur_boxes3d[cur_scores_raw > 0].cpu())

    return ret_dict


def visualize_data(pts_input, gt_boxes3d, pred_boxes3d):
    print("visualization")
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim(-100, 25)
    ax.set_ylim(-100, 50)
    ax.set_zlim(-100, 100)

    x, y, z = pts_input.T
    ax.scatter3D(x, y, z, s=0.5)

    x_gt, y_gt, z_gt = gt_boxes3d[:, 0], gt_boxes3d[:, 1], gt_boxes3d[:, 2]
    ax.scatter3D(x_gt, y_gt, z_gt, s=50, c='r')

    x_pred, y_pred, z_pred = pred_boxes3d[:, 0], pred_boxes3d[:, 1], pred_boxes3d[:, 2]
    ax.scatter3D(x_pred, y_pred, z_pred, s=100, c='g')
    plt.show()


if __name__ == "__main__":
    # merge config and log to file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    if args.eval_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
    elif args.eval_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
    else:
        raise NotImplementedError

    with torch.no_grad():
        # create dataloader & network
        test_loader = create_dataloader()
        model = PointRCNN(num_classes=2, use_xyz=True, mode='TEST')
        model.cuda()

        # load checkpoint
        load_ckpt_based_on_args(model)

        # start evaluation
        eval_one_epoch(model, test_loader)