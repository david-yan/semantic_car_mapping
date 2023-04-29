import _init_path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.net.point_rcnn import PointRCNN
import tools.train_utils.train_utils as train_utils

from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset

import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import argparse
import logging
import re
from lib.utils.bbox_transform import decode_bbox_target

import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(1024)  # set the same seed

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
    
parser.add_argument('--input', default=None, help='Path to the data file')

parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                    help='set extra config keys if needed')

args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def create_dataloader(logger):
    # TODO: new data loader for our use case
    mode = 'TEST' if args.test else 'EVAL'
    DATA_PATH = os.path.join('..', 'data')

    # create dataloader
    test_set = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TEST.SPLIT, mode=mode,
                                random_select=args.random_select,
                                rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
                                rcnn_eval_feature_dir=args.rcnn_eval_feature_dir,
                                classes=cfg.CLASSES,
                                logger=logger)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.workers, collate_fn=test_set.collate_batch)

    return test_loader


def load_ckpt_based_on_args(model, logger):
    if args.ckpt is not None:
        train_utils.load_checkpoint(model, filename=args.ckpt, logger=logger)

    total_keys = model.state_dict().keys().__len__()
    # if cfg.RPN.ENABLED and args.rpn_ckpt is not None:
    #     load_part_ckpt(model, filename=args.rpn_ckpt, logger=logger, total_keys=total_keys)

    # if cfg.RCNN.ENABLED and args.rcnn_ckpt is not None:
    #     load_part_ckpt(model, filename=args.rcnn_ckpt, logger=logger, total_keys=total_keys)


def get_input():
    # TODO: change to get input with dataloader
    df = pd.read_pickle(args.input)
    df.reset_index()

    for i, row in df.iterrows():
        xyz, cuboids, T = row['scan'], row['cuboids'], row['T']
        # xyz_homo = np.vstack((xyz.T, np.ones(xyz.shape[0])))
        # tf_xyz_homo = (T @ xyz_homo).T
        # factor_stacked = np.repeat(
        #     tf_xyz_homo[:, 3].reshape(-1, 1), 3, axis=1)
        # # normalize
        # tf_xyz = np.divide(
        #     tf_xyz_homo[:, :3], factor_stacked)
        # break   # just use one sample for now

    return xyz[np.newaxis, ...]


# def visualize_data(pts_input, pred_boxes3d):
#     fig = plt.figure()
#     fig.clear()

#     ax = Axes3D(fig)
#     ax.set_xlim(-100, 25)
#     ax.set_ylim(-100, 50)
#     ax.set_zlim(-100, 100)

#     x, y, z = pts_input[0].T
#     ax.scatter(x, y, z, s=0.5)
#     ax.set_title('3D Test, i={}'.format(0))
#     # for cuboid in cuboids:
#     #     draw_cuboid(ax, cuboid)
#     plt.show()


def eval_one_epoch(model, dataloader, epoch_id, result_dir, logger):
    # if cfg.RPN.ENABLED and not cfg.RCNN.ENABLED:
    #     ret_dict = eval_one_epoch_rpn(model, dataloader, epoch_id, result_dir, logger)
    # elif not cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
    #     ret_dict = eval_one_epoch_rcnn(model, dataloader, epoch_id, result_dir, logger)
    # elif cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
    #     ret_dict = eval_one_epoch_joint(model, dataloader, epoch_id, result_dir, logger)
    # else:
    #     raise NotImplementedError
    if cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_joint(model, dataloader, epoch_id, result_dir, logger)
    else:
        raise NotImplementedError
    return ret_dict


def eval_one_epoch_joint(model, dataloader, epoch_id, result_dir, logger):
    np.random.seed(666)
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    mode = 'TEST' if args.test else 'EVAL'

    final_output_dir = os.path.join(result_dir, 'final_result', 'data')
    os.makedirs(final_output_dir, exist_ok=True)

    logger.info('---- EPOCH %s JOINT EVALUATION ----' % epoch_id)
    logger.info('==> Output file: %s' % result_dir)
    model.eval()

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    total_roi_recalled_bbox_list = [0] * 5
    cnt = final_total = total_cls_acc = total_cls_acc_refined = total_rpn_iou = 0


    pts_input = get_input()
    batch_size = len(pts_input)
    inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
    input_data = {'pts_input': inputs}
    print(pts_input.shape, batch_size)
    # model inference
    ret_dict = model(input_data)

    roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
    roi_boxes3d = ret_dict['rois']  # (B, M, 7)
    seg_result = ret_dict['seg_result'].long()  # (B, N)

    rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
    rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

    # bounding box regression
    anchor_size = MEAN_SIZE
    if cfg.RCNN.SIZE_RES_ON_ROI:
        assert False

    pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                        anchor_size=anchor_size,
                                        loc_scope=cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                        get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                        get_ry_fine=True).view(batch_size, -1, 7)
    # print(pred_boxes3d, rcnn_cls)
    # scoring
    if rcnn_cls.shape[2] == 1:
        raw_scores = rcnn_cls  # (B, M, 1)

        norm_scores = torch.sigmoid(raw_scores)
        pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()

    # scores thresh
    inds = norm_scores > cfg.RCNN.SCORE_THRESH

    for k in range(batch_size):
        cur_inds = inds[k].view(-1)
        if cur_inds.sum() == 0:
            continue

        pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
        raw_scores_selected = raw_scores[k, cur_inds]
        norm_scores_selected = norm_scores[k, cur_inds]

        # NMS thresh
        # rotated nms
        boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
        keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
        pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
        scores_selected = raw_scores_selected[keep_idx]
        pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.cpu().numpy(), scores_selected.cpu().numpy()
        
        print(pred_boxes3d_selected, scores_selected)
    # visualize_data(pts_input, pred_boxes3d_selected)
    return ret_dict


def eval_single_ckpt(root_result_dir):
    root_result_dir = os.path.join(root_result_dir, 'eval')
    # set epoch_id and output dir
    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    root_result_dir = os.path.join(root_result_dir, 'epoch_%s' % epoch_id, cfg.TEST.SPLIT)
    if args.test:
        root_result_dir = os.path.join(root_result_dir, 'test_mode')

    if args.extra_tag != 'default':
        root_result_dir = os.path.join(root_result_dir, args.extra_tag)
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, 'log_eval_one.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    save_config_to_file(cfg, logger=logger)

    # create dataloader & network
    test_loader = create_dataloader(logger)
    model = PointRCNN(num_classes=2, use_xyz=True, mode='TEST')
    model.cuda()

    # copy important files to backup
    # backup_dir = os.path.join(root_result_dir, 'backup_files')
    # os.makedirs(backup_dir, exist_ok=True)
    # os.system('cp *.py %s/' % backup_dir)
    # os.system('cp ../lib/net/*.py %s/' % backup_dir)
    # os.system('cp ../lib/datasets/kitti_rcnn_dataset.py %s/' % backup_dir)

    # load checkpoint
    load_ckpt_based_on_args(model, logger)

    # start evaluation
    eval_one_epoch(model, test_loader, epoch_id, root_result_dir, logger)


if __name__ == "__main__":
    # merge config and log to file
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    if args.eval_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'rpn', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rpn', cfg.TAG, 'ckpt')
    elif args.eval_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
        root_result_dir = os.path.join('../', 'output', 'rcnn_new', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rcnn_new', cfg.TAG, 'ckpt')
    else:
        raise NotImplementedError

    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir

    if args.output_dir is not None:
        root_result_dir = args.output_dir

    os.makedirs(root_result_dir, exist_ok=True)

    with torch.no_grad():
        if not args.eval_all:
            eval_single_ckpt(root_result_dir)
