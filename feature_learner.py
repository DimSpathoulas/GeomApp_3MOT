from __future__ import print_function
from tqdm import tqdm
import os.path
import sys
import time
import pickle
import copy
import argparse
import json
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from splits import get_scenes_of_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from utils import mkdir_if_missing
from sklearn.decomposition import PCA
from filterpy.kalman import KalmanFilter
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
import umap
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from Nets.test import DIST_COMB_MODULE
from functions.Kalman_Filter import KalmanBoxTracker


from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
    'background'
]


def create_box_annotations(sample_token, nusc):
    ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

    sample = nusc.get('sample', sample_token)
    ann_token = sample['anns']

    for ann in ann_token:

        ann_meta = nusc.get('sample_annotation', ann)
        t_name = ann_meta['category_name']

        for tracking_name in NUSCENES_TRACKING_NAMES:
            if tracking_name in t_name:
                # if tracking_name == 'truck':
                #     pass

                trs = np.array(ann_meta['translation'])
                q = Quaternion(ann_meta['rotation'])
                angle = q.angle if q.axis[2] > 0 else -q.angle
                size = np.array(ann_meta['size'])

                gt_box = np.array([trs[0], trs[1], trs[2],
                                   size[0], size[1], size[2], angle, ann_meta['instance_token']])

                ground_truths[tracking_name].append(gt_box)

    return ground_truths


def hungarian_matching_remake(estims, trues):
    estims_array = np.array([e[:2] for e in estims], dtype=float)
    trues_array = np.array([t[:2] for t in trues], dtype=float)

    cost_matrix = np.linalg.norm(estims_array[:, np.newaxis] - trues_array, axis=2)

    return linear_sum_assignment(cost_matrix)

def feature_learner():

    parser = argparse.ArgumentParser(description="TrainVal G2 with lidar and camera detected characteristics")
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='NuScenes dataset version')
    parser.add_argument('--data_root', type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval',
                        help='Root directory of the NuScenes dataset')
    parser.add_argument('--data', type=str,
                        default="/home/ktsiakas/thesis_new/2D_FEATURE_EXTRACTOR/mrcnn_val_057.pkl",
                        help='Path to detections, train split for train - val split for inference')

    args = parser.parse_args()
    data = args.data
    data_root = args.data_root
    version = args.version

    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    COMB = DIST_COMB_MODULE().to(device)
    params_to_optimize = list(COMB.G1.parameters()) + list(COMB.G11.parameters())

    optimizer = torch.optim.Adam(params_to_optimize, lr=0.001) 

    criterion = nn.BCEWithLogitsLoss()  # built-in sigmoid for stability
    criterion = nn.BCELoss()

    with open(data, 'rb') as f:
        all_results = pickle.load(f)

    processed_scene_tokens = set()

    for sample, sample_data in tqdm(all_results.items()):

        scene_token = nusc.get('sample', sample)['scene_token']

        if scene_token in processed_scene_tokens:
            continue

        first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        current_sample_token = first_sample_token

        while current_sample_token != '':
            dets_all = {track_name: [] for track_name in NUSCENES_TRACKING_NAMES}
            pcds_all = {track_name: [] for track_name in NUSCENES_TRACKING_NAMES}
            fvs_all = {track_name: [] for track_name in NUSCENES_TRACKING_NAMES}
            cam_vecs = {track_name: [] for track_name in NUSCENES_TRACKING_NAMES}
            gts = create_box_annotations(current_sample_token, nusc)

            for i, item in enumerate(all_results[current_sample_token]):
                for name in NUSCENES_TRACKING_NAMES:
                    for dets_outputs in item[name]:
                        dets_all[name].append(dets_outputs['box'])
                        pcd_feature = np.expand_dims(dets_outputs['point_cloud_features'], axis=0) 
                        pcds_all[name].append(pcd_feature)
                        fvs_all[name].append(dets_outputs['feature_vector'])
                        cam_vecs[name].append(dets_outputs['camera_onehot_vector'])

            for name in NUSCENES_TRACKING_NAMES:

                class_idx = NUSCENES_TRACKING_NAMES.index(name)
                onehot_vector = torch.zeros(len(NUSCENES_TRACKING_NAMES))
                onehot_vector[class_idx] = 1
                class_onehot = onehot_vector.unsqueeze(0).repeat(dets.shape[0], 1)

                dets_all[name] = np.vstack(dets_all[name])
                pcds_all[name] = np.vstack(pcds_all[name])
                fvs_all[name] = np.vstack(fvs_all[name])
                cam_vecs[name] = np.vstack(cam_vecs[name])

                predictions, _ = COMB(fvs_all[name], pcds_all[name],cam_vecs[name])
                print(predictions)
                # same shape as class_onehot

                d_idx, d_gt_idx = hungarian_matching_remake(dets_all[name], gts[name])
                dist = np.linalg.norm(dets_array[d_idx] - curr_gts_array[d_gt_idx], axis=1)
                gt_labels = np.zeros_like(dist, dtype=int)
                gt_labels[dist <= threshold] = 1

                # based on this distance create a coressponding groundtruth to compare with predictions

            current_sample_token = nusc.get('sample', current_sample_token)['next']

        processed_scene_tokens.add(scene_token)
    
    print(f'Number of processed scene tokens: {len(processed_scene_tokens)}')

    for name in NUSCENES_TRACKING_NAMES:
        print(f'Shape of pcds_all[{name}]: {pcds_all[name].shape}')
        print(f'Shape of fvs_all[{name}]: {fvs_all[name].shape}')



if __name__ == '__main__':
    feature_learner()