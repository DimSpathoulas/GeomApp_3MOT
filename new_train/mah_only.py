'''
AYTO EXEI MONO TO MAH
DLD EINAI TO BASE PAPER
'''


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

from utils import mkdir_if_missing

from filterpy.kalman import KalmanFilter
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from Nets.net_v1 import Modules
from functions.Kalman_Filter import KalmanBoxTracker
from functions.outer_funcs import create_box_annotations, format_sample_result
from functions.inner_funcs import greedy_match, mahalanobis_distance, associate_detections_to_trackers, \
    expand_and_concat

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Modules()
model.to(device)


def hungarian_matching(estims, trues):
    cost_matrix = np.zeros((len(estims), len(trues)))

    for i, estim in enumerate(estims):
        for j, true in enumerate(trues):
            gt_center = np.array(true[:2], dtype=float)
            distance = np.linalg.norm(estim[:2] - gt_center)
            cost_matrix[i, j] = distance

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind, col_ind


def construct_K_matrix(distance_matrix, dets, curr_gts, trks, prev_gts, threshold=2):
    # dist_m[0,0] is dets[0] and trks[0]
    # dist_m[0,1] is dets[0] and trks[1]
    # etc

    K = torch.ones_like(distance_matrix)
    d_idx, d_gt_idx = hungarian_matching(dets, curr_gts)
    t_idx, t_gt_idx = hungarian_matching(trks, prev_gts)

    for d, gt_d in zip(d_idx, d_gt_idx):
        for t, gt_t in zip(t_idx, t_gt_idx):

            center_det = dets[d][:2]
            gt_center_det = np.array(curr_gts[gt_d][:2], dtype=float)

            center_trk = trks[t][:2]
            gt_center_trk = np.array(prev_gts[gt_t][:2], dtype=float)

            dist_1 = np.linalg.norm(center_det - gt_center_det)
            dist_2 = np.linalg.norm(center_trk - gt_center_trk)

            if curr_gts[gt_d][7] == prev_gts[gt_t][7] and dist_1 <= threshold and dist_2 <= threshold:
                K[d, t] = 0

    return K


def hungarian_matching_remake(estims, trues):
    cost_matrix = np.linalg.norm(estims[:, :2, None] - np.array(trues)[:, :2].T, axis=1)
    return linear_sum_assignment(cost_matrix)


def construct_K_matrix_remake(distance_matrix, dets, curr_gts, trks, prev_gts, threshold=2):
    K = torch.ones_like(distance_matrix)

    d_idx, d_gt_idx = hungarian_matching_remake(dets, curr_gts)
    t_idx, t_gt_idx = hungarian_matching_remake(trks, prev_gts)

    curr_gts_array = np.array(curr_gts)
    prev_gts_array = np.array(prev_gts)

    for i, (d, gt_d) in enumerate(zip(d_idx, d_gt_idx)):
        center_det = dets[d][:2]
        gt_center_det = curr_gts_array[gt_d][:2].astype(float)
        dist_1 = np.linalg.norm(center_det - gt_center_det)

        for j, (t, gt_t) in enumerate(zip(t_idx, t_gt_idx)):
            center_trk = trks[t][:2]
            gt_center_trk = prev_gts_array[gt_t][:2].astype(float)
            dist_2 = np.linalg.norm(center_trk - gt_center_trk)

            if (curr_gts_array[gt_d][7] == prev_gts_array[gt_t][7] and
                    dist_1 <= threshold and dist_2 <= threshold):
                K[d, t] = 0

    return K


def distance_matrix_gen(d_t_map, mah_metric, dets, curr_gts, trks, prev_gts, state):
    D = mah_metric
    D_mod = D
    K = D

    return D_mod, D, K


class AB3DMOT(object):
    def __init__(self, max_age=2, min_hits=3, tracking_name='car', state='train'):
        """
        observation:
                          [0, 1, 2, 3, 4, 5, 6]
          before reorder: [h, w, l, x, y, z, rot_y]
          after reorder:  [x, y, z, rot_y, l, w, h]
          our order [x, y, z, l, w, h, rot_z, dx, dy]
          after [x, y, z, rot_z, l, w, h, dx, dy]
        state:
          [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
        """

        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.tracking_name = tracking_name
        self.state = state
        self.features = []
        self.my_order = [0, 1, 2, 6, 4, 3, 5]  # x, y, z, rot_z, l, w, h
        self.my_order_back = [0, 1, 2, 5, 4, 6, 3]

    def update(self, dets_all, match_threshold):
        """
        Params:
          dets_all: dict
            dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
            info: a array of other info for each det
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        {'dets': np.array(dets[tracking_name]),
                                        'pcbs': np.array(pcbs[tracking_name]),
                                        'fvecs': np.array(fvecs[tracking_name]),
                                        'cam_vecs': np.array(cam_vecs[tracking_name]),
                                        }
        """

        dets, pcbs, feats, cam_vecs, info, curr_gts, prev_gts = (dets_all['dets'],
                                                                 dets_all['pcbs'],
                                                                 dets_all['fvecs'],
                                                                 dets_all['cam_vecs'],
                                                                 dets_all['info'],
                                                                 dets_all['current_gts'],
                                                                 dets_all['previous_gts'])

        print_debug = False

        dets = dets[:, self.my_order]

        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 7))  # N x 7

        if self.features:
            trks_feats = torch.stack([feat for feat in self.features], dim=0)
        else:
            trks_feats = torch.empty((0, 0)).to(device)

        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict().reshape((-1, 1))

            trk[:] = pos[:7].flatten()

            if (np.any(np.isnan(pos))):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)
            self.features.pop(t)

        if print_debug:
            for trk_tmp in self.trackers:
                print('trk_tmp.id: ', trk_tmp.id)

        trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in
                  self.trackers]

        D_mah = mahalanobis_distance(dets=dets, trks=trks, trks_S=trks_S)
        D_mah_module = torch.tensor(D_mah).to(device)

        # print('\n', self.tracking_name, '\n', dets.shape, '\ntrks', trks.shape)
        det_feats = model.G1(feats, pcbs, cam_vecs)

        det_trk_matrix = expand_and_concat(det_feats, trks_feats)

        D_module, D, K = distance_matrix_gen(det_trk_matrix, D_mah, dets, curr_gts, trks, prev_gts, self.state)

        matched_indexes = greedy_match(D)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(matched_indices=matched_indexes,
                                                                                   distance_matrix=D,
                                                                                   dets=dets, trks=trks,
                                                                                   mahalanobis_threshold=match_threshold)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                trk.update(dets[d, :][0], info[d, :][0])

                detection_score = info[d, :][0][-1]

                trk.track_score = detection_score

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            detection_score = info[i][-1]
            track_score = detection_score
            trk = KalmanBoxTracker(dets[i, :], info[i, :], track_score, self.tracking_name)

            self.trackers.append(trk)
            self.features.append(det_feats[i].detach())  # addition  VERY IMPORTANT THAT I ADDED THE DETACH

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.get_state()  # bbox location

            d = d[self.my_order_back]

            if ((trk.time_since_update < self.max_age) and (
                    trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1], trk.info[:-1], [trk.track_score])).reshape(1,
                                                                                                       -1))  # +1 as MOT benchmark requires positive
            i -= 1

            # remove dead tracks
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(i)

                self.features.pop(i)

        if (len(ret) > 0):
            return np.concatenate(ret), D_module, K  # x, y, z, theta, l, w, h, ID, other info, confidence

        return np.empty((0, 15 + 7)), D_module, K


def track_nuscenes(match_threshold=11):

    parser = argparse.ArgumentParser(description="TrainVal G2 with lidar and camera detected characteristics")
    parser.add_argument('--state', type=str, default='val',
                        help='train or val the G2 module')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='NuScenes dataset version')
    parser.add_argument('--data_root', type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval',
                        help='Root directory of the NuScenes dataset')
    parser.add_argument('--detection_file', type=str,
                        default="/home/ktsiakas/thesis_new/2D_FEATURE_EXTRACTOR/mrcnn_val.pkl",
                        help='Path to detections, train split for train - val split for inference')
    parser.add_argument('--model_state', type=str, default='garbage.pth',
                        help='destination and name for model')
    parser.add_argument('--output_path', type=str, default='mah_only.json',
                        help='destination for tracking results (leave blank if val state)')

    args = parser.parse_args()
    state = args.state
    detection_file = args.detection_file
    data_root = args.data_root
    version = args.version
    model_state = args.model_state
    output_path = args.output_path
    split_name = state

    if state == 'train':

        for param in model.g1.parameters():
            param.requires_grad = True
        for param in model.g2.parameters():
            param.requires_grad = True
        for param in model.g3.parameters():
            param.requires_grad = False
        for param in model.g4.parameters():
            param.requires_grad = False

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        # PROSOXH: GIA EMAS TO POSITIVE EINAI TO 0
        # TORA MAUAINOYME POS NA MHN KANOYME LATHOS
        # AN KANAME ME TO 0 OS POSITIVE UA KANAME TRAIN POS NA EINAI GIA NA EIMASTE SOSTOI
        criterion = nn.BCEWithLogitsLoss()  # built-in sigmoid for stability - gitai oxi crossentropyloss
        EPOCHS = 10

    else:
        for param in model.g1.parameters():
            param.requires_grad = False
        for param in model.g2.parameters():
            param.requires_grad = False
        for param in model.g3.parameters():
            param.requires_grad = False
        for param in model.g4.parameters():
            param.requires_grad = False

        EPOCHS = 1

    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    split_names = get_scenes_of_split(split_name=split_name, nusc=nusc)
    split_scenes_tokens_list = [nusc.field2token(table_name='scene', field='name', query=scene_name)
                                for scene_name in split_names]
    split_scenes_tokens = [item for sublist in split_scenes_tokens_list for item in sublist]
    split_scenes_tokens = set(split_scenes_tokens)

    results = {}

    with open(detection_file, 'rb') as f:
        all_results = pickle.load(f)

    for epoch in range(EPOCHS):

        total_time = 0.0
        total_frames = 0

        print('epoch', epoch)
        processed_scene_tokens = set()

        for sample, sample_data in tqdm(all_results.items()):

            try:
                scene_token = nusc.get('sample', sample)['scene_token']
            except:
                continue

            if scene_token in processed_scene_tokens or scene_token not in split_scenes_tokens:
                continue

            first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
            current_sample_token = first_sample_token

            mot_trackers = {tracking_name: AB3DMOT(tracking_name=tracking_name, state=state) for
                            tracking_name in NUSCENES_TRACKING_NAMES}

            u = 0

            prev_ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            prev_trackers = {}

            while current_sample_token != '':

                # print(current_sample_token)
                # if u == 38:
                #     exit()

                current_ground_truths = create_box_annotations(current_sample_token, nusc)

                results[current_sample_token] = []

                dets = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                fvecs = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                pcbs = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                cam_vecs = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                info = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

                try:
                    ts = all_results[current_sample_token]
                except:
                    break

                for i, item in enumerate(all_results[current_sample_token]):
                    for name in NUSCENES_TRACKING_NAMES:
                        for dets_outputs in item[name]:
                            dets[name].append(dets_outputs['box'])
                            pcbs[name].append(dets_outputs['point_cloud_features'])
                            fvecs[name].append(dets_outputs['feature_vector'])
                            cam_vecs[name].append(dets_outputs['camera_onehot_vector'])
                            info[name].append(dets_outputs['pred_score'])

                dets_all = {tracking_name: {'dets': np.array(dets[tracking_name]),
                                            'pcbs': np.array(pcbs[tracking_name]),
                                            'fvecs': np.array(fvecs[tracking_name]),
                                            'cam_vecs': np.array(cam_vecs[tracking_name]),
                                            'info': np.array(info[tracking_name]),
                                            'current_gts': np.array(current_ground_truths[tracking_name]),
                                            'previous_gts': np.array(prev_ground_truths[tracking_name])
                                            }
                            for tracking_name in NUSCENES_TRACKING_NAMES}

                total_frames += 1
                start_time = time.time()

                for tracking_name in NUSCENES_TRACKING_NAMES:
                    if dets_all[tracking_name]['dets'].shape[0] > 0:

                        trackers, D, K = mot_trackers[tracking_name].update(dets_all[tracking_name], match_threshold)

                        if state == 'train':
                            if D is None:
                                continue

                            optimizer.zero_grad()

                            loss = criterion(D, K)

                            loss.backward(retain_graph=False)  # sounds right

                            # for name, param in model.named_parameters():
                            #     if param.requires_grad:
                            #         if param.grad is not None:
                            #             print(f"Gradients of parameter '{name}' exist. Parameter was updated.")
                            #         else:
                            #             print(f"No gradients for parameter '{name}'. Parameter was not updated.")

                            optimizer.step()

                        else:
                            # (N, 9)
                            # (h, w, l, x, y, z, rot_y), tracking_id, tracking_score
                            for i in range(trackers.shape[0]):
                                sample_result, prev_trackers = format_sample_result(current_sample_token, tracking_name,
                                                                                    trackers[i],
                                                                                    prev_trackers)
                                results[current_sample_token].append(sample_result)

                cycle_time = time.time() - start_time
                total_time += cycle_time

                prev_ground_truths = copy.deepcopy(current_ground_truths)
                current_sample_token = nusc.get('sample', current_sample_token)['next']

            # left while loop and mark this scene as processed
            processed_scene_tokens.add(scene_token)

        print("Total learning took: %.3f for %d frames or %.1f FPS" % (
            total_time, total_frames, total_frames / total_time))

    # save model after epochs
    # if state == 'train':
    #     torch.save(model.state_dict(), model_state)

    # save tracking results after inference
    else:
        meta = {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False
        }

        output_data = {'meta': meta, 'results': results}
        with open(output_path, 'w') as outfile:
            json.dump(output_data, outfile)


if __name__ == '__main__':
    print('track nuscenes')
    track_nuscenes()
