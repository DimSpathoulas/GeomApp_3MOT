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

from covariance import Covariance
from utils import mkdir_if_missing

from filterpy.kalman import KalmanFilter
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from Nets.net_v3 import Feature_Fusion, Distance_Combination_Stage_2, Distance_Combination_Stage_1, Track_Init
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


def distance_matrix_gen(d_t_map, mah_metric, DCS1, DCS2):
    if d_t_map.shape[1] == 0:
        return np.empty((0, 0))

    D_mod = DCS1(d_t_map)
    D_mod = D_mod + mah_metric
    D = D_mod.detach().cpu().numpy()

    return D




# def distance_matrix_gen(d_t_map, mah_metric, DCS1, DCS2):
#     if d_t_map.shape[1] == 0:
#         return np.empty((0, 0)), None, None

#     K = None

#     D_mod = DCS1(d_t_map)
#     a_mod, b_mod = DCS2(d_t_map)
#     point_five = torch.tensor(0.5).to(device)
#     D_mod = mah_metric + (a_mod * (D_mod - (point_five + b_mod)))
#     D = D_mod.detach().cpu().numpy()

#     return D, D_mod, K


def construct_C_matrix(estims, trues):
    C = np.zeros((estims.shape[0], 1))

    for i, estim in enumerate(estims):
        for j, true in enumerate(trues):
            gt_center = np.array(true[:2], dtype=float)
            distance = np.linalg.norm(estim[:2] - gt_center)
            if distance <= 1.0:
                C[i] = 1.0
                break

    C = torch.tensor(C, dtype=torch.float32).to(device)

    return C


def construct_C_matrix_remake(estims, trues):
    estims_xy = estims[:, :2]
    trues_xy = np.array(trues[:, :2], dtype=float)

    distances = np.linalg.norm(estims_xy[:, np.newaxis] - trues_xy, axis=2)

    min_distances = np.min(distances, axis=1)

    C = (min_distances <= 2.0).astype(float)

    C = torch.tensor(C, dtype=torch.float32).to(device).unsqueeze(1)

    return C


class AB3DMOT(object):
    def __init__(self, max_age=2, min_hits=3, tracking_name='car', state=0, FF=None, DCS1=None, DCS2=None, TRIN=None):
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
        self.my_order = [0, 1, 2, 6, 3, 4, 5]  # x, y, z, rot_z, l, w, h
        self.my_order_back = [0, 1, 2, 4, 5, 6, 3]

        self.state = state
        self.FF = FF
        self.DCS1 = DCS1
        self.DCS2 = DCS2
        self.TRIN = TRIN

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
        det_feats = self.FF(feats, pcbs, cam_vecs)

        det_trk_matrix = expand_and_concat(det_feats, trks_feats)

        D = distance_matrix_gen(det_trk_matrix, D_mah_module, self.DCS1, self.DCS2)

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
        C = P_l = None

        if unmatched_dets.shape[0] > 0:

            P = torch.zeros(dets.shape[0], 1).to(device)
            unmatched_feats = det_feats[unmatched_dets]
            P[unmatched_dets] = self.TRIN(unmatched_feats)
            if self.state < 10 and curr_gts.shape[0] != 0:
                C = construct_C_matrix_remake(dets[unmatched_dets], curr_gts)
                P_l = P[unmatched_dets]

        for i in unmatched_dets:
            if P[i] > 0.5:
                detection_score = info[i][-1]
                track_score = detection_score
                trk = KalmanBoxTracker(dets[i, :], info[i, :], track_score, self.tracking_name)
                self.trackers.append(trk)
                self.features.append(det_feats[i].detach())

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
            return np.concatenate(ret), P_l, C  # x, y, z, theta, l, w, h, ID, other info, confidence

        return np.empty((0, 15 + 7)), P_l, C



def save_models_combined(G1, G2, G3, G4, path):
    combined_state = {
        'G1': G1.state_dict(),
        'G2': G2.state_dict(),
        'G3': G3.state_dict(),
        'G4': G4.state_dict(),
    }
    torch.save(combined_state, path)


def load_models_combined(FF, DCS1, DCS2, path):
    combined_state = torch.load(path)
    FF.load_state_dict(combined_state['G1'])
    DCS1.load_state_dict(combined_state['G2'])
    # DCS2.load_state_dict(combined_state['G3'])
    return FF, DCS1, DCS2


def track_nuscenes(match_threshold=11):
    parser = argparse.ArgumentParser(description="TrainVal G2 with lidar and camera detected characteristics")
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='NuScenes dataset version')
    parser.add_argument('--data_root', type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval',
                        help='Root directory of the NuScenes dataset')
    parser.add_argument('--dets_train', type=str,
                        default="/home/ktsiakas/thesis_new/2D_FEATURE_EXTRACTOR/mrcnn_val_2.pkl",
                        help='Path to detections, train split for train - val split for inference')
    parser.add_argument('--dets_val', type=str,
                        default="/home/ktsiakas/thesis_new/2D_FEATURE_EXTRACTOR/mrcnn_val_2.pkl",
                        help='Path to detections, train split for train - val split for inference')
    parser.add_argument('--model_state', type=str, default='model_per_sample_2.pth',
                        help='destination and name for model')
    parser.add_argument('--output_path', type=str, default='g4_output.json',
                        help='destination for tracking results (leave blank if val state)')

    args = parser.parse_args()
    dets_train = args.dets_train
    dets_val = args.dets_val
    data_root = args.data_root
    version = args.version
    model_state = args.model_state
    output_path = args.output_path

    FF = Feature_Fusion().to(device)
    DCS1 = Distance_Combination_Stage_1().to(device)
    DCS2 = Distance_Combination_Stage_2().to(device)
    TRIN = Track_Init().to(device).to(device)
    FF, DCS1, _ = load_models_combined(FF, DCS1, DCS2, model_state)

    DCS1.eval()
    for param in DCS1.parameters():
        param.requires_grad = False
        
    DCS2.eval()
    for param in DCS2.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(list(FF.parameters()) + list(TRIN.parameters()), lr=0.001)
    criterion = nn.BCELoss()
    EPOCHS = 11
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    # split_names = get_scenes_of_split(split_name=state, nusc=nusc)
    # split_scenes_tokens_list = [nusc.field2token(table_name='scene', field='name', query=scene_name)
    #                             for scene_name in split_names]
    # split_scenes_tokens = [item for sublist in split_scenes_tokens_list for item in sublist]
    # split_scenes_tokens = set(split_scenes_tokens)

    results = {}

    with open(dets_train, 'rb') as f:
        all_results = pickle.load(f)

    total_time = 0.0
    total_frames = 0
    epoch = 10
    for epoch in range(EPOCHS):

        print('epoch', epoch + 1)

        if epoch == 10:
            with open(dets_val, 'rb') as f:
                all_results = pickle.load(f)

            FF.eval()
            for param in FF.parameters():
                param.requires_grad = False

            TRIN.eval()
            for param in TRIN.parameters():
                param.requires_grad = False

        processed_scene_tokens = set()

        for sample, sample_data in tqdm(all_results.items()):

            scene_token = nusc.get('sample', sample)['scene_token']

            if scene_token in processed_scene_tokens:
                continue

            first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
            current_sample_token = first_sample_token

            mot_trackers = {tracking_name: AB3DMOT(tracking_name=tracking_name, state=epoch, FF=FF, DCS1=DCS1, DCS2=DCS2, TRIN=TRIN)
                            for tracking_name in NUSCENES_TRACKING_NAMES}

            prev_ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            prev_trackers = {}

            while current_sample_token != '':

                current_ground_truths = create_box_annotations(current_sample_token, nusc)

                results[current_sample_token] = []

                dets = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                fvecs = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                pcbs = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                cam_vecs = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                info = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

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

                D_list = []
                K_list = []
                optimizer.zero_grad()

                for tracking_name in NUSCENES_TRACKING_NAMES:
                    if dets_all[tracking_name]['dets'].shape[0] > 0:

                        trackers, D, K = mot_trackers[tracking_name].update(dets_all[tracking_name], match_threshold)

                        if epoch < 10:
                            if D is None or K is None:
                                continue

                            D_list.append(D)
                            K_list.append(K)

                        if epoch == 10:
                            # (N, 9)
                            # (h, w, l, x, y, z, rot_y), tracking_id, tracking_score
                            for i in range(trackers.shape[0]):
                                sample_result, prev_trackers = format_sample_result(current_sample_token, tracking_name,
                                                                                    trackers[i],
                                                                                    prev_trackers)
                                results[current_sample_token].append(sample_result)

                if D_list:

                    total_loss = torch.tensor(0.).to(device)
                    for D, K in zip(D_list, K_list):

                        loss = criterion(D,K)
                        total_loss += loss

                    total_loss.backward(retain_graph=False)

                    optimizer.step()

                cycle_time = time.time() - start_time
                total_time += cycle_time

                prev_ground_truths = copy.deepcopy(current_ground_truths)
                current_sample_token = nusc.get('sample', current_sample_token)['next']

            # left while loop and mark this scene as processed
            processed_scene_tokens.add(scene_token)

        print("Total learning took: %.3f for %d frames or %.1f FPS" % (
            total_time, total_frames, total_frames / total_time))

    # save model after epochs
    # save_models_combined(FF, DCS1, DCS2, model_state)

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
    print('G4 TRAIN-VAL')
    track_nuscenes()
