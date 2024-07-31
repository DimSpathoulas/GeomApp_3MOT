# We implemented our method on top of AB3DMOT's KITTI tracking open-source code

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

NUSCENES_TRACKING_NAMES = [
    'car'
]


def create_box_annotations(sample_token, nusc):
    ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

    # if ann_token is not None:  # not really needed

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


def rot_z(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def create_box(bbox3d_input):
    # [x, y, z, w, l, h, rot]

    bbox3d = copy.copy(bbox3d_input)

    R = rot_z(bbox3d[6])

    w = bbox3d[3]
    l = bbox3d[4]
    h = bbox3d[5]

    # 3d bounding box corners
    x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    y_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    z_corners = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]

    return np.transpose(corners_3d)


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox3D, info, track_score=None, tracking_name='car'):
        """
        Initialises a tracker using initial bounding box.
        """

        self.kf = KalmanFilter(dim_x=11, dim_z=7)
        self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # state transition matrix
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

        covariance = Covariance()
        self.kf.P = covariance.P[tracking_name]
        self.kf.Q = covariance.Q[tracking_name]
        self.kf.R = covariance.R[tracking_name]

        self.kf.x[:7] = bbox3D.reshape((7, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 1  # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.info = info  # other info
        self.track_score = track_score
        self.tracking_name = tracking_name

    def update(self, bbox3D, info):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the fist time

        ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(
                new_theta - predicted_theta) < np.pi * 3 / 2.0:  # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        #########################

        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        self.info = info

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7,))


def angle_in_range(angle):
    '''
    Input angle: -2pi ~ 2pi
    Output angle: -pi ~ pi
    '''
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle


def diff_orientation_correction(det, trk):
    '''
    return the angle diff = det - trk
    if angle diff > 90 or < -90, rotate trk and update the angle diff
    '''
    diff = det - trk
    diff = angle_in_range(diff)
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
    diff = angle_in_range(diff)
    return diff


def greedy_match(distance_matrix):  # EDO PITHANON ANETA KANOYME KATI KALYTERO
    '''
    Find the one-to-one matching using greedy algorithm choosing small distance
    distance_matrix: (num_detections, num_tracks)
    '''

    matched_indices = []

    if distance_matrix.shape[0] == 0:
        return np.empty((0, 2))

    num_detections, num_tracks = distance_matrix.shape
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)

    detection_id_matches_to_tracking_id = [-1] * num_detections
    tracking_id_matches_to_detection_id = [-1] * num_tracks

    for sort_i in range(index_2d.shape[0]):

        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])

        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[
            detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id

            matched_indices.append([detection_id, tracking_id])

    matched_indices = np.array(matched_indices)

    return matched_indices


def mahalanobis_distance(dets=None, trks=None, trks_S=None, print_debug=False):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    dets: N x 7
    trks: M x 7
    trks_S: N x 7 x 7

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """

    if (len(trks) == 0):
        return np.empty((0, 0))

    distance_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)

    assert (dets is not None)
    assert (trks is not None)
    assert (trks_S is not None)

    if print_debug:
        print('dets.shape: ', dets.shape)
        print('dets: ', dets)
        print('trks.shape: ', trks.shape)
        print('trks: ', trks)
        print('trks_S: ', trks_S)
        S_inv = [np.linalg.inv(S_tmp) for S_tmp in trks_S]  # 7 x 7
        S_inv_diag = [S_inv_tmp.diagonal() for S_inv_tmp in S_inv]  # 7
        print('S_inv_diag: ', S_inv_diag)

    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            S_inv = np.linalg.inv(trks_S[t])  # 7 x 7
            diff = np.expand_dims(dets[d] - trks[t], axis=1)  # 7 x 1
            # manual reversed angle by 180 when diff > 90 or < -90 degree
            corrected_angle_diff = diff_orientation_correction(dets[d][3], trks[t][3])
            diff[3] = corrected_angle_diff
            distance_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])

    return distance_matrix


def associate_detections_to_trackers(matched_indices, distance_matrix, dets, trks, mahalanobis_threshold):
    if matched_indices.shape[0] == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(dets)),
                np.empty((0, 2), dtype=int))

    print_debug = False

    unmatched_detections = []
    for d, det in enumerate(dets):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []

    for t, trk in enumerate(trks):
        if len(matched_indices) == 0 or (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        match = True
        if distance_matrix[m[0], m[1]] > mahalanobis_threshold:
            match = False

        if not match:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    if print_debug:
        print('matches: ', matches)
        print('unmatched_detections: ', unmatched_detections)
        print('unmatched_trackers: ', unmatched_trackers)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def expand_and_concat(det_feats, trk_feats):
    if trk_feats.shape[0] == 0:
        return torch.empty((det_feats.shape[0], trk_feats.shape[0],
                            2 * det_feats.shape[1],
                            det_feats.shape[2], det_feats.shape[3])).to(device)

    det_feats_expanded = det_feats.unsqueeze(1)  # Shape: (N, 1, C, H, W)
    trk_feats_expanded = trk_feats.unsqueeze(0)  # Shape: (1, M, C, H, W)

    # Concatenate along the channel dimension
    matrix = torch.cat((det_feats_expanded.expand(-1, trk_feats.shape[0], -1, -1, -1),
                        trk_feats_expanded.expand(det_feats.shape[0], -1, -1, -1, -1)),
                       dim=2)

    return matrix


class Modules(nn.Module):

    def __init__(self):
        super(Modules, self).__init__()

        # FEATURE FUSION MODULE
        self.g1 = nn.Sequential(
            nn.Linear(1024 + 6, 1536),
            nn.ReLU(),
            nn.Linear(1536, 4608)
        )

        # DISTANCE COMBINATION MODULE 1
        self.g2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # DISTANCE COMBINATION MODULE 2
        self.g3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # TRACK INITIALIZATION MODULE
        self.g4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def G1(self, F2D, F3D, cam_onehot_vector):
        F2D = torch.tensor(F2D).to(device)
        F3D = torch.tensor(F3D).to(device)
        cam_onehot_vector = torch.tensor(cam_onehot_vector).to(device)

        fused = torch.cat((F2D, cam_onehot_vector), dim=1)

        fused = self.g1(fused)

        fused = fused.reshape(fused.shape[0], 512, 3, 3)
        fused = fused + F3D

        return fused

    def G2(self, x):
        ds, ts, channels, height, width = x.shape
        x_reshaped = x.view(-1, channels, height, width)

        result = self.g2(x_reshaped)

        y = result.view(ds, ts)

        return y

    def G3(self, x):
        ds, ts, channels, height, width = x.shape
        x_reshaped = x.view(-1, channels, height, width)

        result = self.g3(x_reshaped)

        result_reshaped = result.view(ds, ts, -1)
        a = result_reshaped[:, :, 0]
        b = result_reshaped[:, :, 1]

        return a, b

    def G4(self, x):
        x = x.to(device)
        score = self.g4(x)
        return score


# INITIALIZE AS GLOBAL SO THAT WE CAN CALL THESE EVERYWHERE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Modules()
model.to(device)

# torch.autograd.set_detect_anomaly(True)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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

    K = np.ones_like(distance_matrix)
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
    K = np.ones_like(distance_matrix)

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


def distance_matrix(d_t_map, mah_metric, dets, curr_gts, trks, prev_gts, state):
    D_mod = None
    D = np.empty((0, 0))
    K = np.empty((0, 0))

    if d_t_map.shape[1] > 0:
        D_mod = model.G2(d_t_map)
        D_mod = D_mod + mah_metric
        D = D_mod.detach().cpu().numpy()

        if state == 'train':
            K = construct_K_matrix(distance_matrix=D, dets=dets, curr_gts=curr_gts, trks=trks, prev_gts=prev_gts)

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
        self.my_order_back = [0, 1, 2, 4, 5, 6, 3]

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

        D_module, D, K = distance_matrix(det_trk_matrix, D_mah_module, dets, curr_gts, trks, prev_gts, self.state)

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

        # FALSE DETECTION HANDLING
        return np.empty((0, 15 + 7)), D_module, K


def format_sample_result(sample_token, tracking_name, tracker, prev_trackers):
    '''
    Input:
      tracker: (9): [h, w, l, x, y, z, rot_y], tracking_id, tracking_score
    Output:
    sample_result {
      "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
      "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
      "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
      "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
      "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
      "tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
      "tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                         Note that the tracking_name cannot change throughout a track.
      "tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                         We average over frame level scores to compute the track level score.
                                         The score is used to determine positive and negative tracks via thresholding.
    }
    '''

    current_id = tracker[7]
    current_x = tracker[0]
    current_y = tracker[1]

    if current_id in prev_trackers:
        prev_x, prev_y = prev_trackers[current_id]
        velocity_x = (current_x - prev_x) / 0.5
        velocity_y = (current_y - prev_y) / 0.5
    else:
        velocity_x = 0
        velocity_y = 0

    prev_trackers[current_id] = (current_x, current_y)

    rotation = Quaternion(axis=[0, 0, 1], angle=tracker[6]).elements
    sample_result = {
        'sample_token': sample_token,
        'translation': [tracker[0], tracker[1], tracker[2]],
        'size': [tracker[3], tracker[4], tracker[5]],
        'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]],
        'velocity': [velocity_x, velocity_y],
        'tracking_id': str(int(tracker[7])),
        'tracking_name': tracking_name,
        'tracking_score': tracker[8]
    }

    return sample_result


def track_nuscenes(match_threshold=11):
    split_name = 'train'

    parser = argparse.ArgumentParser(description="TrainVal G2 with lidar and camera detected characteristics")
    parser.add_argument('--state', type=str, default='train',
                        help='train or val the G2 module')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        help='NuScenes dataset version')
    parser.add_argument('--data_root', type=str, default='../../data/nuscenes/v1.0-mini',
                        help='Root directory of the NuScenes dataset')
    parser.add_argument('--detection_file', type=str,
                        default="../../data/tracking_input/sample_mini_train_v4.pkl",
                        help='Path to detections, train split for train - val split for inference')
    parser.add_argument('--model_state', type=str, default='g2_trained_model_test_1.pth',
                        help='destination and name for model')
    parser.add_argument('--output_path', type=str, default='',
                        help='destination for tracking results (leave blank if val state)')

    args = parser.parse_args()
    state = args.state
    detection_file = args.detection_file
    data_root = args.data_root
    version = args.version
    model_state = args.model_state
    output_path = args.output_path

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

                            # D = D.to(device)
                            K = torch.tensor(K).to(device)
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
                            prev_trackers = {}
                            # (N, 9)
                            # (h, w, l, x, y, z, rot_y), tracking_id, tracking_score
                            for i in range(trackers.shape[0]):
                                sample_result = format_sample_result(current_sample_token, tracking_name, trackers[i],
                                                                     prev_trackers)
                                results[current_sample_token].append(sample_result)

                cycle_time = time.time() - start_time
                total_time += cycle_time

                u = u + 1

                prev_ground_truths = copy.deepcopy(current_ground_truths)
                current_sample_token = nusc.get('sample', current_sample_token)['next']

            # left while loop and mark this scene as processed
            processed_scene_tokens.add(scene_token)

        print("Total learning took: %.3f for %d frames or %.1f FPS" % (
            total_time, total_frames, total_frames / total_time))

    # save model after epochs
    if state == 'train':
        torch.save(model.state_dict(), model_state)

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
