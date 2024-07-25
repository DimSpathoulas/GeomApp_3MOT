# We implemented our method on top of AB3DMOT's KITTI tracking open-source code

from __future__ import print_function
from tqdm import tqdm
import os.path
import sys
import time
import pickle
import copy

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
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
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
    matrix = torch.empty((det_feats.shape[0], trk_feats.shape[0],
                       2 * det_feats.shape[1],
                       det_feats.shape[2], det_feats.shape[3]))

    if trk_feats.shape[0] == 0:
        return matrix

    for i in range(det_feats.shape[0]):
        for j in range(trk_feats.shape[0]):
            conc_features = torch.concatenate((det_feats[i], trk_feats[j]))
            matrix[i, j] = conc_features

    return matrix


class Modules(nn.Module):

    def __init__(self):
        super(Modules, self).__init__()

        # FEATURE FUSION MODULE
        self.g1 = nn.Sequential(  # this is G1
            nn.Linear(1030, 1536),
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

        fused = fused.view(fused.shape[0], 512, 3, 3)

        fused = fused + F3D

        return fused

    def G2(self, x):

        x = x.to(device)
        y = torch.empty(x.shape[0], x.shape[1])

        for i, d in enumerate(x):
            ms = torch.cat([dt.unsqueeze(0) for dt in d], dim=0)
            result = self.g2(ms)
            y[i, :] = result.squeeze()

        return y

    def G3(self, x):

        x = x.to(device)
        a = torch.empty(x.shape[0], x.shape[1])
        b = torch.empty(x.shape[0], x.shape[1])

        for i, d in enumerate(x):
            ms = torch.cat([dt.unsqueeze(0) for dt in d], dim=0)
            result = self.g3(ms)
            a[i, :] = result[:, 0].squeeze()
            b[i, :] = result[:, 0].squeeze()

        return a, b

    def G4(self, x):

        x = x.to(device)
        score = self.g4(x)
        return score


# INITIALIZE AS GLOBAL SO THAT WE CAN CALL THESE EVERYWHERE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Modules()
model.to(device)
model.load_state_dict(torch.load('g2_trained_model_dummy.pth', map_location=device))

for param in model.g1.parameters():
    param.requires_grad = True
for param in model.g2.parameters():
    param.requires_grad = False
for param in model.g3.parameters():
    param.requires_grad = False
for param in model.g4.parameters():
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
criterion = nn.BCELoss()
EPOCHS = 10


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


def retrieve_pairs(K):

    pos = []
    neg = []

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if K[i, j] == 0:
                pos.append((i, j))
            else:
                neg.append((i, j))

    return pos, neg


def G3_net_loss(T=11, C_contr=6, C_pos=3, C_neg=3, distance_matrix=None, K=None):

    pos, neg = retrieve_pairs(K)
    L_contr = torch.tensor(0.).to(device)
    L_pos = torch.tensor(0.).to(device)
    L_neg = torch.tensor(0.).to(device)
    zero = torch.tensor(0.).to(device)
    for i, j in pos:
        for ii, jj in neg:
            L_contr += torch.max(zero, C_contr - (distance_matrix[i][j] - distance_matrix[ii][jj]))
    L_contr = L_contr / (len(pos) * len(neg))

    for i, j in pos:
        L_pos += torch.max(zero, C_pos - (T - distance_matrix[i][j]))
    L_pos = L_pos / len(pos)

    for i, j in neg:
        L_neg += torch.max(zero, C_neg - (T - distance_matrix[i][j]))
    L_neg = L_neg / len(neg)

    L_coef = L_contr + L_pos + L_neg  # to kano minimize ayto??

    return L_coef


def construct_C_matrix(estims, trues):

    C = np.ones((estims.shape[0], 1))

    for i, estim in enumerate(estims):
        for j, true in enumerate(trues):
            gt_center = np.array(true[:2], dtype=float)
            distance = np.linalg.norm(estim[:2] - gt_center)
            if distance <= 0.1:
                C[i] = 1
                break

    C = torch.tensor(C, dtype=torch.float32).to(device)

    return C


class AB3DMOT(object):
    def __init__(self, max_age=2, min_hits=3, tracking_name='car'):
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

        self.features = []
        self.my_order = [0, 1, 2, 6, 4, 3, 5]  # x, y, z, rot_z, l, w, h
        self.my_order_back = [0, 1, 2, 4, 5, 6, 3]  # xreiazetai kan ayto?

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
            trks_feats = torch.empty((0, 0))

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

        # print('\n', self.tracking_name, '\n', dets.shape, '\ntrks', trks.shape)

        det_feats = model.G1(feats, pcbs, cam_vecs).to(device)

        det_trk_matrix = expand_and_concat(det_feats, trks_feats)

        # print(det_feats.shape, trks_feats.shape, det_trk_matrix.shape, self.tracking_name)

        D_feat_module = None
        K = np.empty((0, 0))
        D = np.empty((0, 0))

        if det_trk_matrix.shape[1] > 0:

            D_feat_module = model.G2(det_trk_matrix).to(device)

            a_mod, b_mod = model.G3(det_trk_matrix)
            a_mod = a_mod.to(device)
            b_mod = b_mod.to(device)

            D_mah_module = torch.tensor(D_mah).to(device)
            point_five = torch.tensor(0.5).to(device)  # ayto prepei na meinei edo ???
            D_mod = D_mah_module + (a_mod * (D_feat_module - (point_five + b_mod)))  # final D
            D = D_mod.detach().cpu().numpy()

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
        scores = torch.zeros(dets.shape[0], 1, dtype=torch.float32).to(device)
        C = torch.zeros(dets.shape[0], 1, dtype=torch.float32).to(device)
        ind = 0

        if unmatched_dets.shape[0] > 0:
            ind = 1
            mask = torch.ones(scores.size(0), dtype=torch.bool).to(device)
            mask[unmatched_dets] = False
            unmatched_feats = det_feats[unmatched_dets]
            scores[unmatched_dets] = model.G4(unmatched_feats)
            C = construct_C_matrix(dets, curr_gts)

        for i in unmatched_dets:
            if scores[i] > 0.5:
                detection_score = info[i][-1]
                track_score = detection_score
                trk = KalmanBoxTracker(dets[i, :], info[i, :], track_score, self.tracking_name)

                self.trackers.append(trk)
                self.features.append(det_feats[i])

        if ind == 1:
            scores[mask] = 1

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.get_state()  # bbox location

            d = d[self.my_order_back]

            if ((trk.time_since_update < self.max_age) and (
                    trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1], trk.info[:-1], [trk.track_score])).reshape(1,-1))  # +1 as MOT benchmark requires positive
            i -= 1

            # remove dead tracks
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(i)
                self.features.pop(i)

        if (len(ret) > 0):

            if D_feat_module is None:
                return np.concatenate(ret), scores, C, ind

            return np.concatenate(ret), scores, C, ind  # x, y, z, theta, l, w, h, ID, other info, confidence

        if D_feat_module is None:
            return np.empty((0, 15 + 7)), scores, C, ind

        # FALSE DETECTION HANDLING
        ind = 0
        return np.empty((0, 15 + 7)), scores, C, ind


def track_nuscenes(data_split='train', match_threshold=11, save_root='/.results/01'):
    '''
    submission {
      "meta": {
          "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
          "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
          "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
          "use_map":      <bool>  -- Whether this submission uses map data as an input.
          "use_external": <bool>  -- Whether this submission uses external data as an input.
      },
      "results": {
          sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
      }
    }

    '''

    save_dir = os.path.join(save_root, data_split);
    mkdir_if_missing(save_dir)
    if 'train' in data_split:
        split_name = 'mini_train'
        detection_file = '../../data/tracking_input/sample_mini_train_v6.pkl'
        data_root = '../../data/nuscenes/v1.0-mini'
        version = 'v1.0-mini'

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

        for sample, sample_data in tqdm(all_results.items()):  # fainetai san 0% alla metra scenes oxi samples (mallon)

            try:
                scene_token = nusc.get('sample', sample)['scene_token']
            except:
                continue

            if scene_token in processed_scene_tokens or scene_token not in split_scenes_tokens:
                continue

            first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
            # print('sc', scene_token)
            current_sample_token = first_sample_token

            mot_trackers = {tracking_name: AB3DMOT(tracking_name=tracking_name) for
                            tracking_name in NUSCENES_TRACKING_NAMES}

            u = 0

            prev_ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

            while current_sample_token != '':

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
                        trackers, D, K, ind = mot_trackers[tracking_name].update(dets_all[tracking_name], match_threshold)

                        if ind == 0:
                            continue
                            
                        optimizer.zero_grad()
                        
                        loss = criterion(D, K)
                        loss.backward(retain_graph=True)

                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                if param.grad is not None:
                                    print(f"Gradients of parameter '{name}' exist. Parameter was updated.")
                                else:
                                    print(f"No gradients for parameter '{name}'. Parameter was not updated.")

                        optimizer.step()

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
    torch.save(model.state_dict(), 'g2_trained_model_dummy.pth')


if __name__ == '__main__':
    print('track nuscenes')
    track_nuscenes()
