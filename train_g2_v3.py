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
    matrix = np.empty((det_feats.shape[0], trk_feats.shape[0],
                       2 * det_feats.shape[1],
                       det_feats.shape[2], det_feats.shape[3]))

    if trk_feats.shape[0] == 0:
        return matrix

    for i in range(det_feats.shape[0]):
        for j in range(trk_feats.shape[0]):
            conc_features = np.concatenate((det_feats[i], trk_feats[j]))
            matrix[i, j] = conc_features

    return matrix


class Modules(nn.Module):

    def __init__(self):  # edo ua valoyme to config
        super(Modules, self).__init__()

        # FEATURE FUSION MODULE
        self.g1 = nn.Sequential(  # this is G1
            nn.Linear(1024 + 6, 1536),
            nn.ReLU(),
            nn.Linear(1536, 4608),
            # nn.ReLU()  # thelei pali relu?
        )

        # DISTANCE COMBINATION MODULE 1
        self.g2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.Linear(128, 1)
        )

        # # POIO APO TA DYO ?????
        # self.g2 = nn.Sequential(
        #     nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),  # newly added
        #     nn.Linear(256 * 3 * 3, 256),
        #     nn.Linear(256, 128),
        #     nn.Linear(128, 1)
        # )

        # DISTANCE COMBINATION MODULE 2
        self.g3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 2)
        )

        # TRACK INITIALIZATION MODULE
        self.g4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 128),
            # nn.ReLU(),
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

        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.empty(x.shape[0], x.shape[1])

        for i, d in enumerate(x):
            ms = torch.cat([dt.unsqueeze(0) for dt in d], dim=0)
            result = self.g2(ms)
            y[i, :] = result.squeeze()

        return y

    def G3(self, x):

        x = torch.tensor(x, dtype=torch.float32)
        a = torch.empty(x.shape[0], x.shape[1])
        b = torch.empty(x.shape[0], x.shape[1])

        for i, d in enumerate(x):
            ms = torch.cat([dt.unsqueeze(0) for dt in d], dim=0)
            result = self.g3(ms)
            a[i, :] = result[:, 0].squeeze()
            b[i, :] = result[:, 0].squeeze()

        return a, b

    def G4(self, x):
        score = self.G4(x)
        return score


# INITIALIZE AS GLOBAL SO THAT WE CAN CALL THESE EVERYWHERE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Modules()
model.to(device)
# model.load_state_dict(torch.load('g2_trained_model.pth'))  # for save
# DEN NOMIZO OTI EXEI NOHMA AYTO
# for param in model.g1.parameters(): this shouldnt work
#     param.requires_grad = False
# for param in model.g2.parameters(): this shouldnt work
#     param.requires_grad = False
for param in model.g3.parameters():
    param.requires_grad = False
for param in model.g4.parameters():
    param.requires_grad = False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
# criterion = nn.BCELoss()
# PROSOXH: GIA EMAS TO POSITIVE EINAI TO 0
# TORA MAUAINOYME POS NA MHN KANOYME LATHOS
# AN KANAME ME TO 0 OS POSITIVE UA KANAME TRAIN POS NA EINAI GIA NA EIMASTE SOSTOI
criterion = nn.BCEWithLogitsLoss()  # built-in sigmoid for stability - gitai oxi crossentropyloss
EPOCHS = 2


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

    # mhpos yparxei pio aplos tropos?
    K = np.ones_like(distance_matrix)
    d_idx, d_gt_idx = hungarian_matching(dets, curr_gts)
    t_idx, t_gt_idx = hungarian_matching(trks, prev_gts)

    # print(dets, '\n', curr_gts)
    # print(d_idx, d_gt_idx)

    # print('\n\n', trks, '\n', prev_gts)
    # print(t_idx, t_gt_idx)

    # SE MERIKA EINAI ARKETA EKTOS TO ORIENTATION KAI TA DIMS AN KAI EXEI KANEI KALO DETECT TA CENTERS
    # DEN JERO POS NA TO LYSO AYTO EKTOS APO TO NA AYJHSO TO THRESH STO CENTERPOINT
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

    # print(K)

    return K


# under construction !!!!!!!!!!!!!!!! this is for g3
def PNP_NET_loss(T=11, C_contr=6, C_pos=3, C_neg=3, dets=None, trks=None, distance_matrix=None):

    # DEN EINAI PINAKES EINAI MIA TIMH 1X1
    # OSO PIO KONTA STO 0 TOSO TO KALYTERO H OSO PIO MEGALO TOSO TO KALYTERO ???
    # 1. POS VRISKO TA di kai dj - mhpos einai oi times toy D ekei poy yparxei match kai ekei poy den yparxei ?
    # dld apo ton K matrix ??? TO K EINAI H MASKA POY PERNA APO TON D. OSTOSO DEN JERO AN EXEI NOHMA GIA TRAINING AYTO
    # TI SHMAINEI AGGREGATE DISTANCE SE EMAS ?
    # 3. POS KANO TRAIN ME TI LOSS FUNCTION
    L_contr = torch.tensor(0.0)
    zero = torch.tensor(0.0)
    for i in distance_matrix.shape[0]:
        for j in distance_matrix.shape[1]:
            L_contr[i, j] += torch.max(zero, C_contr - (distance_matrix[i] - distance_matrix[j]))

    L_pos = torch.tensor(0.0)
    for i in range(distance_matrix.size(0)):
        L_pos += torch.max(zero, C_pos - (T - distance_matrix[i]))

    L_neg = torch.tensor(0.0)
    for j in range(distance_matrix.size(1)):
        L_neg += torch.max(zero, C_neg - (distance_matrix[j] - T))

    # Overall training loss
    L_coef = L_contr + L_pos + L_neg


class AB3DMOT(object):
    def __init__(self, max_age=2, min_hits=3, tracking_name='car', modeL=None):
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

        # self.modeL = modeL

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

        # print('\n\n\n\n', self.tracking_name, prev_gts, '\ncurr gts', curr_gts)

        print_debug = False

        dets = dets[:, self.my_order]

        self.frame_count += 1

        # !!!!!!!!!! EDO YPHRXE ENA PERIERGO DEBUGGING ME ENA SCENE SYGKEKRIMENO - MALLON MPOREI KAI SAN DEIGMA

        trks = np.zeros((len(self.trackers), 7))  # N x 7
        trks_feats = np.array(self.features)

        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict().reshape((-1, 1))

            trk[:] = pos[:7].flatten()

            if (np.any(np.isnan(pos))):
                to_del.append(t)

        # giati yparxei kan ayto
        # trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

            self.features.pop(t)

        if print_debug:
            for trk_tmp in self.trackers:
                print('trk_tmp.id: ', trk_tmp.id)

        trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in
                  self.trackers]

        det_feats_module = model.G1(feats, pcbs, cam_vecs)
        det_feats = det_feats_module.detach().cpu().numpy()

        # print(f'\n\nframe count for {self.tracking_name}', self.frame_count)
        # print(f'dets {dets.shape}', dets)
        # print(f'trks {trks.shape}', trks)
        # print(f'det_Feats {det_feats}')
        # print(f'trks_feats {trks_feats.shape}', trks_feats)

        D_mah = mahalanobis_distance(dets=dets, trks=trks, trks_S=trks_S)

        det_trk_matrix = expand_and_concat(det_feats, trks_feats)

        # print(det_feats.shape, trks_feats.shape, det_trk_matrix.shape, self.tracking_name)

        # vres ena pio omorfo tropo gia olo ayto
        D_feat = np.empty((0, 0))
        D_feat_module = None
        K = np.zeros_like(D_feat)
        D = np.zeros_like(D_feat)

        if det_trk_matrix.shape[1] > 0:
            # NOTES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # -1. TORA TO GRAPH ETSI OPOS TO EXO KANEI EINAI SOSTO??? GIATI YPARXEI ENA DISCONTINUE STO G1 KAI G2
            # mallon den einai - line 492
            # 0. SE KAPOIA POLY SYGKEKRIMENA DETS EINAI POLY EKTOS TO ORIENTATION KAI TA DIMS
            # ETSI AN KAI YPARXEI SYSXETISH DEN GINONTAI TRACK - TAYTOXRONA EGO TO EXO KANEI NA TA PERNAEI STO K
            # GIA TO LEARNING TOY D_FEAT
            # O MONOS TROPOS NA TO APOFYGO EINAI NA AYJHSO TO PRED_SCORE TOY CENTERPOINT ( MALLON )
            # 1. EMEIS EXOYME TO W SAN L KAI AYTOI TO L SAN W.... TI KANOYME LATHOS ME TO BOX?
            # 2. EINAI TO G1 KAI G2 NETS SOSTA ?? EIDIKA TO PADDING = 1 KAI OI RELU
            # 3. POY NA VALO TO OPTIMIZER.STEP ?? GIA KATHE NAME GIA KATHE SAMPLE H GIA KATHE SCENE - TRACKING NAME LEO
            # 4. AN EXO ETSI TO NETWORK MHPOS META EINAI PIO DYSKOLO NA KANO TRAIN TO ALLO ME TA G3 G4 ???
            # 5. TO BCELOSS MALLON MESA TOY STREIVEI TA VARY PROS TO 1
            # EMEIS THELOYME NA PHGAINEI OSO PIO KONTA STO 0 GINETAI OXI STO 1
            # TORA MATHAINEI POS NA MHN KANEI LATHOS
            # EMEIS THELOUME NA MATHAINEI NA KANEI SOSTO DET TRACK ASSOCIATION
            # 6. TO BCELOOS DEXETAI MATRIXES ?? H MHPOS KALYTERA NA DOYME TO nn.CrossEntropyloss()
            D_feat_module = model.G2(det_trk_matrix)
            D_feat = D_feat_module.detach().cpu().numpy()
            D = D_feat + D_mah

            K = construct_K_matrix(distance_matrix=D, dets=dets, curr_gts=curr_gts, trks=trks, prev_gts=prev_gts)

        # D = D_mah + (a * (D_feat - (0.5 + b))) final D

        # edo D_mah h D ???
        matched_indexes = greedy_match(D)
        # EDO D_mah h D ???
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

        # HERE WE ADD THE G4 MODULE - MALLON
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            detection_score = info[i][-1]
            track_score = detection_score
            trk = KalmanBoxTracker(dets[i, :], info[i, :], track_score, self.tracking_name)

            self.trackers.append(trk)
            self.features.append(det_feats[i])  # addition

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.get_state()  # bbox location

            d = d[self.my_order_back]  # ayto giati yparxei kan

            # what is that :0
            if ((trk.time_since_update < self.max_age) and (
                    trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1], trk.info[:-1], [trk.track_score])).reshape(1,
                                                                                                       -1))  # +1 as MOT benchmark requires positive
            i -= 1

            # remove dead tracks
            # EDO UA FTIAJOYME TO MODULE G5 POY EXO STO MYALO MOY AN EXOYME KAIRO
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(i)

                self.features.pop(i)

        if (len(ret) > 0):

            if D_feat_module is None:
                return np.concatenate(ret), D, K

            return np.concatenate(ret), D_feat_module, K  # x, y, z, theta, l, w, h, ID, other info, confidence

        if D_feat_module is None:
            return np.empty((0, 15 + 7)), D, K

        # # we should never be in here
        # # edo ua mpoyme otan yparxei megalh apoklhsh se dims h orientation me ta actual data
        print(dets, '\n', curr_gts, '\n\n', trks, '\n', prev_gts, K)
        return np.empty((0, 15 + 7)), D_feat_module, K


def format_sample_result(sample_token, tracking_name, tracker):
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
    rotation = Quaternion(axis=[0, 0, 1], angle=tracker[6]).elements
    sample_result = {
        'sample_token': sample_token,
        'translation': [tracker[3], tracker[4], tracker[5]],
        'size': [tracker[1], tracker[2], tracker[0]],
        'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]],
        'velocity': [0, 0],
        'tracking_id': str(int(tracker[7])),
        'tracking_name': tracking_name,
        'tracking_score': tracker[8]
    }

    return sample_result


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
        output_path = os.path.join(save_dir, 'results_train_probabilistic_tracking.json')

    elif 'val' in data_split:
        detection_file = '../../data/tracking_input/sample_mini_train_v1.pkl'
        data_root = '../../data/nuscenes/v1.0-mini'
        version = 'v1.0-mini'
        output_path = os.path.join(save_dir, 'results_val_probabilistic_tracking.json')

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
            print('sc', scene_token)
            current_sample_token = first_sample_token

            mot_trackers = {tracking_name: AB3DMOT(tracking_name=tracking_name) for
                            tracking_name in NUSCENES_TRACKING_NAMES}

            u = 0

            prev_ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

            while current_sample_token != '':

                print('\n', current_sample_token)

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

                # print('\n\n\n\n', current_sample_token)

                for tracking_name in NUSCENES_TRACKING_NAMES:
                    if dets_all[tracking_name]['dets'].shape[0] > 0:

                        trackers, D, K = mot_trackers[tracking_name].update(dets_all[tracking_name], match_threshold)

                        # (N, 9)
                        # (h, w, l, x, y, z, rot_y), tracking_id, tracking_score
                        # print('trackers: ', trackers)
                        # for i in range(trackers.shape[0]):
                        #     sample_result = format_sample_result(current_sample_token, tracking_name, trackers[i])
                        #     results[current_sample_token].append(sample_result)

                        # # TRAIN D WITH K WITH GLOBALLY INITIALIZED LOSS
                        # # AND STEP OPTIMIZE AFTER EACH SCENE OR SAMPLE
                        # print(type(D), tracking_name)  # TO CENTER DISTANCE EINAI KALO ALLA TO ORIEN KAI DIMS OXI
                        # ETSI TO D_MAH TO KOVEI ALLA AYTO OXI ( TOYLAXISTON AKOMA )

                        if D.shape[0] == 0:
                            continue
                        D = D.to(device)
                        K = torch.tensor(K).to(device)

                        # mhpos prepei na ta kano flatten prota ????
                        loss = criterion(D, K)  # den jero an einai sosto ayto etsi....
                        loss.backward()

                        # # EDO ???
                        optimizer.zero_grad()
                        optimizer.step()

                    # # EDO ???
                    # optimizer.zero_grad()
                    # optimizer.step()

                cycle_time = time.time() - start_time
                total_time += cycle_time

                u = u + 1

                # # EDO ???
                # optimizer.zero_grad()
                # optimizer.step()

                prev_ground_truths = copy.deepcopy(current_ground_truths)
                current_sample_token = nusc.get('sample', current_sample_token)['next']

            # left while loop and mark this scene as processed
            processed_scene_tokens.add(scene_token)

        # # finished tracking all scenes, write output data
        # output_data = {'meta': meta, 'results': results}
        # with open(output_path, 'w') as outfile:
        #     json.dump(output_data, outfile)

        print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

    # save model after epochs
    torch.save(model.state_dict(), 'g2_trained_model_dummy.pth')


if __name__ == '__main__':
    print('track nuscenes')
    track_nuscenes()
