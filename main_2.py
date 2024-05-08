# We implemented our method on top of AB3DMOT's KITTI tracking open-source code

from __future__ import print_function

import json
import numpy as np
import os.path
import sys
import time

from filterpy.kalman import KalmanFilter
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from pyquaternion import Quaternion
from tqdm import tqdm

from covariance import Covariance
from utils import mkdir_if_missing


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


def greedy_match(distance_matrix):
    '''
    Find the one-to-one matching using greedy allgorithm choosing small distance
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


def old_associate_detections_to_trackers(dets=None, trks=None, trks_S=None,
                                         mahalanobis_threshold=0.1, print_debug=False):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    dets: N x 7
    trks: M x 7
    trks_S: N x 7 x 7

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trks) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty((0, 8, 3), dtype=int)

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

    matched_indices = greedy_match(distance_matrix)  # na do ayto ti kanei akrivos

    if print_debug:
        print('distance_matrix.shape: ', distance_matrix.shape)
        print('distance_matrix: ', distance_matrix)
        print('matched_indices: ', matched_indices)

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


class AB3DMOT(object):
    def __init__(self, max_age=2, min_hits=3, tracking_name='car'):
        """
        observation:
                          [0, 1, 2, 3, 4, 5, 6]
          before reorder: [h, w, l, x, y, z, rot_y]
          after reorder:  [x, y, z, rot_y, l, w, h]
          our order [x, y, z, l, w, h, rot_z]
          after [x, y, z, rot_z, l, w, h]
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
        self.my_order = [0, 1, 2, 6, 3, 4, 5]
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
        """
        dets_o, info, det_feats, gts = dets_all['dets'], dets_all['info'], dets_all['feats'], dets_all['gts']

        print_debug = False
        dets = dets_o[:, self.my_order]

        self.frame_count += 1

        # !!!!!!!!!! EDO YPHRXE ENA PERIERGO DEBUGGING ME ENA SCENE SYGKEKRIMENO - MALLON MPOREI KAI SAN DEIGMA

        trks = np.zeros((len(self.trackers), 7))  # N x 7

        trks_feats = self.features  # addition

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

            self.features.pop(t)  # addition

        print(f'\n\nframe count for {self.tracking_name}', self.frame_count)
        print(f'dets {dets.shape}', dets)  # kai ayto me ti order einai
        print(f'gts {gts.shape}', gts)
        # print(f'feats {det_feats.shape}', det_feats)
        # print(f'trks {trks.shape}', trks)  # ma me ti order einai ayto
        # print(f'trks_feats {trks_feats}')

        if print_debug:
            for trk_tmp in self.trackers:
                print('trk_tmp.id: ', trk_tmp.id)

        trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in
                  self.trackers]

        D_mah = mahalanobis_distance(dets=dets, trks=trks, trks_S=trks_S)

        matched_ind = greedy_match(D_mah)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(matched_indices=matched_ind,
                                                                                   distance_matrix=D_mah,
                                                                                   dets=dets, trks=trks,
                                                                                   mahalanobis_threshold=match_threshold)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                trk.update(dets[d, :][0], info[d, :][0])

                detection_score = info[d, :][0][-1]  # den exo krathsei to score

                trk.track_score = detection_score

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
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(i)

                self.features.pop(i)  # addition

        if (len(ret) > 0):
            return np.concatenate(ret)  # x, y, z, theta, l, w, h, ID, other info, confidence
        return np.empty((0, 15 + 7))

        # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets=dets,
        #                                                                            trks=trks, trks_S=trks_S,
        #                                                                            mahalanobis_threshold=match_threshold,
        #                                                                            print_debug=print_debug)
        #
        # print(matched, unmatched_dets, unmatched_trks)
        # # update matched trackers with assigned detections
        # for t, trk in enumerate(self.trackers):
        #     if t not in unmatched_trks:
        #         d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
        #         trk.update(dets[d, :][0], info[d, :][0])
        #
        #         detection_score = info[d, :][0][-1]  # den exo krathsei to score
        #         trk.track_score = detection_score
        #
        # # create and initialise new trackers for unmatched detections
        # for i in unmatched_dets:  # a scalar of index
        #     detection_score = info[i][-1]
        #     track_score = detection_score
        #     trk = KalmanBoxTracker(dets[i, :], info[i, :], track_score, self.tracking_name)
        #
        #     self.trackers.append(trk)
        #
        #     self.features.append(det_feats[i])  # addition
        #
        # i = len(self.trackers)
        #
        # for trk in reversed(self.trackers):
        #     d = trk.get_state()  # bbox location
        #
        #     d = d[self.my_order_back]  # ayto giati yparxei kan
        #
        #     # what is that :0
        #     if ((trk.time_since_update < self.max_age) and (
        #             trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
        #         ret.append(np.concatenate((d, [trk.id + 1], trk.info[:-1], [trk.track_score])).reshape(1,
        #                                                                                                -1))  # +1 as MOT benchmark requires positive
        #     i -= 1
        #
        #     # remove dead tracks
        #     if (trk.time_since_update >= self.max_age):
        #         self.trackers.pop(i)
        #
        #         self.features.pop(i)  # addition
        #
        # if (len(ret) > 0):
        #     return np.concatenate(ret)  # x, y, z, theta, l, w, h, ID, other info, confidence
        # return np.empty((0, 15 + 7))


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
                trs = np.array(ann_meta['translation'])
                q = Quaternion(ann_meta['rotation'])
                angle = q.angle if q.axis[2] > 0 else -q.angle  # ayto giati antistoixei se rot_y ???
                size = np.array(ann_meta['size'])

                gt_box = np.array([trs[0], trs[1], trs[2],
                                   size[0], size[1], size[2], angle])

                ground_truths[tracking_name].append(gt_box)

    return ground_truths


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


def track_nuscenes(data_split, covariance_id, match_distance, match_threshold, match_algorithm, save_root,
                   use_angular_velocity):
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
        detection_file = 'nuscenes_megvii_dets/megvii_train.json'
        data_root = '../../data/nuscenes/v1.0-mini'
        version = 'v1.0-mini'
        output_path = os.path.join(save_dir, 'results_train_probabilistic_tracking.json')
    elif 'val' in data_split:
        detection_file = 'nuscenes_megvii_dets/megvii_val.json'
        data_root = '../../data/nuscenes/v1.0-mini'
        version = 'v1.0-mini'
        output_path = os.path.join(save_dir, 'results_val_probabilistic_tracking.json')

    # elif 'test' in data_split:
    #     detection_file = '/juno/u/hkchiu/dataset/nuscenes_new/megvii_test.json'
    #     data_root = '/juno/u/hkchiu/dataset/nuscenes/test'
    #     version = 'v1.0-test'
    #     output_path = os.path.join(save_dir, 'results_test_probabilistic_tracking.json')

    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    results = {}

    total_time = 0.0
    total_frames = 0

    with open(detection_file) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    all_results = EvalBoxes.deserialize(data['results'], DetectionBox)
    meta = data['meta']
    print('meta: ', meta)
    print("Loaded results from {}. Found detections for {} samples."
          .format(detection_file, len(all_results.sample_tokens)))

    processed_scene_tokens = set()
    for sample_token_idx in tqdm(range(len(all_results.sample_tokens))):
        sample_token = all_results.sample_tokens[sample_token_idx]

        try:
            scene_token = nusc.get('sample', sample_token)['scene_token']
        except:
            continue

        if scene_token in processed_scene_tokens:
            continue

        first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        current_sample_token = first_sample_token

        mot_trackers = {tracking_name: AB3DMOT(tracking_name=tracking_name) for
                        tracking_name in NUSCENES_TRACKING_NAMES}

        u = 1

        while current_sample_token != '':

            if u == 3:
                exit()

            gts = create_box_annotations(current_sample_token, nusc=nusc)
            results[current_sample_token] = []
            dets = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            info = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

            dets_2 = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            feats = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

            box_counter = 1

            for box in all_results.boxes[current_sample_token]:
                if box.detection_name not in NUSCENES_TRACKING_NAMES:
                    continue
                q = Quaternion(box.rotation)
                angle = q.angle if q.axis[2] > 0 else -q.angle
                # print('box.rotation,  angle, axis: ', box.rotation, q.angle, q.axis)
                # print('box.rotation,  angle, axis: ', q.angle, q.axis)
                # [h, w, l, x, y, z, rot_y]
                # detection = np.array([
                #     box.size[2], box.size[0], box.size[1],
                #     box.translation[0], box.translation[1], box.translation[2],
                #     angle])

                detection = np.array([
                    box.translation[0], box.translation[1], box.translation[2],
                    box.size[0], box.size[1], box.size[2],  # why though !!!!!! NA TO DO META AYTO OTAN TO VALO SE MENA
                    angle])

                information = np.array([box.detection_score])
                dets[box.detection_name].append(detection)
                info[box.detection_name].append(information)

                dets_2[box.detection_name].append(detection)
                feature = np.eye(2) * box_counter
                u_column = np.full((2, 1), u)
                feature_with_u = np.hstack((feature, u_column))
                feats[box.detection_name].append(feature_with_u)
                box_counter += 1

                # ftiaje ena dummy features edo gia na deis pos mporeis na to kaneis map sta dets sosta
                # me ena mikro size gia na eimaste ok kai na ne ena vector me times to 1,1,1 to allo 2,2,2 ktl

            dets_all = {tracking_name: {'dets': np.array(dets[tracking_name]),
                                        # 'dets_2': np.array(dets_2[tracking_name]),
                                        'info': np.array(info[tracking_name]),
                                        'gts': np.array(gts[tracking_name]),
                                        'feats': np.array(feats[tracking_name])}
                        for tracking_name in NUSCENES_TRACKING_NAMES}

            total_frames += 1
            start_time = time.time()

            print('\n\n\n\n', current_sample_token)

            for tracking_name in NUSCENES_TRACKING_NAMES:
                if dets_all[tracking_name]['dets'].shape[0] > 0:  # AYTO EINAI SOSTO ????
                    trackers = mot_trackers[tracking_name].update(dets_all[tracking_name], match_threshold)
                    # (N, 9)
                    # (h, w, l, x, y, z, rot_y), tracking_id, tracking_score
                    # print('trackers: ', trackers)
                    for i in range(trackers.shape[0]):
                        sample_result = format_sample_result(current_sample_token, tracking_name, trackers[i])
                        results[current_sample_token].append(sample_result)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            # get next frame and continue the while loop
            u = u + 1
            current_sample_token = nusc.get('sample', current_sample_token)['next']

        # left while loop and mark this scene as processed
        processed_scene_tokens.add(scene_token)

    # finished tracking all scenes, write output data
    output_data = {'meta': meta, 'results': results}
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile)

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))


if __name__ == '__main__':
    if len(sys.argv) != 9:
        print(
            "Usage: python main.py data_split(train, val, test) covariance_id(0, 1, 2) match_distance(iou or m) match_threshold match_algorithm(greedy or h) use_angular_velocity(true or false) dataset save_root")
        sys.exit(1)

    data_split = sys.argv[1]
    covariance_id = int(sys.argv[2])
    match_distance = sys.argv[3]
    match_threshold = float(sys.argv[4])
    match_algorithm = sys.argv[5]
    use_angular_velocity = sys.argv[6] == 'True' or sys.argv[6] == 'true'
    dataset = sys.argv[7]
    save_root = os.path.join('./' + sys.argv[8])

    if dataset == 'kitti':
        print('track kitti not supported')
    elif dataset == 'nuscenes':
        print('track nuscenes')
        track_nuscenes(data_split, covariance_id, match_distance, match_threshold, match_algorithm, save_root,
                       use_angular_velocity)
