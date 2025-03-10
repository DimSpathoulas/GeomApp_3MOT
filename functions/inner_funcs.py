from __future__ import print_function
from tqdm import tqdm
import os.path
import sys
import time
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
