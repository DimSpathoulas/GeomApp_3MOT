'''
EDO FTIAXNOYME TA GROUND TRUTHS X, Y, Z, , W, L, H, ANGLE, INSTANCE TOKEN
INSTANCE TOKEN = MONADIKO GIA KATHE GT TRACK

KAI TO FEATURE VECTOR TOY KATHE TRACK
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

    return sample_result, prev_trackers
