from __future__ import print_function
import time

from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from torch import nn
from collections import defaultdict
from functions.Kalman_Filter import KalmanBoxTracker
from functions.inner_funcs import greedy_match, mahalanobis_distance, associate_detections_to_trackers
from itertools import product
import torch.nn.init as init
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss_g4(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='none'):
        super(FocalLoss_g4, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute binary cross-entropy loss
        # print(inputs, targets)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        
        # Compute pt (predicted probability for true class)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        
        # Apply focal loss scaling factor
        loss = ( self.alpha * ( (1 - pt) ** self.gamma ) )* bce_loss
        
        loss_s = loss.sum()
        # print(loss_s)
        return loss_s




def compute_margin_loss(P, C, margin=0.30, lambda_margin=1):
    """
    Computes the margin loss to encourage a gap between positive and negative predictions.

    Args:
        P (torch.Tensor): Predicted probabilities (N x 1).
        C (torch.Tensor): Corresponding ground truth labels (N x 1).
        margin (float): Desired margin between positive and negative predictions.
        lambda_margin (float): Weight for the margin loss term.

    Returns:
        torch.Tensor: Margin loss.
    """

    total = 0

    C_pos = 0.65
    C_neg = 0.35
    # Extract positive and negative indices
    pos_indices = (C == 1).nonzero(as_tuple=True)[0]
    neg_indices = (C == 0).nonzero(as_tuple=True)[0]

    # # Ensure we have valid positive and negative indices
    # if len(pos_indices) > 0 and len(neg_indices) > 0:
    #     P_pos = P[pos_indices]  # Positive predictions (n_pos x 1)
    #     P_neg = P[neg_indices]  # Negative predictions (n_neg x 1)

    #     # Compute pairwise differences using broadcasting
    #     diff = P_pos.unsqueeze(1) - P_neg.unsqueeze(0)  # (n_pos x n_neg)

    #     # Compute margin loss
    #     margin_loss = torch.clamp(margin - diff, min=0).sum() / ( len(pos_indices) * len(neg_indices))
    #     total = total + margin_loss
    
    if len(pos_indices) > 0:
        
        P_pos = P[pos_indices] 
        margin_pos = torch.clamp(C_pos - P_pos, min=0).sum() / len(pos_indices)

        total = total + margin_pos

    if len(neg_indices) > 0:
        
        P_neg = P[neg_indices]
        margin_neg = torch.clamp(P_neg - C_neg, min=0).sum() / len(neg_indices)

        total = total + margin_neg

    return total


class TrackerNN(nn.Module):
    def __init__(self):
        super(TrackerNN, self).__init__()

        # Neural network components - INITIALIZED ONCE
        self.G1 = nn.Sequential(
            nn.Linear(1024 + 6, out_features=1536),
            nn.ReLU(),
            nn.Linear(1536, out_features=4 * 1152),
        )

        # self.G4 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=0, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=0, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(16, 1),
        #     # nn.ReLU(),
        #     # nn.Linear(32, 1),
        #     nn.Sigmoid()
        # )

        # self.G4 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=0, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1),
        #     nn.Sigmoid()
        # )

        # self.G4 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=0, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(32, 1),
        #     # nn.ReLU(),
        #     # nn.Linear(32, 1),
        #     nn.Sigmoid()
        # )

        self.G4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # WEIGHT INIT
        # self.apply(self.initialize_weights)

        # INIT STATES - WILL BE CLEARED AFTER EACH SCENE
        self.tracking_states = {}

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                init.constant_(m.bias, 1.0)  # Bias close to 1 after sigmoid
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                init.constant_(m.bias, 1.0)  # Bias close to 1 after sigmoid

    # SUB-FORWARDS
    def feature_fusion(self, F2D, F3D, cam_onehot_vector):

        # # L2 normalize across spatial dimensions
        lidar_feats = torch.nn.functional.normalize(F3D.view(F3D.shape[0], -1), p=2, dim=1)
        lidar_feats = lidar_feats.view(lidar_feats.shape[0], F3D.shape[1], F3D.shape[2], F3D.shape[3])

        cam_feats = torch.cat((F2D, cam_onehot_vector), dim=1)
        cam_feats = self.G1(cam_feats)

        cam_feats = torch.nn.functional.normalize(cam_feats.view(cam_feats.shape[0], -1), p=2, dim=1)
        cam_feats = cam_feats.view(cam_feats.shape[0], F3D.shape[1], F3D.shape[2], F3D.shape[3])
        
        # Fuse the normalized features
        fused = cam_feats + lidar_feats
        
        # normalize the fused output again
        fused = torch.nn.functional.normalize(fused.view(fused.shape[0], -1), p=2, dim=1)
        fused = fused.view(fused.shape[0], F3D.shape[1], F3D.shape[2], F3D.shape[3])

        return fused
    

    def track_initialization(self, x):

        y = self.G4(x).float()
        
        return y


    def clear_tracking_states(self):
        """Clear all tracking states."""
        self.tracking_states.clear()


    # Initialize AB3DMOT attributes - CALL FOR EACH NEW SCENE
    def reinit_ab3dmot(self, tracking_name, max_age=2, min_hits=3, training=False, criterion=None, epoch=0):
        self.tracking_states[tracking_name] = {
            'max_age': max_age,
            'min_hits': min_hits,
            'trackers': [],
            'track_init_thresh': 0.5,
            'frame_count': 0,
            'mahanalobis_thresh': 11,
            'features': [],
            'order': [0, 1, 2, 6, 3, 4, 5],  # x, y, z, rot_z, l, w, h
            'order_back': [0, 1, 2, 4, 5, 6, 3],
        }
        self.training = training
        self.criterion = criterion
        self.epoch = epoch

    def construct_C_matrix(self, estimates, truths, distance_threshold=2.0):
        """
        Compute proximity scores between detections and ground truths.
        Returns 1 if detection is close to any ground truth, 0 otherwise.
        
        Args:
            estimates: detection coordinates [N, dim]
            truths: ground truth coordinates [M, dim]
            distance_threshold: maximum distance to consider a detection close to ground truth
            
        Returns:
            torch.Tensor: Binary proximity scores [N, 1]
        """
        # Extract and convert xy coordinates
        estimates_xy = torch.from_numpy(estimates[:, :2].astype(np.float32)).to(device=device)
        truths_xy = torch.from_numpy(truths[:, :2].astype(np.float32)).to(device=device)
        
        # Compute pairwise distances between all detections and ground truths
        distances = torch.norm(estimates_xy[:, None] - truths_xy, dim=2)
        
        # Find minimum distance to any ground truth for each detection
        min_distances = torch.min(distances, dim=1).values
        
        # Convert to binary scores based on threshold
        C = (min_distances <= distance_threshold).float().unsqueeze(1)
        
        return C
    

    def construct_C_matrix_2(self, estimates, truths, distance_threshold=2.0):
        """
        Compute proximity scores between detections and ground truths.
        Ensures each ground truth is used only once.
        
        Args:
            estimates: detection coordinates [N, dim]
            truths: ground truth coordinates [M, dim]
            distance_threshold: maximum distance to consider a detection close to ground truth
        
        Returns:
            torch.Tensor: Binary proximity scores [N, 1]
        """
        # Extract and convert xy coordinates
        estimates_xy = torch.from_numpy(estimates[:, :2].astype(np.float32)).to(device=device)
        truths_xy = torch.from_numpy(truths[:, :2].astype(np.float32)).to(device=device)
        
        # Compute pairwise distances between all detections and ground truths
        distances = torch.norm(estimates_xy[:, None] - truths_xy, dim=2)
        
        # Initialize binary scores tensor
        C = torch.zeros(estimates.shape[0], 1, dtype=torch.float32).to(device=device)
        
        # Track used ground truths
        used_truths = set()
        
        # Find the closest detection for each ground truth
        for i in range(len(truths)):
            # Find valid detections close to this ground truth
            valid_detections = torch.where(distances[:, i] <= distance_threshold)[0]
            
            if len(valid_detections) > 0:
                # Find the closest detection
                closest_detection_idx = valid_detections[torch.argmin(distances[valid_detections, i])]
                
                # Mark this detection as valid if the ground truth hasn't been used
                if i not in used_truths:
                    C[closest_detection_idx] = 1.0
                    used_truths.add(i)
        
        return C


    def compute_pairwise_cosine_similarity(self, det_feats, trk_feats):
        if trk_feats.shape[1] == 0:
            return torch.empty(0, 0).to(device=device)
        # Flatten spatial dimensions
        m = det_feats.shape[0]
        n = trk_feats.shape[0]
        
        # # Reshape to (m, 512*3*3)
        det_feats = det_feats.view(m, -1)
        trk_feats = trk_feats.view(n, -1)
        
        det_feats = torch.nn.functional.normalize(det_feats, p=2, dim=1)
        trk_feats = torch.nn.functional.normalize(trk_feats, p=2, dim=1)
        
        # similarity matrix
        similarity_matrix = torch.mm(det_feats, trk_feats.t())
        
        return similarity_matrix


    def forward(self, dets_all, tracking_name):

        tracking_state = self.tracking_states[tracking_name]

        tracking_state['frame_count'] += 1  # NEW FRAME

        gamma = 0.3
        
        # LOAD CURRENT INFORMATION
        dets, pcbs, feats, cam_vecs, info, curr_gts, prev_gts = (
            dets_all['dets'], dets_all['pcbs'], dets_all['fvecs'],
            dets_all['cam_vecs'], dets_all['info'], dets_all['current_gts'],
            dets_all['previous_gts']
        )

        loss = None

        # LOAD FEATURES
        F2D = torch.tensor(feats, dtype=torch.float32, requires_grad=True).to(device)
        F3D = torch.tensor(pcbs, dtype=torch.float32, requires_grad=True).to(device)
        cam_onehot_vector = torch.tensor(cam_vecs, dtype=torch.float32, requires_grad=True).to(device)

        # LOAD AND ORDER DETS BASED ON KALMAN FILTER
        dets = dets[:, tracking_state['order']]
 
        # LOAD TRACK FEATURES IF EXISTANT
        if tracking_state['features']:
            trks_feats = torch.stack([feat for feat in tracking_state['features']], dim=0)
        else:
            trks_feats = torch.empty((0, 0)).to(device)

        # LOAD CORRESPONDING TRACKS IF EXISTANT
        trks = np.zeros((len(tracking_state['trackers']), 7))  # N x 7
        
        # PREDICTION STEP
        for t, trk in enumerate(trks):
            pos = tracking_state['trackers'][t].predict().reshape((-1, 1))
            trk[:] = pos[:7].flatten()

        # COMPUTE S MATRIX
        trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R
                  for tracker in tracking_state['trackers']]

        # COMPUTE MAHALANOBIS DISTANCE
        D = mahalanobis_distance(dets=dets, trks=trks, trks_S=trks_S)

        # FEATURE FUSION (EXISTS IN EVERY STATE)
        det_feats = self.feature_fusion(F2D, F3D, cam_onehot_vector)

        # GREEDY MATCH
        matched_indices = greedy_match(D)

        # # RETRIEVE MATCHED AND UNMATCHED
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            matched_indices,
            distance_matrix=D,
            dets=dets,
            trks=trks,
            mahalanobis_threshold= tracking_state['mahanalobis_thresh'])  # change based on val or train !!!!!!!
                                                                        
        # UPDATE MATCHED TRACKERS BASED ON PAIRED DETECTIONS
        for t, trk in enumerate(tracking_state['trackers']):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                trk.update(dets[d, :][0], info[d, :][0])
                trk.track_score = info[d, :][0][-1]
                blended_feature = det_feats[d].detach() * gamma + trks_feats[t] * (1.0- gamma)
                tracking_state['features'][t] = blended_feature.squeeze(0)





        # for i in unmatched_dets:  # a scalar of index
        #     detection_score = info[i][-1]
        #     track_score = detection_score
        #     trk = KalmanBoxTracker(dets[i, :], info[i, :], track_score, tracking_name)
        #     tracking_state['trackers'].append(trk)
        #     tracking_state['features'].append(det_feats[i].detach())



        if unmatched_dets.shape[0] > 0:
            P = torch.zeros(dets.shape[0], 1, device=device)
            unmatched_feats = det_feats[unmatched_dets]
            P[unmatched_dets] = self.track_initialization(unmatched_feats)

            # if curr_gts.shape[0] != 0:
            #     C = self.construct_C_matrix_2(dets[unmatched_dets], curr_gts)
            #     C1 = self.construct_C_matrix(dets[unmatched_dets], curr_gts)
            #     print(P[unmatched_dets], C, C1)

            for idx in unmatched_dets:
                if P[idx] > tracking_state['track_init_thresh']:

                    new_track = KalmanBoxTracker(dets[idx], info[idx], info[idx, -1], tracking_name)
                    tracking_state['trackers'].append(new_track)
                    tracking_state['features'].append(det_feats[idx].detach())

            if curr_gts.shape[0] != 0  and self.training == True:
                # Sample balanced positives and negatives
                unmatched = torch.tensor(unmatched_dets).to(device)

                C = self.construct_C_matrix_2(dets[unmatched_dets], curr_gts)
                n_matches= max((C == 1).sum(), 1) 
                n_non_matches = max((C == 0).sum(), 1)
                pos_weight = ( n_non_matches/n_matches ).item()
                focal_loss = FocalLoss_g4(pos_weight, gamma=2.0)
                # loss_mar = compute_margin_loss(P[unmatched], C) ## add the los backkkk !!!!!!!!!!!
                loss = focal_loss(P[unmatched], C)
                loss = loss




        # TRACK MANAGEMENT
        ret = []
        for i, trk in reversed(list(enumerate(tracking_state['trackers']))):
            d = trk.get_state()[tracking_state['order_back']]
            if (trk.time_since_update < tracking_state['max_age'] and
                    (trk.hits >= tracking_state['min_hits'] or tracking_state['frame_count'] <= tracking_state[
                        'min_hits'])):
                ret.append(np.concatenate((d, [trk.id + 1], trk.info[:-1], [trk.track_score])).reshape(1, -1))

            if trk.time_since_update >= tracking_state['max_age']:
                tracking_state['trackers'].pop(i)
                tracking_state['features'].pop(i)

        # RETURN CURRENT TRACKS AND LOSS
        if len(ret) > 0:
            return np.concatenate(ret), loss
        return np.empty((0, 15 + 7)), loss

