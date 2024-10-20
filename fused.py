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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerNorm1d(nn.Module):
    def forward(self, x):
        return nn.functional.layer_norm(x, x.shape[1:])

class TrackerNN(nn.Module):
    def __init__(self):
        super(TrackerNN, self).__init__()

        # Neural network components - INITIALIZED ONCE
        self.G1 = nn.Sequential(
            nn.Linear(1024 + 6, 1536),
            # LayerNorm1d(),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1536, 4608)
        )
        
        self.G2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=0, stride=1),
            # nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            # LayerNorm1d(),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.G3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.G4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # WEIGHT INIT
        self.apply(self.initialize_weights)
        
        # INIT STATES - WILL BE CLEARED AFTER EACH SCENE
        self.tracking_states = {}

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def clear_tracking_states(self):
        """Clear all tracking states."""
        self.tracking_states.clear()
    # Initialize AB3DMOT attributes - CALL FOR EACH NEW SCENE
    # training and state are constant for all scenes
    def reinit_ab3dmot(self, tracking_name, max_age=2, min_hits=3, training=False, state=0, criterion=None):
        self.tracking_states[tracking_name] = {
            'max_age': max_age,
            'min_hits': min_hits,
            'trackers': [],
            'frame_count': 0,
            'mahanalobis_thresh': 11,
            'track_init_thresh': 0.5,
            'features': [],
            'order': [0, 1, 2, 6, 3, 4, 5],  # x, y, z, rot_z, l, w, h
            'order_back': [0, 1, 2, 4, 5, 6, 3]
        }
        self.training = training
        self.state = state
        self.criterion = criterion

    # EXPAND AND CONCAT FOR G2 G3
    def expand_and_concat(self, det_feats, trk_feats):
        if trk_feats.shape[0] == 0:
            return torch.empty((det_feats.shape[0], trk_feats.shape[0],
                                2 * det_feats.shape[1],
                                det_feats.shape[2], det_feats.shape[3])).to(device)

        det_feats_expanded = det_feats.unsqueeze(1)  # Shape: (N, 1, C, H, W)
        trk_feats_expanded = trk_feats.unsqueeze(0)  # Shape: (1, M, C, H, W)

        # Concatenate along the channel dimension
        map = torch.cat((det_feats_expanded.expand(-1, trk_feats.shape[0], -1, -1, -1),
                         trk_feats_expanded.expand(det_feats.shape[0], -1, -1, -1, -1)),
                        dim=2)

        return map

    # USED IN construct_K_matrix
    def hungarian_matching(self, estims, trues):
        if len(estims) == 0 or len(trues) == 0:
            return [], []  # Return empty matches if either input is empty

        estims_array = np.array([e[:2] for e in estims], dtype=float)
        trues_array = np.array([t[:2] for t in trues], dtype=float)

        # Compute pairwise distances
        cost_matrix = np.linalg.norm(estims_array[:, np.newaxis] - trues_array, axis=2)

        return linear_sum_assignment(cost_matrix)

    # K GT IND MATRIX COMPUTER FOR DSC1 AND DSC2
    def construct_K_matrix(self, distance_matrix, dets, curr_gts, trks, prev_gts, threshold=2):
        
        if prev_gts.shape[0] == 0 or curr_gts.shape[0] == 0 or distance_matrix.shape[0] == 0:
            return torch.empty(0,0).to(device=device)
        
        K = torch.ones_like(distance_matrix)

        d_idx, d_gt_idx = self.hungarian_matching(dets, curr_gts)
        t_idx, t_gt_idx = self.hungarian_matching(trks, prev_gts)

        if len(d_idx) == 0 or len(t_idx) == 0:
            return torch.empty(0,0).to(device=device)
        
        dets_array = np.array([det[:2] for det in dets])
        curr_gts_array = np.array([gt[:2] for gt in curr_gts], dtype=float)
        trks_array = np.array([trk[:2] for trk in trks])
        prev_gts_array = np.array([gt[:2] for gt in prev_gts], dtype=float)

        dist_1 = np.linalg.norm(dets_array[d_idx] - curr_gts_array[d_gt_idx], axis=1)
        dist_2 = np.linalg.norm(trks_array[t_idx] - prev_gts_array[t_gt_idx], axis=1)

        curr_gts_ids = np.array([gt[7] for gt in curr_gts])
        prev_gts_ids = np.array([gt[7] for gt in prev_gts])

        for i, (d, gt_d) in enumerate(zip(d_idx, d_gt_idx)):
            for j, (t, gt_t) in enumerate(zip(t_idx, t_gt_idx)):
                if (curr_gts_ids[gt_d] == prev_gts_ids[gt_t] and
                        dist_1[i] <= threshold and
                        dist_2[j] <= threshold):
                    K[d, t] = 0

        return K

    # PART OF DCS2 COSTUM LOSS FUNCTION
    def retrieve_pairs_remake(self, K):
        pos_indices = torch.nonzero(K == 0, as_tuple=False)
        neg_indices = torch.nonzero(K != 0, as_tuple=False)

        pos = [tuple(idx) for idx in pos_indices.tolist()]
        neg = [tuple(idx) for idx in neg_indices.tolist()]
        # print('pos', pos, '\n', 'neg', neg, '\n\n', "K", K, '\n')
        return pos, neg

    # COSTUM DCS2 (G3) LOSS FUNCTION
    def Criterion(self, distance_matrix=None, K=None):
        pos, neg = self.retrieve_pairs_remake(K)

        T, C_contr, C_pos, C_neg = map(lambda x: torch.tensor(x, device=device), [11.0, 8.0, 5.0, 5.0])
        L_contr = L_pos = L_neg = torch.tensor(0., device=device)

        if pos or neg:
            pos_indices = torch.tensor(pos, device=device).T if pos else torch.empty((2, 0), device=device,
                                                                                     dtype=torch.long)
            neg_indices = torch.tensor(neg, device=device).T if neg else torch.empty((2, 0), device=device,
                                                                                     dtype=torch.long)

            pos_distances = distance_matrix[pos_indices[0], pos_indices[1]]
            neg_distances = distance_matrix[neg_indices[0], neg_indices[1]]
            # print(distance_matrix, '\n', 'pos', '\n', pos_distances, 'neg', '\n', neg_distances, '\n\n\n\n\n\n')

            if pos and neg:
                L_contr = torch.clamp(C_contr - (pos_distances.unsqueeze(1) - neg_distances.unsqueeze(0)), min=0).mean()

            L_pos = torch.clamp(C_pos - (T - pos_distances), min=0).mean() if pos else torch.tensor(0., device=device)
            L_neg = torch.clamp(C_neg - (neg_distances - T), min=0).mean() if neg else torch.tensor(0., device=device)

        L_coef = L_contr + L_pos + L_neg

        return L_coef

    # CREATE C MATRIX (GTS) FOR TRACK INIT
    def construct_C_matrix(self, estims, trues):
        estims_xy = estims[:, :2]
        trues_xy = np.array(trues[:, :2], dtype=float)

        distances = np.linalg.norm(estims_xy[:, np.newaxis] - trues_xy, axis=2)

        min_distances = np.min(distances, axis=1)

        C = (min_distances <= 2.0).astype(float)

        C = torch.tensor(C, dtype=torch.float).to(device).unsqueeze(1)

        return C
    
    def construct_C_matrix2(self, estimates, truths):
        estimates_xy = estimates[:, :2]
        truths_xy = torch.as_tensor(truths[:, :2], dtype=torch.float32) # Assuming truths is already a torch tensor

        distances = torch.norm(estimates_xy[:, None] - truths_xy, dim=2)
        min_distances = torch.min(distances, dim=1).values
        
        C = (min_distances <= 2.0).float().unsqueeze(1)
        
        return C

    def feature_fusion(self, F2D, F3D, cam_onehot_vector):
        fused = torch.cat((F2D, cam_onehot_vector), dim=1)
        fused = self.G1(fused)
        fused = fused.reshape(fused.shape[0], 512, 3, 3)
        fused = F3D + fused

        return fused

    def distance_combination_stage_1(self, x):
        if x.shape[1] == 0:
            return torch.empty(0, 0).to(device=device)
        
        ds, ts, channels, height, width = x.shape
        x_reshaped = x.view(-1, channels, height, width)
        result = self.G2(x_reshaped)
        return result.view(ds, ts)

    def distance_combination_stage_2(self, x):
        if x.shape[1] == 0:
            return torch.empty(0, 0).to(device=device), torch.empty(0, 0).to(device=device)
        
        ds, ts, channels, height, width = x.shape
        x_reshaped = x.view(-1, channels, height, width)
        result = self.G3(x_reshaped)
        result_reshaped = result.view(ds, ts, -1)
        a = result_reshaped[:, :, 0]
        b = result_reshaped[:, :, 1]
        return a, b

    def track_initialization(self, x):
        return self.G4(x)





    def forward(self, dets_all, tracking_name):  # FOR EACH SAMPLE IN EACH SCENE FORWARD BASED ON self.state

        tracking_state = self.tracking_states[tracking_name]

        tracking_state['frame_count'] += 1  # NEW FRAME

        # LOAD CURRENT INFORMATION
        dets, pcbs, feats, cam_vecs, info, curr_gts, prev_gts = (
            dets_all['dets'], dets_all['pcbs'], dets_all['fvecs'],
            dets_all['cam_vecs'], dets_all['info'], dets_all['current_gts'],
            dets_all['previous_gts']
        )

        loss = None

        # LOAD FEATURES
        F2D = torch.tensor(feats).to(device)
        F3D = torch.tensor(pcbs).to(device)
        cam_onehot_vector = torch.tensor(cam_vecs).to(device)

        # LOAD AND ORDER DETS BASED ON KALMAN FILTER
        dets = dets[:, tracking_state['order']]
 

        # LOAD TRACK FEATURES IF EXISTANT
        if tracking_state['features']:
            trks_feats = torch.stack([feat for feat in tracking_state['features']], dim=0)
        else:
            trks_feats = torch.empty((0, 0)).to(device)

        # LOAD CORRESPONDING TRACKS IF EXISTANT
        trks = np.zeros((len(tracking_state['trackers']), 7))  # N x 7
        
        # print('dets', dets.shape, trks.shape, trks_feats.shape, tracking_name, '\n\n')
        # [PREDICTION STEP
        to_del = []
        for t, trk in enumerate(trks):
            pos = tracking_state['trackers'][t].predict().reshape((-1, 1))
            trk[:] = pos[:7].flatten()
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            tracking_state['trackers'].pop(t)
            tracking_state['features'].pop(t)

        # COMPUTE S MATRIX
        trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R
                  for tracker in tracking_state['trackers']]

        # COMPUTE MAHALANOBIS DISTANCE
        mah_dist = mahalanobis_distance(dets=dets, trks=trks, trks_S=trks_S)
        D_mah = torch.tensor(mah_dist).to(device)

        # FEATURE FUSION (EXISTS IN EVERY STATE)
        det_feats = self.feature_fusion(F2D, F3D, cam_onehot_vector)

        # CREATE FEATURE MAP BASED ON FEATURES OF DETS AND TRACKS
        feature_map = self.expand_and_concat(det_feats, trks_feats)

        # DISTANCE COMBINATION STAGE 1 (EXISTS IN EVERY STATE)
        D_feat = self.distance_combination_stage_1(feature_map)

        # DISTANCE COMBINATION STAGE 2
        # IF STATE == 1 THEN USE DCS2 ONLY
        # IF STATE == 2 WE USE TRACK_INIT BUT (INHERINTENLY) IT NEEDS DCS2
        if self.state >= 1:
            point_five = torch.tensor(0.5).to(device)
            a, b = self.distance_combination_stage_2(feature_map)
            D_module = D_mah + (a * (D_feat - (point_five + b)))


        # IF TRAIN AND WE TRAIN FOR D_FEAT
        # THERE IS NO VAL MODE IN STAGE 1 OF DC
        if self.training == True and self.state == 0:
            tracking_state['mahanalobis_thresh'] = 0.1
            K = self.construct_K_matrix(distance_matrix=D_feat, dets=dets, curr_gts=curr_gts, trks=trks,
                                            prev_gts=prev_gts)                  # K = torch.randint(0, 2, D_mah.shape, dtype=torch.float, device=device)

            if K.shape[0] > 0:
                loss = self.criterion(D_feat, K)   # criterion is nn.BCEWithLogitsLoss()
                D = K.detach().cpu().numpy()  # set D = K for the perfect matching

            else:  # we are here if prev_gts is empty or D_feat is empty = first sample
                D = np.ones_like(mah_dist)


        # IF TRAIN AND WE TRAIN FOR COMBINATION STAGE 2
        if self.training == True and self.state == 1:
            tracking_state['mahanalobis_thresh'] = 0.1
            K = self.construct_K_matrix(distance_matrix=D_module, dets=dets, curr_gts=curr_gts, trks=trks,
                                        prev_gts=prev_gts)
            
            if K.shape[0] > 0:
                loss = self.Criterion(D_module, K)  # criterion is costum loss
                D = K.detach().cpu().numpy()

            else:
                D = np.ones_like(mah_dist)

        # ELSE WE ARE IN VAL MODE AND WE USE D_MODULE AS D WITH MAH_THRESH 11
        if not self.training:
            D = D_module.cpu().numpy()  


        # GREEDY MATCH
        matched_indices = greedy_match(D)

        # RETRIEVE MATCHED AND UNMATCHED
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

        # NEW TRACKS INITIALIZATION
        if self.state == 2:  # IF AND ONLY IF STATE IS 2 (NOT 0 OR 1)

            if unmatched_dets.shape[0] > 0:
                P = torch.zeros(dets.shape[0], 1, device=self.device)
                unmatched_feats = det_feats[unmatched_dets]
                P[unmatched_dets] = self.track_initialization(unmatched_feats)

                new_track_mask = P[unmatched_dets].squeeze() > self.track_init_thresh
                new_track_indices = unmatched_dets[new_track_mask]

                new_tracks = [
                    KalmanBoxTracker(dets[i], info[i], info[i, -1], tracking_name)
                    for i in new_track_indices]
                tracking_state['trackers'].extend(new_tracks)
                tracking_state['features'].extend([det_feats[i].detach() for i in new_track_indices])

        else:
            for i in unmatched_dets:
                detection_score = info[i][-1]
                trk = KalmanBoxTracker(dets[i, :], info[i, :], detection_score, tracking_name)
                tracking_state['trackers'].append(trk)
                tracking_state['features'].append(det_feats[i].detach())

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


            if self.state == 2:  # IF TRACK INIT
                if curr_gts.shape[0] != 0 and unmatched_dets.shape[0] > 0:
                    C = self.construct_C_matrix2(dets[unmatched_dets], curr_gts)
                    loss = self.criterion(P[unmatched_dets], C)  # criterion is nn.BCELoss()


        # RETURN CURRENT TRACKS AND LOSS
        if len(ret) > 0:
            return np.concatenate(ret), loss
        return np.empty((0, 15 + 7)), loss

    # Helper methods (mahalanobis_distance, expand_and_concat, greedy_match, associate_detections_to_trackers)
    # should be implemented here. You can copy them from the original implementation.

# You'll also need to include the KalmanBoxTracker class implementation.
