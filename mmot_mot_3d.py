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

class ChannelSelector(nn.Module):
    def __init__(self, in_channels=512, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        
        # Channel attention
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Both max and avg pooling for better feature selection
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        channel_weights = (avg_out + max_out).view(b, c, 1, 1)
        return x * channel_weights


class LayerNorm1d(nn.Module):
    def forward(self, x):
        return nn.functional.layer_norm(x, x.shape[1:])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TrackerNN(nn.Module):
    def __init__(self):
        super(TrackerNN, self).__init__()

        # Neural network components - INITIALIZED ONCE
        self.G1 = nn.Sequential(
            nn.Linear(256 + 6, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 72),
            nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(128, 72),
            # nn.ReLU()
        )
        
        self.G2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(in_features=8, out_features=1),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(4, 1),
            nn.Sigmoid()
        )

        self.G3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(in_features=8, out_features=2)
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(4, 2)
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

        self.deepG2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=0),
            nn.ReLU(),

            # Final processing
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.deepG3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=0),
            nn.ReLU(),

            # Final processing
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 2)
        )

        self.deepG3w = nn.Sequential(
            # First block
            nn.Conv2d(1024, 256, 3, padding=0),
            nn.ReLU(),
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(),

            # # Second block
            # nn.Conv2d(512, 256, 3, padding=1),
            # nn.ReLU(),
            # # nn.Conv2d(256, 256, 3, padding=1),
            # # nn.ReLU(),

            # # Third block
            # nn.Conv2d(256, 128, 3, padding=0),
            # nn.ReLU(),

            # Final processing
            nn.Flatten(),
            nn.Linear(256, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

        self.bactiv = nn.Tanh()

        self.channel_selector = ChannelSelector(512)

        # Transformer blocks
        # from attention is all you need
        self.ff_transformer = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)
        self.dc1_transformer = TransformerBlock(d_model=256, num_heads=4, d_ff=1024)
        self.dc2_transformer = TransformerBlock(d_model=256, num_heads=4, d_ff=1024)
        self.ti_transformer = TransformerBlock(d_model=256, num_heads=4, d_ff=1024)

        # WEIGHT INIT
        # self.apply(self.initialize_weights)
        
        # INIT STATES - WILL BE CLEARED AFTER EACH SCENE
        self.tracking_states = {}

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            # Check if this is the last layer before sigmoid
            if isinstance(getattr(m, '_next_module', None), nn.Sigmoid):
                # Initialize the last layer with smaller weights
                nn.init.xavier_normal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, -2)  # Start with lower values before sigmoid
            else:
                # For other layers, use Kaiming init
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # SUB-FORWARDS
    def feature_fusion(self, F2D, F3D, cam_onehot_vector):
        cam_feats = torch.cat((F2D, cam_onehot_vector), dim=1)
        cam_feats = self.G1(cam_feats)

        # cam_feats = cam_feats.reshape(cam_feats.shape[0],F3D.shape[1] , 3, 3)
        lidar_feats = F3D

        # # L2 normalize across spatial dimensions
        lidar_feats = torch.nn.functional.normalize(F3D.view(F3D.shape[0], -1), p=2, dim=1)
        lidar_feats = lidar_feats.view(lidar_feats.shape[0], F3D.shape[1], F3D.shape[2], F3D.shape[3])

        cam_feats = torch.nn.functional.normalize(cam_feats.view(cam_feats.shape[0], -1), p=2, dim=1)
        cam_feats = cam_feats.view(cam_feats.shape[0], F3D.shape[1], F3D.shape[2], F3D.shape[3])
        
        # Fuse the normalized features
        fused = cam_feats + lidar_feats
        
        # normalize the fused output again
        # fused = torch.nn.functional.normalize(fused.view(fused.shape[0], -1), p=2, dim=1)
        # fused = fused.view(fused.shape[0], 512, 3, 3)

        # fused = self.inter(fused)
        
        # fused = fused.reshape(fused.shape[0], 512, -1).transpose(1, 2)
        # fused = self.ff_transformer(fused)
        # fused = fused.transpose(1, 2).reshape(fused.shape)

        # fused_r = torch.randn(F2D.shape[0], 512, 3, 3).to(device)
        # fused = self.inter(fused)

        return lidar_feats, cam_feats


    def distance_combination_stage_1(self, x):
        if x.shape[1] == 0:
            return torch.empty(0, 0).to(device=device)
        
        ds, ts, channels, height, width = x.shape
        x_reshaped = x.reshape(-1, channels, height, width).float()
        x_trans = self.G2(x_reshaped)

        return x_trans.reshape(ds, ts)

    def distance_combination_stage_2(self, x):
        if x.shape[1] == 0:
            return torch.empty(0, 0).to(device=device), torch.empty(0, 0).to(device=device)
        
        ds, ts, channels, height, width = x.shape
        x_reshaped = x.reshape(-1, channels, height, width).float()

        result = self.G3(x_reshaped)

        result_reshaped = result.reshape(ds, ts, -1)
        a = result_reshaped[:, :, 0]
        b = result_reshaped[:, :, 1]
        # b = self.bactiv(b)

        return a, b

    def track_initialization(self, x):

        y = self.G4(x).float()
        
        return y


    def clear_tracking_states(self):
        """Clear all tracking states."""
        self.tracking_states.clear()


    # Initialize AB3DMOT attributes - CALL FOR EACH NEW SCENE
    def reinit_ab3dmot(self, tracking_name, max_age=2, min_hits=3, training=False, state=0, criterion=None, epoch=0):
        self.tracking_states[tracking_name] = {
            'max_age': max_age,
            'min_hits': min_hits,
            'trackers': [],
            'frame_count': 0,
            'mahanalobis_thresh': 11,
            'track_init_thresh': 0.5,
            'features': [],
            'order': [0, 1, 2, 6, 3, 4, 5],  # x, y, z, rot_z, l, w, h
            'order_back': [0, 1, 2, 4, 5, 6, 3],
            'F2DS': []
        }
        self.training = training
        self.state = state
        self.criterion = criterion
        self.epoch = epoch


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
    def construct_K_matrix(self, distance_matrix, dets, curr_gts, trks, prev_gts, threshold=2, epoch=0):
        
        if prev_gts.shape[0] == 0 or curr_gts.shape[0] == 0 or distance_matrix.shape[0] == 0:
            return torch.empty(0,0).to(device=device), torch.empty(0,0).to(device=device)
        
        K = torch.ones_like(distance_matrix)

        d_idx, d_gt_idx = self.hungarian_matching(dets, curr_gts)
        t_idx, t_gt_idx = self.hungarian_matching(trks, prev_gts)

        if len(d_idx) == 0 or len(t_idx) == 0:
            return torch.empty(0,0).to(device=device), torch.empty(0,0).to(device=device)
        
        dets_array = np.array([det[:2] for det in dets])
        curr_gts_array = np.array([gt[:2] for gt in curr_gts], dtype=float)
        trks_array = np.array([trk[:2] for trk in trks])
        prev_gts_array = np.array([gt[:2] for gt in prev_gts], dtype=float)

        dist_1 = np.linalg.norm(dets_array[d_idx] - curr_gts_array[d_gt_idx], axis=1)
        dist_2 = np.linalg.norm(trks_array[t_idx] - prev_gts_array[t_gt_idx], axis=1)

        curr_gts_ids = np.array([gt[7] for gt in curr_gts])
        prev_gts_ids = np.array([gt[7] for gt in prev_gts])

        # Relaxed distance threshold based on epoch
        # threshold = (threshold * (2.0 - min(epoch/5.0, 1.0)))
        mask = torch.zeros_like(distance_matrix, dtype=torch.bool)

        for i, (d, gt_d) in enumerate(zip(d_idx, d_gt_idx)):
            for j, (t, gt_t) in enumerate(zip(t_idx, t_gt_idx)):
                if (curr_gts_ids[gt_d] == prev_gts_ids[gt_t] 
                    and
                        dist_1[i] <= threshold and
                        dist_2[j] <= threshold
                    ):
                    K[d, t] = 0
                    mask[d, t] = True

                if dist_1[i] <= threshold and dist_2[j] <= threshold:
                    mask[d, t] = True


        return K, mask


    # PART OF DCS2 COSTUM LOSS FUNCTION
    def retrieve_pairs_remake(self, K, mask):
        # Only consider positions where mask is True
        pos_indices = torch.nonzero((K == 0) & mask, as_tuple=False)
        neg_indices = torch.nonzero((K != 0) & mask, as_tuple=False)
        
        pos = [tuple(idx) for idx in pos_indices.tolist()]
        neg = [tuple(idx) for idx in neg_indices.tolist()]
        
        return pos, neg
    

    # COSTUM DCS2 (G3) LOSS FUNCTION
    def Criterion(self, distance_matrix=None, K=None, mask=None):
        pos, neg = self.retrieve_pairs_remake(K, mask)

        T, C_contr, C_pos, C_neg = map(lambda x: torch.tensor(x, device=device), [11.0, 6.0, 3.0, 3.0])
        L_contr = L_pos = L_neg = torch.tensor(0., device=device)

        if not pos:
            return None
        
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

        else:
            return None
        
        L_coef = L_contr + L_pos + L_neg
        # print(L_coef)

        return L_coef
    
    def compute_masked_bce_loss(self, D_feat, K, mask):

        # Apply mask to matrices
        D_feat_masked = D_feat[mask]
        K_masked = K[mask]

        if not (K_masked == 0).any():
            return None
        
        n_non_matches = max((K_masked == 1).sum(), 1) 
        n_matches = max((K_masked == 0).sum(), 1)
        pos_weight = (n_matches / n_non_matches)
        criterion = nn.BCELoss(weight=torch.tensor([pos_weight], device=K.device))
        loss = criterion(D_feat_masked, K_masked)
        return loss


    # CREATE C MATRIX (GTS) FOR TRACK INIT
    def construct_C_matrix_old(self, estims, trues):
        estims_xy = estims[:, :2]
        trues_xy = np.array(trues[:, :2], dtype=float)

        distances = np.linalg.norm(estims_xy[:, np.newaxis] - trues_xy, axis=2)

        min_distances = np.min(distances, axis=1)

        C = (min_distances <= 2.0).astype(float)

        C = torch.tensor(C, dtype=torch.float).to(device).unsqueeze(1)

        return C
    
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
        estimates_xy = torch.from_numpy(estimates[:, :2].astype(np.float32))
        truths_xy = torch.from_numpy(truths[:, :2].astype(np.float32))
        
        # Compute pairwise distances between all detections and ground truths
        distances = torch.norm(estimates_xy[:, None] - truths_xy, dim=2)
        
        # Find minimum distance to any ground truth for each detection
        min_distances = torch.min(distances, dim=1).values
        
        # Convert to binary scores based on threshold
        C = (min_distances <= distance_threshold).float().unsqueeze(1)
        
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
        
        # Option 1: L2 normalization (for cosine)
        # det_feats = torch.nn.functional.normalize(det_feats, p=2, dim=1)
        # trk_feats = torch.nn.functional.normalize(trk_feats, p=2, dim=1)
        
        # OR
        
        # Option 2: mean-std normalization
        # det_feats = (det_feats - det_feats.mean(dim=1, keepdim=True)) / (det_feats.std(dim=1, keepdim=True) + 1e-8)
        # trk_feats = (trk_feats - trk_feats.mean(dim=1, keepdim=True)) / (trk_feats.std(dim=1, keepdim=True) + 1e-8)
        
        # similarity matrix
        similarity_matrix = torch.mm(det_feats, trk_feats.t())
        
        return similarity_matrix


    def forward(self, dets_all, tracking_name):  # FOR EACH SAMPLE IN EACH SCENE FORWARD BASED ON self.state

        tracking_state = self.tracking_states[tracking_name]

        tracking_state['frame_count'] += 1  # NEW FRAME

        gamma = 0.2
        
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

        # if tracking_state['F2DS']:
        #     trks_feats_c = torch.stack([feat for feat in tracking_state['F2DS']], dim=0)
        # else:
        #     trks_feats_c = torch.empty((0, 0)).to(device)

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
        mah_dist = mahalanobis_distance(dets=dets, trks=trks, trks_S=trks_S)
        D_mah = torch.tensor(mah_dist).to(device)

        # FEATURE FUSION (EXISTS IN EVERY STATE)
        det_feats, F2DS = self.feature_fusion(F2D, F3D, cam_onehot_vector)

        # CREATE FEATURE MAP BASED ON FEATURES OF DETS AND TRACKS
        feature_map = self.expand_and_concat(det_feats, trks_feats)

        # DISTANCE COMBINATION STAGE 1 (EXISTS IN EVERY STATE)
        D_feat = self.distance_combination_stage_1(feature_map)

        cos_met = self.compute_pairwise_cosine_similarity(det_feats=det_feats, trk_feats=trks_feats)

        # DISTANCE COMBINATION STAGE 2
        # IF STATE == 1 THEN USE DCS2 ONLY
        # IF STATE == 2 WE USE TRACK_INIT BUT (INHERINTENLY) IT NEEDS DCS2
        if self.state >= 1:
            point_five = torch.tensor(0.5).to(device)
            one = torch.tensor(1.0).to(device)
            a, b = self.distance_combination_stage_2(feature_map)

            if self.training == True:
                warmup_factor = min(self.epoch  / 5, 1.0)
                a = a * warmup_factor

            D_module = D_mah + (a * ( (D_feat) - (point_five + b)))

        # IF WE TRAIN FOR D_FEAT
        # THERE IS NO VAL MODE IN STAGE 1 OF DC
        if self.training == True and self.state == 0:
            K, mask = self.construct_K_matrix(distance_matrix=D_feat, dets=dets, curr_gts=curr_gts, trks=trks,
                                            prev_gts=prev_gts, epoch=self.epoch)  
                            # K = torch.randint(0, 2, D_mah.shape, dtype=torch.float, device=device)
            if K.shape[0] > 0:
                # print(cos_met, K, mask)
                loss = self.compute_masked_bce_loss(D_feat, K, mask)

            D = mah_dist

        # IF TRAIN AND WE TRAIN FOR COMBINATION STAGE 2
        if self.training == True and self.state == 1:
            K, mask = self.construct_K_matrix(distance_matrix=D_module, dets=dets, curr_gts=curr_gts, trks=trks,
                                        prev_gts=prev_gts, epoch=self.epoch)
            
            if K.shape[0] > 0:
                # print(cos_met, K)
                loss = self.Criterion(D_module, K, mask)  # criterion is costum loss
                # print(D_feat, K)

            D = D_module.detach().cpu().numpy()  

        # ELSE WE ARE IN VAL MODE (OR TRAIN G4) AND WE USE D_MODULE AS D WITH MAH_THRESH 11
        if self.training == False or self.state == 2:
            D = D_module.cpu().numpy()  

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

        # NEW TRACKS INITIALIZATION
        if self.state == 2:  # IF AND ONLY IF STATE IS 2 (NOT 0 OR 1)

            if unmatched_dets.shape[0] > 0:
                P = torch.zeros(dets.shape[0], 1, device=D_feat.device)
                unmatched_feats = det_feats[unmatched_dets]
                P[unmatched_dets] = self.track_initialization(unmatched_feats)

                for idx in unmatched_dets:
                    if P[idx] > tracking_state['track_init_thresh']:
                        new_track = KalmanBoxTracker(dets[idx], info[idx], info[idx, -1], tracking_name)
                        tracking_state['trackers'].append(new_track)
                        tracking_state['features'].append(det_feats[idx].detach())
                        # tracking_state['F2DS'].append(F2DS[idx].detach())

                if curr_gts.shape[0] != 0  and self.training == True:
                    C = self.construct_C_matrix(dets[unmatched_dets], curr_gts)
                    loss = self.criterion(P[unmatched_dets], C)  # criterion is nn.BCELoss()

        # # INITIALIZE BASED ON GTS IF ON TRAIN
        # if self.training == True:
        #   if unmatched_dets.shape[0] > 0 and curr_gts.shape[0] > 0:
        #         C = torch.zeros(dets.shape[0], 1)
        #         C[unmatched_dets] = self.construct_C_matrix(dets[unmatched_dets], curr_gts)

        #         for idx in unmatched_dets:
        #             if C[idx] > tracking_state['track_init_thresh']:
        #                 new_track = KalmanBoxTracker(dets[idx], info[idx], info[idx, -1], tracking_name)
        #                 tracking_state['trackers'].append(new_track)
        #                 tracking_state['features'].append(det_feats[idx].detach())
        
        # meaning if val
        else:
        # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:  # a scalar of index
                detection_score = info[i][-1]
                track_score = detection_score
                trk = KalmanBoxTracker(dets[i, :], info[i, :], track_score, tracking_name)
                tracking_state['trackers'].append(trk)
                tracking_state['features'].append(det_feats[i].detach())
                # tracking_state['F2DS'].append(F2DS[i].detach())

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
                # tracking_state['F2DS'].pop(i)

        # RETURN CURRENT TRACKS AND LOSS
        if len(ret) > 0:
            return np.concatenate(ret), loss
        return np.empty((0, 15 + 7)), loss

