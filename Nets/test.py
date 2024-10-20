from torch import nn
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DIST_COMB_MODULE(nn.Module):

    def __init__(self):
        super(DIST_COMB_MODULE, self).__init__()

        # FEATURE FUSION MODULE
        self.G1 = nn.Sequential(
            nn.Linear(1024 + 6, 1536),
            nn.ReLU(),
            nn.Linear(1536, 4608),
            nn.ReLU()
        )
        
        self.G11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.Sigmoid()
        )

    #     # DISTANCE COMBINATION MODULE 1
    #     self.G2 = nn.Sequential(
    #         nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=0, stride=1),
    #         nn.Flatten(),
    #         nn.ReLU(),
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 1)
    #     )
        
    #     self.G3 = nn.Sequential(
    #         nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=0, stride=1),
    #         nn.ReLU(),
    #         nn.Flatten(),
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 2)
    #     )

    # def expand_and_concat(self, det_feats, trk_feats):
    #     if trk_feats.shape[0] == 0:
    #         return torch.empty((det_feats.shape[0], trk_feats.shape[0],
    #                             2 * det_feats.shape[1],
    #                             det_feats.shape[2], det_feats.shape[3])).to(device)

    #     det_feats_expanded = det_feats.unsqueeze(1)  # Shape: (N, 1, C, H, W)
    #     trk_feats_expanded = trk_feats.unsqueeze(0)  # Shape: (1, M, C, H, W)

    #     # Concatenate along the channel dimension
    #     matrix = torch.cat((det_feats_expanded.expand(-1, trk_feats.shape[0], -1, -1, -1),
    #                         trk_feats_expanded.expand(det_feats.shape[0], -1, -1, -1, -1)),
    #                     dim=2)

        # return matrix
    
    def forward(self, F2D, F3D, cam_onehot_vector):
        
        fused = torch.cat((F2D, cam_onehot_vector), dim=1)

        fused = self.G1(fused)

        fused = fused.reshape(fused.shape[0], 512, 3, 3)

        fused = F3D + fused

        predictions = self.G11(fused)
        return predictions, fused
    
        feat_map = self.expand_and_concat(fused, track_feats)

        if feat_map.numel() == 0:
            return torch.empty((0, 0)), fused
        
        ds, ts, channels, height, width = feat_map.shape

        x_reshaped = feat_map.view(-1, channels, height, width)

        result = self.G2(x_reshaped)

        D_feat = result.view(ds, ts)

        if state > 0:

            result = self.G3(x_reshaped)
            result_reshaped = result.view(ds, ts, -1)

            a = result_reshaped[:, :, 0]
            b = result_reshaped[:, :, 1]

            point_five = torch.tensor(0.5).to(device)

            D = mah_metric + (a * (D_feat - (point_five + b)))

            return D, fused

        return D_feat, fused





