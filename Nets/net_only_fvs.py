'''
KATHE Gi EINAI JEXORISTH KLASH ME DIKIA THS FORWARD

'''



from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Feature_Fusion(nn.Module):

    def __init__(self):
        super(Feature_Fusion, self).__init__()

        # FEATURE FUSION MODULE
        self.G1 = nn.Sequential(
            nn.Linear(1024 + 6, 1536),
            nn.ReLU(),
            nn.Linear(1536, 4608),
            nn.ReLU(),
        )

    def forward(self, F2D, F3D, cam_onehot_vector):
        F2D = torch.tensor(F2D).to(device)
        # F3D = torch.tensor(F3D).to(device)
        cam_onehot_vector = torch.tensor(cam_onehot_vector).to(device)

        fused = torch.cat((F2D, cam_onehot_vector), dim=1)

        fused = self.G1(fused)

        fused = fused.reshape(fused.shape[0], 512, 3, 3)
        # fused = F3D + fused

        return fused

class Distance_Combination_Stage_1(nn.Module):

    def __init__(self):
        super(Distance_Combination_Stage_1, self).__init__()

        # DISTANCE COMBINATION MODULE 1
        self.G2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        ds, ts, channels, height, width = x.shape
        x_reshaped = x.view(-1, channels, height, width)

        result = self.G2(x_reshaped)

        y = result.view(ds, ts)

        return y


class Distance_Combination_Stage_2(nn.Module):

    def __init__(self):
        super(Distance_Combination_Stage_2, self).__init__()

        # DISTANCE COMBINATION MODULE 2
        self.g3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        ds, ts, channels, height, width = x.shape
        x_reshaped = x.view(-1, channels, height, width)

        result = self.g3(x_reshaped)

        result_reshaped = result.view(ds, ts, -1)
        a = result_reshaped[:, :, 0]
        b = result_reshaped[:, :, 1]

        return a, b


class Track_Init(nn.Module):

    def __init__(self):
        super(Track_Init, self).__init__()
        
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
        
    def forward(self, x):
        score = self.g4(x)
        
        return score






