'''
TA NETS ME MIA FORWARD GIA KATHENA Gi
MIA KLASH GIA OLA

SYNDEETAI ME TO g2_trainval_v1.py
'''



from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Modules(nn.Module):

    def __init__(self):
        super(Modules, self).__init__()

        # FEATURE FUSION MODULE
        self.g1 = nn.Sequential(
            nn.Linear(1024 + 6, 1536),
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

        fused = fused.reshape(fused.shape[0], 512, 3, 3)
        fused = fused + F3D

        return fused

    def G2(self, x):
        ds, ts, channels, height, width = x.shape
        x_reshaped = x.view(-1, channels, height, width)

        result = self.g2(x_reshaped)

        y = result.view(ds, ts)

        return y

    def G3(self, x):
        ds, ts, channels, height, width = x.shape
        x_reshaped = x.view(-1, channels, height, width)

        result = self.g3(x_reshaped)

        result_reshaped = result.view(ds, ts, -1)
        a = result_reshaped[:, :, 0]
        b = result_reshaped[:, :, 1]

        return a, b

    def G4(self, x):

        score = self.g4(x)
        return score