import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Nets.dist_comb_net import COMBINATION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

COMB = COMBINATION().to(device)
optimizer = torch.optim.Adam(COMB.parameters(), lr=0.001) 
criterion = nn.BCEWithLogitsLoss()

writer = SummaryWriter('runs/graphG2')

# Create dummy inputs
N = 7  # Batch size
m = 3  # Number of tracks
F2D_dummy = torch.randn(N, 1024).to(device)  # Dummy input for F2D
F3D_dummy = torch.randn(N, 512, 3, 3).to(device)  # Dummy input for F3D
cam_onehot_vector_dummy = torch.randn(N, 6).to(device)  # Dummy input for camera vector
tracks = torch.randn(m, 512, 3, 3).to(device)  # Dummy tracks input
K = torch.randn(N,m).to(device)
mah = torch.randn(N,m).to(device)
state = torch.tensor(0).to(device)
# Forward pass through the Feature Fusion module
map, fused = COMB(F2D_dummy, F3D_dummy, cam_onehot_vector_dummy, tracks, mah, state)

print(map.requires_grad)
print(map.grad_fn)
print(fused.requires_grad)
print(fused.grad_fn)

loss = criterion(map, K)

# Perform backward pass
loss.backward()

for param in COMB.parameters():
    print(param.grad is not None)

for param in COMB.parameters():
    print(param.requires_grad)

for name, param in COMB.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

for name, param in COMB.named_parameters():
    if param.grad is not None:
        print(f"{name}: Gradient sum: {param.grad.sum()}")
    else:
        print(f"{name}: No gradient")

# Log the model graphs to TensorBoard for both FF and DCS1
writer.add_graph(COMB, (F2D_dummy, F3D_dummy, cam_onehot_vector_dummy, tracks, mah, state))  # Log FF
writer.close()

print("Model graph has been logged to TensorBoard!")
