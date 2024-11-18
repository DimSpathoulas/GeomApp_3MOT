import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MultiModalEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # LiDAR encoder (512x3x3 -> latent_dim/2)
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, latent_dim // 2),
            nn.ReLU()
        )
        
        # Image feature encoder (1024 -> latent_dim/2)
        self.image_encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim // 2),
            nn.ReLU()
        )
        
        # Camera position encoder (6 -> 32)
        self.camera_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU()
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + 32, latent_dim),
            nn.ReLU()
        )
        
    def forward(self, lidar_features, image_features, camera_pos):
        # Encode each modality
        lidar_encoded = self.lidar_encoder(lidar_features)
        image_encoded = self.image_encoder(image_features)
        camera_encoded = self.camera_encoder(camera_pos)
        
        # Concatenate encodings
        combined = torch.cat([lidar_encoded, image_encoded, camera_encoded], dim=1)
        
        # Final fusion
        latent = self.fusion(combined)
        return latent

class MultiModalDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # LiDAR decoder
        self.lidar_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 3 * 3),
            nn.ReLU(),
            nn.Unflatten(1, (128, 3, 3)),
            nn.ConvTranspose2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 512, kernel_size=3, padding=1)
        )
        
        # Image feature decoder
        self.image_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )
        
        # Camera position decoder
        self.camera_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
            nn.Softmax(dim=1)
        )
        
    def forward(self, latent):
        # Decode each modality
        lidar_reconstructed = self.lidar_decoder(latent)
        image_reconstructed = self.image_decoder(latent)
        camera_reconstructed = self.camera_decoder(latent)
        
        return lidar_reconstructed, image_reconstructed, camera_reconstructed

class MultiModalTrackingAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        
        self.encoder = MultiModalEncoder(latent_dim)
        self.decoder = MultiModalDecoder(latent_dim)
        
        # Class-specific layers
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        self.class_attention = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )
        
    def forward(self, lidar_features, image_features, camera_pos, class_ids=None):
        # Encode features
        latent = self.encoder(lidar_features, image_features, camera_pos)
        
        if class_ids is not None:
            # Get class embeddings
            class_emb = self.class_embedding(class_ids)
            
            # Compute attention weights
            attention = self.class_attention(torch.cat([latent, class_emb], dim=1))
            
            # Apply class-specific attention
            latent = latent * attention
        
        # Decode features
        lidar_reconstructed, image_reconstructed, camera_reconstructed = self.decoder(latent)
        
        return latent, (lidar_reconstructed, image_reconstructed, camera_reconstructed)

class TrackingDataset(Dataset):
    def __init__(self, lidar_features, image_features, camera_positions, class_ids):
        self.lidar_features = lidar_features
        self.image_features = image_features
        self.camera_positions = camera_positions
        self.class_ids = class_ids
        
    def __len__(self):
        return len(self.lidar_features)
    
    def __getitem__(self, idx):
        return (self.lidar_features[idx], 
                self.image_features[idx],
                self.camera_positions[idx],
                self.class_ids[idx])

def train_autoencoder(model, train_loader, val_loader, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Reconstruction loss weights
    lidar_weight = 1.0
    image_weight = 1.0
    camera_weight = 0.5
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            lidar_features, image_features, camera_pos, class_ids = [x.to(device) for x in batch]
            
            # Forward pass
            latent, (lidar_recon, image_recon, camera_recon) = model(
                lidar_features, image_features, camera_pos, class_ids
            )
            
            # Compute losses
            lidar_loss = F.mse_loss(lidar_recon, lidar_features)
            image_loss = F.mse_loss(image_recon, image_features)
            camera_loss = F.cross_entropy(camera_recon, torch.argmax(camera_pos, dim=1))
            
            # Combined loss
            loss = (lidar_weight * lidar_loss + 
                   image_weight * image_loss + 
                   camera_weight * camera_loss)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                lidar_features, image_features, camera_pos, class_ids = [x.to(device) for x in batch]
                
                latent, (lidar_recon, image_recon, camera_recon) = model(
                    lidar_features, image_features, camera_pos, class_ids
                )
                
                lidar_loss = F.mse_loss(lidar_recon, lidar_features)
                image_loss = F.mse_loss(image_recon, image_features)
                camera_loss = F.cross_entropy(camera_recon, torch.argmax(camera_pos, dim=1))
                
                val_loss += (lidar_weight * lidar_loss + 
                           image_weight * image_loss + 
                           camera_weight * camera_loss).item()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')

# Example usage
def process_features(model, lidar_feature, image_feature, camera_pos, class_id):
    model.eval()
    with torch.no_grad():
        latent, _ = model(
            lidar_feature.unsqueeze(0),
            image_feature.unsqueeze(0),
            camera_pos.unsqueeze(0),
            torch.tensor([class_id])
        )
    return latent.squeeze(0)