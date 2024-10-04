import torch
import torch.nn as nn

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

class Feature_Fusion(nn.Module):
    def __init__(self):
        super(Feature_Fusion, self).__init__()
        
        # FEATURE FUSION MODULE
        self.G1 = nn.Sequential(
            nn.Linear(1024 + 6, 1536),
            nn.ReLU(),
            nn.Linear(1536, 4608),
            nn.ReLU()
        )
        
        # Add Transformer block
        self.transformer = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)
    
    def forward(self, F2D, F3D, cam_onehot_vector):
        fused = torch.cat((F2D, cam_onehot_vector), dim=1)
        fused = self.G1(fused)
        fused = fused.reshape(fused.shape[0], 512, 3, 3)
        
        # Apply transformer to reshaped fused features
        fused = fused.view(fused.shape[0], 512, -1).transpose(1, 2)
        fused = self.transformer(fused)
        fused = fused.transpose(1, 2).view(fused.shape[0], 512, 3, 3)
        
        fused = F3D + fused
        return fused

class Distance_Combination_Stage_1(nn.Module):
    def __init__(self):
        super(Distance_Combination_Stage_1, self).__init__()
        
        # DISTANCE COMBINATION MODULE 1
        self.G2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=0, stride=1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Add Transformer block
        self.transformer = TransformerBlock(d_model=256, num_heads=4, d_ff=1024)
    
    def forward(self, x):
        ds, ts, channels, height, width = x.shape
        x_reshaped = x.view(-1, channels, height, width)
        
        # Apply conv2d
        x_conv = self.G2[0](x_reshaped)
        
        # Apply transformer
        x_trans = x_conv.view(x_conv.shape[0], 256, -1).transpose(1, 2)
        x_trans = self.transformer(x_trans)
        x_trans = x_trans.transpose(1, 2).view(x_conv.shape)
        
        # Continue with the rest of G2
        result = self.G2[1:](x_trans)
        
        y = result.view(ds, ts)
        return y