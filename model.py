import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CombinedClassifier(nn.Module):
    """
    Combined classifier that processes both latent embeddings and flow maps.
    
    Architecture:
    - Latent branch: Processes latent embeddings (640, 7, 8, 8) 
    - Flow branch: Processes optical flow (2, 7, H, W)
    - Fusion: Combines both branches with timestep information
    - Output: Single score (0-1, where 1=success, 0=failure)
    """
    def __init__(self, latent_channels=640, flow_channels=2, hidden_dim=128, 
                 latent_spatial_size=8, flow_spatial_size=480, 
                 fusion_dim=256, dropout=0.3):
        """
        Args:
            latent_channels: Number of channels in latent embeddings (640)
            flow_channels: Number of flow channels (2 for u,v)
            hidden_dim: Hidden dimension for conv layers
            latent_spatial_size: Spatial size of latent features (8x8)
            flow_spatial_size: Spatial size of flow maps (480x640 or similar)
            fusion_dim: Dimension for fusion layer
            dropout: Dropout probability
        """
        super().__init__()
        
        self.latent_channels = latent_channels
        self.flow_channels = flow_channels
        self.latent_spatial_size = latent_spatial_size
        self.flow_spatial_size = flow_spatial_size
        
        # === Latent Embedding Branch ===
        # Input: (B, 640, 7, 8, 8)
        # Use 3D convolutions to process spatio-temporal latent features
        self.latent_branch = nn.Sequential(
            nn.Conv3d(latent_channels, hidden_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            
            nn.Conv3d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
        )
        
        # Global pooling for latent features: (B, C, T, H, W) -> (B, C)
        self.latent_pool = nn.AdaptiveAvgPool3d(1)
        latent_feature_dim = hidden_dim * 2
        
        # === Flow Branch ===
        # Input: (B, 2, 7, H, W)
        # Use 3D convolutions to process spatio-temporal flow
        self.flow_branch = nn.Sequential(
            nn.Conv3d(flow_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            
            nn.Conv3d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
        )
        
        # Global pooling for flow features: (B, C, T, H, W) -> (B, C)
        self.flow_pool = nn.AdaptiveAvgPool3d(1)
        flow_feature_dim = hidden_dim * 4
        
        # === Timestep Embedding ===
        timestep_embed_dim = 64
        self.timestep_encoder = nn.Sequential(
            nn.Linear(1, timestep_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(timestep_embed_dim, timestep_embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # === Fusion Layer ===
        # Combine latent features + flow features + timestep embedding
        combined_dim = latent_feature_dim + flow_feature_dim + timestep_embed_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_dim // 2, 1)  # Output: single logit
        )
    
    def forward(self, latent, flow, timestep, return_embedding=False):
        """
        Args:
            latent: (B, 640, 7, 8, 8) - latent embeddings
            flow: (B, 2, 7, H, W) - optical flow maps
            timestep: (B, 1) - diffusion timesteps
        
        Returns:
            score: (B, 1) - raw logits (apply sigmoid for probabilities)
        """
        batch_size = latent.size(0)
        
        # Process latent branch
        latent_feat = self.latent_branch(latent)  # (B, hidden_dim*2, 7, 8, 8)
        latent_feat = self.latent_pool(latent_feat)  # (B, hidden_dim*2, 1, 1, 1)
        latent_feat = latent_feat.view(batch_size, -1)  # (B, hidden_dim*2)
        
        # Process flow branch
        flow_feat = self.flow_branch(flow)  # (B, hidden_dim*4, T', H', W')
        flow_feat = self.flow_pool(flow_feat)  # (B, hidden_dim*4, 1, 1, 1)
        flow_feat = flow_feat.view(batch_size, -1)  # (B, hidden_dim*4)
        
        # Process timestep
        timestep_feat = self.timestep_encoder(timestep)  # (B, timestep_embed_dim)
        
        # Concatenate all features
        combined = torch.cat([latent_feat, flow_feat, timestep_feat], dim=1)  # (B, combined_dim)
        
        # Fusion and classification
        embedding = self.fusion[:-1](combined)  # Penultimate embedding
        score = self.fusion[-1](embedding)  # (B, 1)

        if return_embedding:
            return score, embedding
        return score


class CombinedConvLSTM(nn.Module):
    """
    Alternative architecture using ConvLSTM for temporal processing.
    Processes latent and flow separately with ConvLSTM, then fuses.
    """
    def __init__(self, latent_channels=640, flow_channels=2, hidden_dim=128,
                 lstm_hidden=256, fusion_dim=256, dropout=0.3):
        super().__init__()
        
        # === Latent Branch with ConvLSTM ===
        # Process each latent frame, then aggregate with LSTM
        self.latent_frame_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # (B, hidden_dim*4, 1, 1)
        )
        
        self.latent_lstm = nn.LSTM(
            input_size=hidden_dim * 4,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # === Flow Branch with ConvLSTM ===
        # Spatial processing for flow frames
        self.flow_frame_encoder = nn.Sequential(
            nn.Conv2d(flow_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1)  # (B, hidden_dim*2, 1, 1)
        )
        
        self.flow_lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # === Timestep Embedding ===
        timestep_embed_dim = 64
        self.timestep_encoder = nn.Sequential(
            nn.Linear(1, timestep_embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # === Fusion ===
        # Use final hidden states from both LSTMs + timestep
        combined_dim = lstm_hidden * 2 + timestep_embed_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_dim // 2, 1)
        )
    
    def forward(self, latent, flow, timestep, return_embedding=False):
        """
        Args:
            latent: (B, 640, 7, 8, 8)
            flow: (B, 2, 7, H, W)
            timestep: (B, 1)
        
        Returns:
            score: (B, 1)
        """
        B, C_lat, T_lat, H_lat, W_lat = latent.shape
        B, C_flow, T_flow, H_flow, W_flow = flow.shape
        
        # Process latent frames
        # Reshape: (B, C, T, H, W) -> (B*T, C, H, W)
        latent_frames = latent.permute(0, 2, 1, 3, 4).reshape(B * T_lat, C_lat, H_lat, W_lat)
        latent_feat = self.latent_frame_encoder(latent_frames)  # (B*T, hidden_dim*4, 1, 1)
        latent_feat = latent_feat.view(B, T_lat, -1)  # (B, T, hidden_dim*4)
        
        # LSTM over latent sequence
        _, (latent_hidden, _) = self.latent_lstm(latent_feat)  # hidden: (num_layers, B, lstm_hidden)
        latent_hidden = latent_hidden[-1]  # Take last layer: (B, lstm_hidden)
        
        # Process flow frames
        # Reshape: (B, C, T, H, W) -> (B*T, C, H, W)
        flow_frames = flow.permute(0, 2, 1, 3, 4).reshape(B * T_flow, C_flow, H_flow, W_flow)
        flow_feat = self.flow_frame_encoder(flow_frames)  # (B*T, hidden_dim*2, 1, 1)
        flow_feat = flow_feat.view(B, T_flow, -1)  # (B, T, hidden_dim*2)
        
        # LSTM over flow sequence
        _, (flow_hidden, _) = self.flow_lstm(flow_feat)  # hidden: (num_layers, B, lstm_hidden)
        flow_hidden = flow_hidden[-1]  # Take last layer: (B, lstm_hidden)
        
        # Process timestep
        timestep_feat = self.timestep_encoder(timestep)  # (B, timestep_embed_dim)
        
        # Fuse
        combined = torch.cat([latent_hidden, flow_hidden, timestep_feat], dim=1)
        embedding = self.fusion[:-1](combined)  # Penultimate embedding
        score = self.fusion[-1](embedding)

        if return_embedding:
            return score, embedding
        return score


class LatentVideoTransformer(nn.Module):
    """
    Transformer-based classifier for latent embeddings + raw x0 video.

    Design:
    - Latent encoder: patchifies each 8x8 latent frame and encodes with a Transformer.
    - Video encoder: patchifies each RGB frame and encodes with a Transformer.
    - Fusion encoder: concatenates latent frame tokens + video frame tokens + timestep token,
      then predicts from a learnable CLS token.
    """

    def __init__(
        self,
        latent_channels=640,
        video_channels=3,
        d_model=256,
        num_heads=8,
        latent_patch_size=2,
        video_patch_size=16,
        latent_frames=7,
        video_frames=8,
        video_size=128,
        latent_encoder_layers=2,
        video_encoder_layers=2,
        fusion_encoder_layers=3,
        ff_dim=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.latent_frames = latent_frames
        self.video_frames = video_frames
        self.video_size = video_size

        # ---- Latent patch embedding ----
        self.latent_patch_embed = nn.Conv2d(
            latent_channels,
            d_model,
            kernel_size=latent_patch_size,
            stride=latent_patch_size,
        )
        latent_patches_per_frame = (8 // latent_patch_size) * (8 // latent_patch_size)
        self.latent_spatial_pos = nn.Parameter(
            torch.randn(1, 1, latent_patches_per_frame, d_model) * 0.02
        )
        self.latent_temporal_pos = nn.Parameter(
            torch.randn(1, latent_frames, 1, d_model) * 0.02
        )

        latent_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.latent_encoder = nn.TransformerEncoder(
            latent_layer, num_layers=latent_encoder_layers
        )

        # ---- Video patch embedding ----
        self.video_patch_embed = nn.Conv2d(
            video_channels,
            d_model,
            kernel_size=video_patch_size,
            stride=video_patch_size,
        )
        video_patches_per_frame = (video_size // video_patch_size) * (video_size // video_patch_size)
        self.video_spatial_pos = nn.Parameter(
            torch.randn(1, 1, video_patches_per_frame, d_model) * 0.02
        )
        self.video_temporal_pos = nn.Parameter(
            torch.randn(1, video_frames, 1, d_model) * 0.02
        )

        video_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.video_encoder = nn.TransformerEncoder(video_layer, num_layers=video_encoder_layers)

        # ---- Timestep and fusion transformer ----
        self.timestep_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Sequence order: [CLS] + latent_frame_tokens + video_frame_tokens + timestep_token
        self.fusion_pos = nn.Parameter(
            torch.randn(1, 1 + latent_frames + video_frames + 1, d_model) * 0.02
        )

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(
            fusion_layer, num_layers=fusion_encoder_layers
        )

        self.norm = nn.LayerNorm(d_model)
        self.pre_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(d_model, 1)

    def _encode_latent(self, latent):
        # latent: (B, 640, 7, 8, 8)
        bsz, channels, t_lat, h_lat, w_lat = latent.shape
        if t_lat != self.latent_frames:
            raise ValueError(f"Expected {self.latent_frames} latent frames, got {t_lat}")

        latent_bt = latent.permute(0, 2, 1, 3, 4).reshape(bsz * t_lat, channels, h_lat, w_lat)
        latent_tokens = self.latent_patch_embed(latent_bt)  # (B*T, D, Hp, Wp)
        latent_tokens = latent_tokens.flatten(2).transpose(1, 2)  # (B*T, Np, D)

        n_patches = latent_tokens.size(1)
        latent_tokens = latent_tokens.view(bsz, t_lat, n_patches, -1)
        latent_tokens = (
            latent_tokens
            + self.latent_spatial_pos[:, :, :n_patches, :]
            + self.latent_temporal_pos[:, :t_lat, :, :]
        )

        latent_tokens = latent_tokens.view(bsz, t_lat * n_patches, -1)
        latent_tokens = self.latent_encoder(latent_tokens)
        latent_tokens = latent_tokens.view(bsz, t_lat, n_patches, -1)

        # Collapse spatial patches to one token per latent timestep.
        return latent_tokens.mean(dim=2)  # (B, T_lat, D)

    def _encode_video(self, video):
        # video: (B, 3, 8, H, W)
        bsz, channels, t_vid, h_vid, w_vid = video.shape
        if t_vid != self.video_frames:
            raise ValueError(f"Expected {self.video_frames} video frames, got {t_vid}")

        video_bt = video.permute(0, 2, 1, 3, 4).reshape(bsz * t_vid, channels, h_vid, w_vid)
        if (h_vid, w_vid) != (self.video_size, self.video_size):
            video_bt = F.interpolate(
                video_bt,
                size=(self.video_size, self.video_size),
                mode='bilinear',
                align_corners=False,
            )

        video_tokens = self.video_patch_embed(video_bt)  # (B*T, D, Hp, Wp)
        video_tokens = video_tokens.flatten(2).transpose(1, 2)  # (B*T, Np, D)

        n_patches = video_tokens.size(1)
        video_tokens = video_tokens.view(bsz, t_vid, n_patches, -1)
        video_tokens = (
            video_tokens
            + self.video_spatial_pos[:, :, :n_patches, :]
            + self.video_temporal_pos[:, :t_vid, :, :]
        )

        video_tokens = video_tokens.view(bsz, t_vid * n_patches, -1)
        video_tokens = self.video_encoder(video_tokens)
        video_tokens = video_tokens.view(bsz, t_vid, n_patches, -1)

        # Collapse spatial patches to one token per video timestep.
        return video_tokens.mean(dim=2)  # (B, T_vid, D)

    def forward(self, latent, video, timestep, return_embedding=False):
        """
        Args:
            latent: (B, 640, 7, 8, 8)
            video: (B, 3, 8, H, W) raw x0 video frames (with condition frame)
            timestep: (B, 1)

        Returns:
            score: (B, 1) logits
        """
        if video.max() > 1.0:
            video = video / 255.0

        latent_frame_tokens = self._encode_latent(latent)
        video_frame_tokens = self._encode_video(video)
        timestep_token = self.timestep_proj(timestep).unsqueeze(1)  # (B, 1, D)

        bsz = latent.size(0)
        cls = self.cls_token.expand(bsz, -1, -1)
        fused = torch.cat([cls, latent_frame_tokens, video_frame_tokens, timestep_token], dim=1)
        fused = fused + self.fusion_pos[:, :fused.size(1), :]

        fused = self.fusion_encoder(fused)
        cls_out = self.norm(fused[:, 0])
        embedding = self.pre_head(cls_out)
        score = self.head(embedding)

        if return_embedding:
            return score, embedding
        return score


def random_time_masking(x, mask_prob=0.15):
    """
    Randomly mask out entire time steps during training for regularization.
    
    Args:
        x: Tensor of shape (B, C, T, H, W)
        mask_prob: Probability of masking each timestep
    
    Returns:
        Masked tensor
    """
    if mask_prob <= 0 or not x.requires_grad:  # Only during training
        return x
    
    B, C, T, H, W = x.shape
    # Create random mask: (B, T)
    mask = torch.rand(B, T, device=x.device) > mask_prob
    # Reshape mask to (B, 1, T, 1, 1) for broadcasting
    mask = mask.view(B, 1, T, 1, 1).float()
    
    return x * mask


# For backward compatibility with loading scripts that might expect these names
VideoScoreModel = CombinedClassifier
VideoClassifier = CombinedConvLSTM
