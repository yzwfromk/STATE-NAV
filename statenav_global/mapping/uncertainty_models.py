import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import MultiScaleDeformableAttention
import torchvision.models as models


# -------------------------
# Spatial Attention Module
# -------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)    # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)
        x_cat = torch.cat([avg_out, max_out], dim=1)    # (B,2,H,W)
        attention = self.sigmoid(self.conv(x_cat))      # (B,1,H,W)
        return x * attention


class ResNetFeatureExtractor(nn.Module):
    """
    A simple wrapper around ResNet-18 to extract feature maps
    up to the last conv layer. The output feature map has shape
    (B, 512, H_out, W_out) at 1/32 the input resolution.
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Load a pretrained ResNet-18
        base_model = models.resnet18(pretrained=pretrained)
        
        # Keep everything except the final avgpool + fc:
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x shape: (B, 3, H, W)
        features = self.backbone(x)  # -> (B, 512, H_out, W_out)
        return features


# ------------------------------
# Deformable Attention Module
# ------------------------------
class DeformableAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, num_levels, num_points):
        super(DeformableAttentionModule, self).__init__()
        
        self.deformable_attention = MultiScaleDeformableAttention(
            embed_dims=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, embed_dim, H, W)
        """
        B, embed_dim, H, W = x.size()
        
        # Flatten spatial dimensions -> (HW, B, C)
        x_flat = x.view(B, embed_dim, -1).permute(2, 0, 1)  # (H*W, B, embed_dim)

        # Spatial shapes and level index
        spatial_shapes = torch.tensor([[H, W]], device=x.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=x.device, dtype=torch.long)
        
        # Reference points in [0,1] for deformable attention
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, H, device=x.device),
            torch.linspace(0.0, 1.0, W, device=x.device),
            indexing="ij"
        )
        reference_points = torch.stack((grid_x, grid_y), dim=-1).view(1, H * W, 1, 2)
        reference_points = reference_points.repeat(B, 1, spatial_shapes.size(0), 1)  # (B, H*W, num_levels, 2)

        # Deformable attention
        x_attn = self.deformable_attention(
            query=x_flat,
            key=None,
            value=None,
            query_pos=None,
            key_pos=None,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )
        
        # Reshape back to (B, embed_dim, H, W)
        x_out = x_attn.permute(1, 2, 0).view(B, embed_dim, H, W)
        return x_out
    

class TransformerFeatureExtractor(nn.Module):
    """
    This module does a stride-based or "patchify" extraction, adds positional embeddings,
    and passes flattened tokens through a Transformer Encoder. Finally, it reshapes the
    tokens back to a 2D feature map (B, embed_dim, H_out, W_out).
    """
    def __init__(
        self,
        in_channels=3,
        embed_dim=512,
        patch_size=8,
        num_encoder_layers=6,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()

        # "Patchify" the input
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional encoding (2D sine-cosine)
        self.positional_encoding = PositionEmbeddingSine2D(embed_dim // 2)

        # Standard Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)  # -> (B, embed_dim, H_out, W_out)
        B, C, H_out, W_out = x.shape

        # Flatten to (N, B, C), where N = H_out*W_out
        x_flat = x.flatten(2).permute(2, 0, 1)

        # Create positional embeddings -> (N, C), replicate for each batch -> (N, B, C)
        pos_emb = self.positional_encoding(H_out, W_out, device=x.device)
        pos_emb = pos_emb.unsqueeze(1).expand(-1, B, -1)

        # Transformer encoder
        x_enc = self.transformer_encoder(x_flat + pos_emb)  # (N, B, C)

        # Reshape back to (B, C, H_out, W_out)
        x_out = x_enc.permute(1, 2, 0).view(B, C, H_out, W_out)
        return x_out


class PositionEmbeddingSine2D(nn.Module):
    """
    2D sine-cosine positional embeddings, shaped as used in many DETR-like architectures.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        import math
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, height, width, device):
        mask = torch.zeros(height, width, device=device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[-1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        # interleave sine and cos
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x), dim=2)  # (H, W, 2*num_pos_feats)
        pos = pos.view(height * width, -1)      # (H*W, 2*num_pos_feats)
        return pos


class TransformerDecoderHead(nn.Module):
    """
    A minimal Transformer decoder head that:
      - Takes flattened memory (N, B, E) from an encoder or conv features.
      - Takes commands (B, 3) => embedded into (B, E).
      - Runs a TransformerDecoder:
         query shape = (Q=1, B, E)
         memory shape = (N, B, E)
      - Outputs (B, E) => final linear => (B, 1).
    """
    def __init__(self, embed_dim=512, nhead=8, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # Now embedding 3 inputs (v, omega, terrain_level) => E
        self.cmd_embed = nn.Linear(2, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False  # (seq, batch, channel)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, memory, commands):
        """
        Args:
          memory: (N, B, E) flattened feature map
          commands: (B, 3)  # e.g. [v, omega, terrain_level]
        Returns:
          (B, 1)
        """
        B = commands.shape[0]

        # Embed commands => (B, E)
        query_embed = self.cmd_embed(commands)  # (B, E)

        # We need (Q, B, E). Let Q=1 => single query for regression
        query_embed = query_embed.unsqueeze(0)  # (1, B, E)

        decoded = self.transformer_decoder(query_embed, memory)  # => (1, B, E)
        decoded = decoded.squeeze(0)  # => (B, E)

        # Final => (B, 1)
        output = self.out_proj(decoded)
        return output


class ElevationOnlyNetworkMSE(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        num_levels=1,
        num_points=4,
        dropout_p=0.1,
        freeze_backbone=False,
        backbone_type="resnet",
        decoder_type="transformer",  # or "transformer"
    ):
        super().__init__()

        # 1. Choose feature extractor
        if backbone_type.lower() == "transformer":
            self.feature_extractor = TransformerFeatureExtractor(
                in_channels=3,
                embed_dim=embed_dim,
                patch_size=8,
                num_encoder_layers=8,
                nhead=num_heads,
                dim_feedforward=1024,
                dropout=dropout_p
            )
        else:
            self.feature_extractor = ResNetFeatureExtractor(
                pretrained=True,
                freeze_backbone=freeze_backbone
            )

        # 2. Spatial/Deformable attention modules (optional usage)
        self.spatial_attention = SpatialAttention(kernel_size=3)
        self.deformable_attention = DeformableAttentionModule(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            num_levels=num_levels, 
            num_points=num_points
        )

        # 3. Decoder => MLP or Transformer
        if decoder_type.lower() == "mlp":
            self.fc_fusion = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.Dropout(p=dropout_p),
                nn.Linear(256, 128), 
                nn.Dropout(p=dropout_p),
                nn.Linear(128, 64),
            )
            self.fc_pred = nn.Sequential(
                nn.Linear(64 + 2, 32),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(16, 1), 
            )
            self.decoder_head = None
        else:
            # Transformer decoder
            self.fc_fusion = None
            self.fc_pred = None
            self.decoder_head = TransformerDecoderHead(
                embed_dim=embed_dim,
                nhead=num_heads,
                num_layers=2,
                dim_feedforward=1024,
                dropout=dropout_p
            )

        self.backbone_type = backbone_type.lower()
        self.decoder_type = decoder_type.lower()
        self.embed_dim = embed_dim

    def forward(self, elevation_map, commands):
        """
        Args:
            elevation_map: (B, 1, H, W)
            commands:      (B, 3)  # e.g. [v, omega, terrain_level]
        """
        # Convert 1-channel -> 3-channel
        elevation_map = elevation_map.repeat(1, 3, 1, 1)

        # Feature extraction
        features = self.feature_extractor(elevation_map)  # => (B, E, H', W') or (B, 512, H', W')
        B, C, H_out, W_out = features.shape

        # [Optional] spatial/deformable attention
        features = self.spatial_attention(features)
        features = self.deformable_attention(features)

        if self.decoder_type == "mlp":
            # Pool => (B, C)
            x = F.adaptive_avg_pool2d(features, (1, 1)).view(B, -1)  # => (B, E)

            x = self.fc_fusion(x)  # => (B, 32)

            x_cat = torch.cat((x, commands), dim=1)

            pred = self.fc_pred(x_cat)  # => (B, 1)
            return pred

        else:
            # Transformer decoder path
            memory = features.view(B, C, -1).permute(2, 0, 1)  # => (N, B, E)
            pred = self.decoder_head(memory, commands)        # => (B, 1)
            return pred


class TransformerDecoderHeadMLL(nn.Module):
    """
    Similar to TransformerDecoderHead but returns (B, E) so we can
    predict both mean and logstd with separate linear heads.
    Now also expects commands (B, 3).
    """
    def __init__(self, embed_dim=512, nhead=8, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.cmd_embed = nn.Linear(2, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, memory, commands):
        """
        Returns: a single embedding (B, E)
        memory:   (N, B, E)
        commands: (B, 3)
        """
        B = commands.shape[0]
        query_embed = self.cmd_embed(commands).unsqueeze(0)  # (1, B, E)
        decoded = self.transformer_decoder(query_embed, memory)  # => (1, B, E)
        return decoded.squeeze(0)  # => (B, E)


class ElevationOnlyNetworkMLL(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        num_levels=1,
        num_points=4,
        dropout_p=0.2,
        freeze_backbone=False,
        backbone_type="resnet",
        decoder_type="transformer"  # or "transformer"
    ):
        super().__init__()

        # 1. Feature Extractor
        if backbone_type.lower() == "transformer":
            self.feature_extractor = TransformerFeatureExtractor(
                in_channels=3,
                embed_dim=embed_dim,
                patch_size=8,
                num_encoder_layers=8,
                nhead=num_heads,
                dim_feedforward=1024,
                dropout=dropout_p
            )
        else:
            self.feature_extractor = ResNetFeatureExtractor(
                pretrained=True,
                freeze_backbone=freeze_backbone
            )

        # 2. Attention Modules (optional)
        self.spatial_attention = SpatialAttention(kernel_size=3)
        self.deformable_attention = DeformableAttentionModule(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            num_levels=num_levels, 
            num_points=num_points
        )

        # 3. Decoder
        self.decoder_type = decoder_type.lower()
        if self.decoder_type == "mlp":
            # MLP approach
            self.fc_fusion = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.Dropout(p=dropout_p),
                nn.Linear(256, 128), 
                nn.Dropout(p=dropout_p),
                nn.Linear(128, 64),
            )
            # Now we have 3 input dims in commands => (32 + 3)
            self.fc_pred = nn.Sequential(
                nn.Linear(64 + 2, 32),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(16, 1), 
            )
            self.fc_logstd = nn.Sequential(
                nn.Linear(64 + 2, 32),
                nn.Dropout(p=dropout_p),
                nn.Linear(32, 16),
                nn.Dropout(p=dropout_p),
                nn.Linear(16, 1), 
            )
            self.decoder_head = None
        else:
            # Transformer decoder approach
            self.fc_fusion = None
            self.fc_pred = nn.Linear(embed_dim, 1)
            self.fc_logstd = nn.Linear(embed_dim, 1)
            self.decoder_head = TransformerDecoderHeadMLL(
                embed_dim=embed_dim,
                nhead=num_heads,
                num_layers=2,           # You can change
                dim_feedforward=1024,
                dropout=dropout_p
            )

    def forward(self, elevation_map, commands):
        """
        Returns: pred_mean, pred_logstd
        elevation_map: (B, 1, H, W)
        commands:      (B, 3)  # e.g. [v, omega, terrain_level]
        """
        # Convert 1-channel -> 3-channel
        elevation_map = elevation_map.repeat(1, 3, 1, 1)
        features = self.feature_extractor(elevation_map)
        B, C, H_out, W_out = features.shape

        # Optionally apply attention
        features = self.spatial_attention(features)
        features = self.deformable_attention(features)

        if self.decoder_type == "mlp":
            x = F.adaptive_avg_pool2d(features, (1, 1)).view(B, -1)
            x = self.fc_fusion(x)
            # x_cat => (B, 32 + 3)
            x_cat = torch.cat((x, commands), dim=1)

            pred_mean = self.fc_pred(x_cat)
            pred_logstd = self.fc_logstd(x_cat)
            return pred_mean, pred_logstd
        else:
            # Transformer path
            memory = features.view(B, C, -1).permute(2, 0, 1)  # (N, B, E)
            decoded = self.decoder_head(memory, commands)      # (B, E)

            pred_mean = self.fc_pred(decoded)    # (B, 1)
            pred_logstd = self.fc_logstd(decoded)  # (B, 1)
            return pred_mean, pred_logstd
            
