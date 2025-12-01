import torch
import torch.nn as nn


class CrossImageEncoder(nn.Module):
    """
    Cross-image Transformer encoder for learning relationships between
    multiple images from the same place.
    
    This module enhances cluster features by allowing images from the same
    place to share information through self-attention.
    """

    def __init__(self, cluster_dim=128, num_clusters=64, img_per_place=4, num_layers=2):
        super().__init__()

        self.cluster_dim = cluster_dim
        self.num_clusters = num_clusters
        self.img_per_place = img_per_place

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cluster_dim * num_clusters,
            nhead=8,
            dim_feedforward=1024,
            activation="gelu",
            dropout=0.1,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, s):
        """
        Apply cross-image attention enhancement.
        
        Args:
            s: [B, cluster_dim, num_clusters] where B = num_places * img_per_place
        Returns:
            s_enhanced: [B, cluster_dim, num_clusters] with residual connection
        """
        B, cluster_dim, num_clusters = s.shape
        num_places = B // self.img_per_place

        # Group by place
        s_place = s.view(num_places, self.img_per_place, cluster_dim, num_clusters)
        s_seq = s_place.flatten(2)  # [num_places, img_per_place, embed_dim]
        s_seq = s_seq.permute(1, 0, 2)  # [img_per_place, num_places, embed_dim]

        # Cross-image attention
        s_encoded = self.encoder(s_seq)

        # Reshape back
        s_encoded = s_encoded.permute(1, 0, 2)
        s_encoded = s_encoded.contiguous().view(B, cluster_dim, num_clusters)
        s_encoded = nn.functional.normalize(s_encoded, p=2, dim=1)

        return s + s_encoded  # Residual connection
