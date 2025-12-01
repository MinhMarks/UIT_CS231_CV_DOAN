import math
import torch
import torch.nn as nn


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem."""
    M = M / reg

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Code adapted from OpenGlue, MIT license
def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm


class SALADBase(nn.Module):
    """
    Base SALAD module - the original single-image aggregator.
    Can be loaded with pretrained weights independently.
    """

    def __init__(
        self,
        num_channels=1536,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim),
        )

        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )

        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )

        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        Args:
            x (tuple): (features [B, C, H, W], token [B, C])
        Returns:
            s: cluster features [B, cluster_dim, num_clusters]
            t: token features [B, token_dim]
        """
        x, t = x

        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        p = p[:, :-1, :]

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        s = (f * p).sum(dim=-1)

        return s, t


class CrossImageEncoder(nn.Module):
    """
    Cross-image Transformer encoder for learning relationships between
    multiple images from the same place.
    """

    def __init__(self, cluster_dim=128, num_clusters=64, img_per_place=4):
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, s):
        """
        Args:
            s: [B, cluster_dim, num_clusters] where B = num_places * img_per_place
        Returns:
            s_enhanced: [B, cluster_dim, num_clusters]
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

        return s + s_encoded  # Residual


class SALAD(nn.Module):
    """
    Full SALAD with optional cross-image learning.
    
    Usage:
        # Load pretrained base weights only
        model = SALAD(...)
        model.load_base_weights('pretrained_salad.pth')
        
        # Or load full model
        model.load_state_dict(torch.load('full_model.pth'))
    """

    def __init__(
        self,
        num_channels=1536,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
        img_per_place=4,
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.img_per_place = img_per_place

        # Base SALAD (can load pretrained weights)
        self.base = SALADBase(
            num_channels=num_channels,
            num_clusters=num_clusters,
            cluster_dim=cluster_dim,
            token_dim=token_dim,
            dropout=dropout,
        )

        # Cross-image encoder (trained from scratch or separately)
        self.cross_encoder = CrossImageEncoder(
            cluster_dim=cluster_dim,
            num_clusters=num_clusters,
            img_per_place=img_per_place,
        )

    def load_base_weights(self, checkpoint_path, strict=True):
        """
        Load pretrained weights for SALADBase only.
        Cross-image encoder remains randomly initialized.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce key matching
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Filter and remap keys for base module
        base_state_dict = {}
        for k, v in state_dict.items():
            # Remove common prefixes from full model checkpoints
            new_key = k
            for prefix in ["aggregator.", "model.aggregator.", "base."]:
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    break

            # Only load base SALAD keys
            if any(
                new_key.startswith(name)
                for name in ["token_features", "cluster_features", "score", "dust_bin"]
            ):
                base_state_dict[new_key] = v

        self.base.load_state_dict(base_state_dict, strict=strict)
        print(f"Loaded {len(base_state_dict)} base weights from {checkpoint_path}")

    def freeze_base(self):
        """Freeze base SALAD weights, only train cross-image encoder."""
        for param in self.base.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        """Unfreeze base SALAD weights."""
        for param in self.base.parameters():
            param.requires_grad = True

    def _build_descriptor(self, s, t):
        """Build final normalized descriptor."""
        f = torch.cat(
            [
                nn.functional.normalize(t, p=2, dim=-1),
                nn.functional.normalize(s, p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )
        return nn.functional.normalize(f, p=2, dim=-1)

    def forward(self, x, use_cross_image=True):
        """
        Args:
            x (tuple): (features [B, C, H, W], token [B, C])
            use_cross_image: Enable cross-image learning (training/database mode)
        """
        s, t = self.base(x)

        # Apply cross-image only during training with valid batch size
        if use_cross_image and self.training and s.shape[0] % self.img_per_place == 0:
            s = self.cross_encoder(s)

        return self._build_descriptor(s, t)

    def forward_single(self, x):
        """Forward for single image (query inference)."""
        s, t = self.base(x)
        return self._build_descriptor(s, t)

    def forward_database(self, x):
        """
        Forward for database images with cross-image enhancement.
        Use this when creating database embeddings with grouped images.
        """
        s, t = self.base(x)
        if s.shape[0] % self.img_per_place == 0:
            s = self.cross_encoder(s)
        return self._build_descriptor(s, t)
