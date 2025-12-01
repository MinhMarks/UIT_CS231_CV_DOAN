"""
Original SALAD implementation - no cross-image modifications.
Use this to evaluate pretrained weights and establish baseline.
"""
import math
import torch
import torch.nn as nn


def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    """Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem."""
    M = M / reg

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    """Sinkhorn optimal transport matching."""
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


class SALADOriginal(nn.Module):
    """
    Original SALAD - Sinkhorn Algorithm for Locally Aggregated Descriptors.
    No cross-image modifications.
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

        if dropout > 0:
            dropout_layer = nn.Dropout(dropout)
        else:
            dropout_layer = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim),
        )

        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )

        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )

        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        Args:
            x (tuple): (features [B, C, H, W], token [B, C])
        Returns:
            descriptor [B, num_clusters * cluster_dim + token_dim]
        """
        x, t = x

        f = self.cluster_features(x).flatten(2)  # [B, cluster_dim, num_patch]
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        # Sinkhorn algorithm
        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        p = p[:, :-1, :]  # [B, num_cluster, num_patch]

        # Weighted aggregation
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        s = (f * p).sum(dim=-1)  # [B, cluster_dim, num_clusters]

        # Build descriptor
        f = torch.cat(
            [
                nn.functional.normalize(t, p=2, dim=-1),
                nn.functional.normalize(s, p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )

        return nn.functional.normalize(f, p=2, dim=-1)
