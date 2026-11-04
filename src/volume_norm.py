import torch.nn as nn
import torch

class VolumeAdaptiveInstanceNorm(nn.Module):
    """
    During evaluation, volume adaptive IN uses learned mean and standard-deviation,
    discarding dependency over the sampled point batch's holistic statistics
    """
    def __init__(self,
            num_features: int = 256,
            eps: float = 1e-5,
            momentum: float = 1e-4,
            affine: bool = False,
            track_running_stats: bool = True
        ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.instance_norm = nn.InstanceNorm1d(
            num_features = self.num_features, eps = self.eps, momentum = self.momentum, affine = self.affine, track_running_stats = self.track_running_stats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        expects the shape of tensor x as,

        x.shape = B, R, N, C
        
        B: Batch
        R: Rays
        N: Number of Sampled Points per Rays
        C: Feature dim

        and returns torch.Tensor of Shape = (B, R, N, C)  
        """
        B, N_rays, N_samples, C = x.shape
        x = x.permute(0, 1, 3, 2) # x.shape = (B, n_rays, n_points, C) --> (B, n_rays, C, n_points)
        x = x.reshape(B * N_rays, C, N_samples)

        # Because 'nn.InstanceNorm1d' expects Tensor Shape as (N, C, L)
        x = self.instance_norm(x) # x_norm.shape = (B * N_rays, C, N_samples)
        x = x.reshape(B, N_rays, C, N_samples)
        x = x.permute(0, 1, 3, 2) # Back to (B, n_rays, n_points, C)
        return x # (B, R, N, C)