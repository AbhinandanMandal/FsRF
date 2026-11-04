import torch
import torch.nn as nn
from utils import PositionalEncodings
from mlp import MLP

class NeRFFeatureFieldLearner(nn.Module):
    """
        Learns a NeRF-style 3D field that predicts density and
        high-dimensional content features for sampled 3D points.

        The learned density is used for volume rendering weights,
        while the content features are integrated along rays to
        produce a 2D feature representation
    """
    def __init__(self, L_p: int, content_feature_dim: int):
        super().__init__()
        self.position_pos_enc = PositionalEncodings(L = L_p)
        self.mlp = MLP(position_dim = self.position_pos_enc.out_dim, content_feature_dim = content_feature_dim)

    # def forward(self, sampled_points: torch.Tensor, sampled_directions: torch.Tensor):
    def forward(self, sampled_points: torch.Tensor):
        sampled_points_pos_enc = self.position_pos_enc(sampled_points)
        # sampled_directions_pos_enc = self.direction_pos_enc(sampled_directions)
        # sigma, content_features = self.mlp(sampled_points_pos_enc, sampled_directions_pos_enc) # now every sampled point in space has a feature vector
        sigma, content_features = self.mlp(sampled_points_pos_enc) # now every sampled point in space has a feature vector
        return {
            "sigma": sigma, # sigma.shape = torch.Size([2, 1024, 64, 1])
            "content_features": content_features # content_features.shape = torch.Size([2, 1024, 64, 256])
        }

class FeatureField(nn.Module):
    def __init__(
        self,
        method: str = "nerf",
        *,
        L_p: int,
        L_d: int,
        content_feature_dim: int
    ):
        super().__init__()

        if method == "nerf":
            self.learner = NeRFFeatureFieldLearner(
                L_p,
                content_feature_dim
            )

    def forward(self, sampled_points):
        return self.learner(
            sampled_points=sampled_points
        )

