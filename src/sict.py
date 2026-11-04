import torch
import torch.nn as nn
import torch.nn.functional as F
from volume_norm import VolumeAdaptiveInstanceNorm
from utils import VolumeRenderer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

class SICT(nn.Module):
    """
    The goal of SICT is to transform the extracted features F_i
    so that they can be better stylized
    """
    def __init__(self,
                input_dim: int = 256,
                embed_dim: int = 32,
                eps: float = 1e-5,
                momentum: float = 1e-4,
                affine: bool = False,
                track_running_stats: bool = True,
                *,
                train_chunk_size: int,
            ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.train_chunk_size = train_chunk_size

        self.instance_norm = VolumeAdaptiveInstanceNorm(
            num_features = input_dim,
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats
        )

        # Self-attention modules
        self.q = nn.Conv1d(
            input_dim, embed_dim, kernel_size = 1
        )
        self.k = nn.Conv1d(
            input_dim, embed_dim, kernel_size = 1
        )
        self.v = nn.Conv1d(
            input_dim, embed_dim, kernel_size = 1
        )

        # self.unzipper = nn.Conv2d(
        #     embed_dim, input_dim, kernel_size = 1
        # )

        self._initialize_weights()

    # Chunking because it won't fit in my VRAM
    def _render_chunk(self, x_chunk):
        """
        Process one chunk of 3D content features.
        Input:
            x_chunk: [B, R_chunk, N, C]

        Output:
            [B, R_chunk, N, embed_dim]
        """

        B, R_chunk, N, C = x_chunk.shape
        q_x = x_chunk.reshape(B, R_chunk * N, C)
        q_x = q_x.transpose(1, 2).contiguous()

        Q = self.q(q_x)
        K = self.k(q_x)
        V = self.v(q_x)
        # [B, 32, S], where S = R_chunk*N

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        # Q, K: [B, S, 32], where S = R_chunk*N
        
        c_cov = (Q.unsqueeze(-1)* K.unsqueeze(-2)) # [B, S, 32, 32]

        attn = torch.softmax(c_cov, dim=-1) # # [B, S, 32, 32]

        V = V.transpose(1, 2).unsqueeze(-1) # V -> # [B, S, 32, 1]
        transformed = torch.matmul(attn, V) # [B, S, 32, 1]
        transformed = transformed.squeeze(-1) # [B, S, 32]
        transformed = transformed.reshape(B,R_chunk,N,self.embed_dim)
        return transformed

    def forward(self, x):
        """
        Apply SICT using ray-wise chunking.

        expects the shape of tensor x as,

        x.shape = B, R, N, C
        
        B: Batch
        R: Rays
        N: Number of Sampled Points per Rays
        C: Feature dim

        returns a Tensor of same Shape as x,
        i.e output Tensor Shape = [B, R, N, C]
        """
        B, R, N, C = x.shape
        assert C == self.input_dim, (f"Expected feature dimension {self.input_dim}, "f"got {C}")

        x = self.instance_norm(x)
        outputs = []

        for start in range(0, R, self.train_chunk_size):
            x_chunk = x[: , start:start + self.train_chunk_size,:,:]

            if self.training:
                output_chunk = cp.checkpoint(
                    self._render_chunk,
                    x_chunk,
                    use_reentrant=False
                )

            else:
                output_chunk = self._render_chunk(
                    x_chunk
                )

            outputs.append(output_chunk)
        return torch.cat(outputs, dim = 1)

    def _initialize_weights(self):
        for layer in [self.q, self.k, self.v]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Zero-init the unzipper so SICT starts as a near-no-op:
        # attention output initially contributes ~nothing until trained.
        # (Same trick used for adapter/LoRA-style modules.)

        # as i don't wanna use residual (for experiments) i'm removing it
        # nn.init.zeros_(self.unzipper.weight)
        # nn.init.zeros_(self.unzipper.bias)


class SICTPipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sict = SICT(
            input_dim=config["content_feature_dim"],
            embed_dim=config["embed_dim"],
            train_chunk_size = config["train_chunk_size"]
        )
        self.volume_renderer = VolumeRenderer()

    def forward(self, features):
        """
        features: dict returned by PreDSTFeaturePipeline.infer() / forward():
            {content_features, sigma, rendered_content_features,
             weights, alpha, transmittance, depth}
        """
        content_features = features["content_features"]   # (B, R, N, C)
        weights = features["weights"]

        # Original 256-D content → volume rendering
        original_rendered = self.volume_renderer.render_with_weights(
            content_features=content_features,
            weights=weights,
        )

        sict_features = self.sict(content_features)          # (B, R, N, C) — still content, now sampling-invariant
        # print("SICT output:", sict_features.shape)

        rendered = self.volume_renderer.render_with_weights(  # (B, R, C)
            content_features = sict_features,
            weights = weights,
        )

        # print(
        #     "Rendered output:",
        #     rendered["rendered_content_features"].shape
        # )

        return {
            "original_rendered_content_features": original_rendered["rendered_content_features"],  # [B,R,256]
            "sict_content_features": sict_features,
            "rendered_content_features": rendered["rendered_content_features"],  # (B, R, C) —> 2D grid, content-only
            "weights": weights,
            "depth": features.get("depth"),
        }