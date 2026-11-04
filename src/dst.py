import torch
import torch.nn as nn
from vgg import normalize_vgg, VGGFeatureExtractor

class StyleStatistics(nn.Module):
    def __init__(self):
        super().__init__()

    def __get_cov_matrix(self, F_s: torch.Tensor):
        """
        F_s.shape = (B, C, N); where N = H*W

        F_s is already shifted by it's mean value, directly compute the covariance matrix
        """
        F_cov = (F_s @ F_s.transpose(-1, -2)) / F_s.shape[-1]
        return F_cov

    def forward(self, F_s):
        """
        Takes F_s (features extracted by the VGG-19 (relu3_1) of the Style Image)
        input shape: (B, C, H, W)
        output: [mean, std and cov] of this style image feature space
        """
        B, C, H, W = F_s.shape
        F_s = F_s.reshape(B, C, -1)

        mean = torch.mean(F_s, dim = -1, keepdim = True) # dim = -1 is spatial dim
        std = torch.std(F_s, dim = -1, keepdim = True) # dim = -1 is spatial dim

        F_s = F_s - mean
        F_cov = self.__get_cov_matrix(F_s)

        return mean, std, F_cov

class CNN(nn.Module):
    """
    This CNN Class Assumes that we are using features from relu3_1 
    """
    def __init__(self, T_size: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            # nn.ReLU(inplace = True),
            nn.LeakyReLU(0.01, inplace = True),
            nn.Conv2d(128, 64, 3, 1, 1),
            # nn.ReLU(inplace = True),
            nn.LeakyReLU(0.01, inplace = True),
            nn.Conv2d(64, T_size, 3, 1, 1),
            nn.LeakyReLU(0.01, inplace = True),
        )

        self.linear = nn.Linear(
            T_size * T_size, T_size * T_size
        )

        self.T_size = T_size

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        expects the shape as (B, C, H, W) and returns embeddings of x by following
        "Learning Linear Transformations for Fast Image and Video Style Transfer",
        from Section 3.2: Learning Transformation T

        returns: embeddings of x: (B, T, T)
        """
        B, C, H, W = x.shape

        x_prime = self.conv(x) # [B, T_size, H, W]

        B, T, H, W = x_prime.shape
        x_prime = x_prime.reshape(B, T, -1) # [B, T, N], N = H * W
        x_cov = self.__get_cov_matrix(x_prime) # [B, T, T]

        x_cov = x_cov.flatten(1)
        return self.linear(x_cov)

        
    def __get_cov_matrix(self, x: torch.Tensor):
            """
            returns the covariance matrix of x
            x.shape = (B, T, N) where N = H*W
            """
            x = x - x.mean(dim = -1, keepdim=True) # (B, T, N) @ (B, T, N).T(-2, -1) = (B, T, T)
            x_cov = (x @ x.transpose(-1, -2)) / x.shape[-1]
            return x_cov

    def _initialize_weights(self):
        nn.init.zeros_(self.linear.weight)
        with torch.no_grad():
            identity = torch.eye(self.T_size).flatten()
            self.linear.bias.copy_(identity)
     
class LinearTransformation(nn.Module):
    def __init__(self, T_size: int = 32, *, device: torch.device):
        super().__init__()

        # Single branch — official StyleRF derives T from the style
        # image's own feature covariance only (see renderer.py: the
        # stats/T are computed once from style_feature before the
        # ray-chunk loop even starts — there's no content input to it).
        # self.conv_net1 = CNN(T_size = T_size).to(device)
        self.conv_net = CNN(T_size = T_size).to(device)
        self.T_size = T_size


    def forward(self, F_s: torch.Tensor):
        """
        Implements the exact methodology proposed in the paper
        "Learning Linear Transformations for Fast Image and Video Style Transfer",
        following Section 3.2: Learning Transformation T

        Input:
            F_c.shape = B*C*H*W
            F_s.shape = B*C*H*W

        returns:
            Learned Transformation Matrix T of shape = [B, C, C]
        """

        # content_embedding = self.conv_net1(F_c) # [B, T_size, T_size]
        style_embedding = self.conv_net(F_s) # [B, T_size, T_size]

        B = style_embedding.shape[0]
        T = style_embedding.reshape(B, self.T_size, self.T_size)
        return T # [B, T_size, T_size]

class DST(nn.Module):
    def __init__(self, content_feature_dim: int, T_size: int, device: torch.device, match_decoder_distribution: bool = False):
        super().__init__()
        self.style_statistics  = StyleStatistics()
        self.style_transformation_matrix = LinearTransformation(T_size = T_size, device = device)
        self.match_decoder_distribution = match_decoder_distribution
        self.T_size = T_size

        # Expand T_size -> content_feature_dim after applying T
        self.unzipper = nn.Conv1d(
            T_size, content_feature_dim, kernel_size = 1, bias = False
        ).to(device)

        self.device = device

        # nn.init.xavier_uniform_(self.zipper.weight)
        nn.init.xavier_uniform_(self.unzipper.weight)

        # self.gamma = nn.Parameter(torch.ones(1, 1, content_feature_dim))
        # self.beta  = nn.Parameter(torch.zeros(1, 1, content_feature_dim))

    def forward(self, volume_rendered_content_features, vgg_content_features, vgg_style_features, weights, original_rendered_content_features=None):
        """
        volume_rendered_content_features:
            Content features after differentiable Volume Rendering.
            Shape = [B, R, C]

            B = batch size
            R = number of rays
            C = content_feature_dim (256)

        vgg_content_features:
            Content feature maps extracted from the ReLU3_1 layer of a pretrained
            VGG-19 network.
            Shape = [B, 256, H, W]

        vgg_style_features:
            Style feature maps extracted from the ReLU3_1 layer of a pretrained
            VGG-19 network.
            Shape = [B, 256, H, W]

        weights:
            Volume rendering weights of every sampled point along each ray,
            obtained from the VolumeRenderer.
            Shape = [B, R, N, 1]

        Returns:
            Stylized feature representation obtained by following the complete
            Deferred Style Transformation (DST) pipeline proposed in StyleRF.
            Shape = [B, R, C]
        """

        # Test-05
        # ------------------------------------------------------------------------------------------------
        mean, std, _ = self.style_statistics(F_s = vgg_style_features)
        mean = mean.transpose(-1, -2)
        std = std.transpose(-1, -2)
        # ------------------------------------------------------------------------------------------------

        
        content = volume_rendered_content_features.transpose(-2, -1)   # (B, C, R)
        # content_compressed = self.zipper(content)                       # (B, T_size, R)

        # Test-04
        # -------------------------------------
        T = self.style_transformation_matrix(F_s = vgg_style_features)  # (B, T_size, T_size)
        # print("T norm:", T.norm().item(), "T diag mean:", torch.diagonal(T[0]).mean().item(), "T off-diag mean:", (T[0].sum() - torch.diagonal(T[0]).sum()).item() / (T[0].numel() - T[0].shape[0]))
        # -------------------------------------

        style = torch.bmm(T, content)                        # (B, T_size, R)
        style = self.unzipper(style).permute(0, 2, 1)                   # (B, R, C)

        # Test-05
        # ------------------------------------------------------------------------------------------------
        w_r = weights.sum(dim=-2, keepdim=True).squeeze(-1)
        # ------------------------------------------------------------------------------------------------


        F_cs = (style * std) + mean * w_r

        # F_cs = self.gamma * F_cs + self.beta # for matching the distribution of learned CNN's decoder

        # TEMPORARY DEBUG: Match F_cs distribution to SICT content feature distribution
        if self.match_decoder_distribution:
            decoder_features = original_rendered_content_features

            target_mean = decoder_features.mean(
                dim=(0, 1),
                keepdim=True
            )

            target_std = decoder_features.std(
                dim=(0, 1),
                keepdim=True
            )

            fcs_mean = F_cs.mean(
                dim=(0, 1),
                keepdim=True
            )

            fcs_std = F_cs.std(
                dim=(0, 1),
                keepdim=True
            )

            F_cs = (
                (F_cs - fcs_mean) /
                (fcs_std + 1e-6)
            ) * target_std + target_mean

        return F_cs

class DSTPipeline(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.dst = DST(
            content_feature_dim = config["content_feature_dim"],
            T_size = config["T_size"],
            device = device,
        ).to(device)

        self.vgg_feature_extractor = VGGFeatureExtractor(device)

    def forward(self, batch, sict_dict: dict):
        content_img = batch["target_rgb"]   # (B, 3, H, W) — composited RGB
        style_img = batch["style_img"]      # (B, 3, H, W)

        F_c = self.vgg_feature_extractor(normalize_vgg(content_img))["relu3_1"]
        F_s = self.vgg_feature_extractor(normalize_vgg(style_img))["relu3_1"]
        original_rendered_content_feature = sict_dict["original_rendered_content_features"]
        F_cs = self.dst(
            volume_rendered_content_features = sict_dict["rendered_content_features"],
            vgg_content_features = F_c,
            vgg_style_features = F_s,
            weights = sict_dict["weights"],
            original_rendered_content_features = original_rendered_content_feature
        )

        return {
            "F_c": F_c,
            "F_s": F_s,
            "F_cs": F_cs
        }