import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from utils import RayGenerator, SpatialGridRaySampler, StratifiedSampler, VolumeRenderer
from feature_field_learner import FeatureField
from config import config
from decoder import CNNDecoder
from vgg import normalize_vgg, VGGFeatureExtractor


import torch.nn as nn
import torch
import torch.utils.checkpoint as cp

class PreDSTFeaturePipeline(nn.Module):
    """
    This class is responsible to compute the Content Feature map by following the SICT discussed in the paper
    it offere's how you wanna generate the content feature map (NeRF or TensoRF)

    The output of this class is then directly be used by the DST
    """
    def __init__(self, num_rays: int, t_n: int, t_f: int, num_sampled_points: int, L_p: int, L_d: int, content_feature_dim: int, embed_dim: int, method: str = "nerf", *, volume_renderer_eps: float, train_chunk_size: int = 2048):
        super().__init__()
        self.ray_generator = RayGenerator()
        self.spatial_ray_sampler = SpatialGridRaySampler(grid_size=config["grid_size"])
        self.stratified_sampler = StratifiedSampler(t_n = t_n, t_f = t_f, N = num_sampled_points)
        self.learner = FeatureField(method = method, L_p = L_p, L_d = L_d, content_feature_dim = content_feature_dim)
        self.volum_renderer = VolumeRenderer(volume_renderer_eps)
        self.train_chunk_size = train_chunk_size

    def _render_chunk(self, dirs_chunk, origins_chunk):
        sampled_points, t_vals, deltas = self.stratified_sampler([dirs_chunk, origins_chunk])
        learner_dict = self.learner(sampled_points=sampled_points)

        sigma = learner_dict["sigma"]
        content_features = learner_dict["content_features"]

        rendered = self.volum_renderer(
            sigma = sigma,
            content_features = content_features,
            t_vals = t_vals, deltas = deltas
        )
        return (sigma, content_features, rendered["rendered_content_features"], rendered["weights"],
                rendered["alpha"], rendered["transmittance"], rendered["depth"])

    def forward(self, batch):
        directions, origins = self.ray_generator(batch)  # Generate rays

        device = next(self.parameters()).device
        directions = directions.to(device)
        origins = origins.to(device)
        
        if self.training:
            H = batch["image"].shape[-2]
            W = batch["image"].shape[-1]

            directions, origins, spatial_idx = self.spatial_ray_sampler(
                directions=directions,
                origins=origins,
                height=H,
                width=W,
            )
        else:
            spatial_idx = None

        sigmas, feats, weights, alphas, trans, depths, pre_render_features = [], [], [], [], [], [], []
        for start in range(0, directions.shape[1], self.train_chunk_size):
            d = directions[:, start:start + self.train_chunk_size]
            o = origins[:, start:start + self.train_chunk_size]

            if self.training:
                sigma, content_features, f, w, a, t, dp = cp.checkpoint(self._render_chunk, d, o, use_reentrant=False)
            else:
                sigma, content_features, f, w, a, t, dp = self._render_chunk(d, o)

            sigmas.append(sigma)
            pre_render_features.append(content_features)
            feats.append(f)
            weights.append(w)
            alphas.append(a)
            trans.append(t)
            depths.append(dp)

        return {
            "content_features": torch.cat(pre_render_features, dim=1), # Input to SICT (B, R, N, C) [before Volume rendering]
            "sigmas": torch.cat(sigmas, dim = 1),
            "rendered_content_features": torch.cat(feats, dim=1),
            "spatial_idx": spatial_idx,
            "weights": torch.cat(weights, dim=1),
            "depth": torch.cat(depths, dim=1),
            "alpha": torch.cat(alphas, dim=1),
            "transmittance": torch.cat(trans, dim=1),
        }

    @torch.no_grad()
    def infer(self, batch, chunk_size=512):
        """
        Grid-resolution inference using chunked rays.
        Reuses the same _render_chunk helper — no checkpointing needed
        since there's no backward pass at inference time.
        """
        directions, origins = self.ray_generator(batch)

        H = batch["image"].shape[-2]
        W = batch["image"].shape[-1]
        directions, origins, _ = self.spatial_ray_sampler(
            directions=directions, origins=origins, height=H, width=W
        )

        device = next(self.parameters()).device
        directions = directions.to(device)
        origins = origins.to(device)

        # outputs = []
        sigmas = []
        content_features = []
        rendered_features = [] # after volume rendering
        weights = []
        alphas = []
        trans = []
        depths = []

        content_features = []
        for start in range(0, directions.shape[1], chunk_size):
            d = directions[:, start:start + chunk_size]
            o = origins[:, start:start + chunk_size]
            
            (sigma, cf, f, w, a, t, dpt) = self._render_chunk(d, o)

            sigmas.append(sigma)
            content_features.append(cf)
            rendered_features.append(f)
            weights.append(w)
            alphas.append(a)
            trans.append(t)
            depths.append(dpt)
        
        return {
            "content_features": torch.cat(content_features, dim = 1),
            "sigma": torch.cat(sigmas, dim = 1),
            "rendered_content_features": torch.cat(rendered_features, dim = 1),
            "weights": torch.cat(weights, dim = 1),
            "alpha": torch.cat(alphas, dim = 1),
            "transmittance": torch.cat(trans, dim = 1),
            "depth": torch.cat(depths, dim = 1)
        }

class PreFeatureGridTrainingStagePipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["num_rays"] == config["grid_size"] ** 2, (
            "num_rays must equal (grid_size ** 2)"
        )
        self.feature_pipeline = PreDSTFeaturePipeline(
            num_rays = config["num_rays"],
            t_n = config["t_n"],
            t_f = config["t_f"],
            num_sampled_points = config["n_points"],
            L_p = config["position_L"],
            L_d = config["direction_L"],
            content_feature_dim = config["content_feature_dim"],
            embed_dim = config["embed_dim"],
            method = config["method"],
            volume_renderer_eps = config["volume_renderer_eps"],
            train_chunk_size = config.get("train_chunk_size", 2048),
        )

        self.decoder = CNNDecoder(C = config["content_feature_dim"], H = config["img_H"], W = config["img_W"], grid_size=config["grid_size"])

    def forward(self, batch: torch.Tensor):
        pre_dst = self.feature_pipeline(batch)
        rendered_features = pre_dst["rendered_content_features"]
        rendered_img = self.decoder(rendered_features)

        return {
            "rendered_features": rendered_features, # we will need this for feature grid loss
            "rendered_img": rendered_img, # we will need this for reconstruction loss
            "weights": pre_dst["weights"],
            "depth": pre_dst["depth"],
            "alpha": pre_dst["alpha"],
            "spatial_idx": pre_dst["spatial_idx"],
        }

class FeatureGridLoss(nn.Module):
    """
    Computes the three losses required for Feature Grid Training:

        1. Feature reconstruction loss
        2. RGB reconstruction loss
        3. Perceptual loss
    """

    def __init__(self, H: int, W: int, device: torch.device, grid_size: int):
        super().__init__()
        self.device = device
        self.vgg_feature_extractor = VGGFeatureExtractor(self.device)
        self.H = H
        self.W = W
        self.grid_size = grid_size


    # def forward(self, rendered_features: torch.Tensor, rendered_img: torch.Tensor, target_img: torch.Tensor, target_features: torch.Tensor):
    def forward(self, batch, output: torch.Tensor):
        content_img = batch["target_rgb"].to(self.device)
        with torch.no_grad():
            target_vgg_features  = self.vgg_feature_extractor(normalize_vgg(content_img)) # (B, 256, H, W)

        # 1) Feature Reconstruction Loss
        # Bilinear upsampling
        vgg_features  = F.interpolate(
            target_vgg_features["relu3_1"],
            mode = "bilinear",
            size=(self.grid_size, self.grid_size),
            align_corners=False
        )
        vgg_features = vgg_features.permute(0, 2, 3, 1) # (B, H, W, 256) --> (B, H, W, 256)
        vgg_features = vgg_features.reshape(vgg_features.shape[0], -1, 256) # (B, H*W, 256)
        # target_features = vgg_features[:, output["spatial_idx"], :] # (B, 1024, 256) # change_1

        # feature_loss = F.mse_loss( # change_2
        #     output["rendered_features"].to(self.device), target_features
        # )
        feature_loss = F.mse_loss(
            output["rendered_features"].to(self.device), vgg_features
        )

        # 2) RGB Reconstruction Loss
        rgb_loss = F.mse_loss(
            output["rendered_img"], content_img
        )

        # 3) Perceptual loss
        perceptual_rendered_features = self.vgg_feature_extractor(normalize_vgg(output["rendered_img"].to(self.device)))
        perceptual_loss = F.mse_loss(
            perceptual_rendered_features["relu3_1"],
            target_vgg_features["relu3_1"]
        )

        return {
            "feature_loss": feature_loss,
            "rgb_loss": rgb_loss,
            "perceptual_loss": perceptual_loss,
        }