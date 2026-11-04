import torch
import torch.nn as nn



class RayGenerator(nn.Module):
    """
    > The Rays are generated entirely from the camera's intrinsic parameters (focal length, principal point, image size) and extrinsic parameters (camera pose in the world)
    > Each NeRF Dataset contains:
        - Image (RGBA img)
        - Camera Pose (transform_matrix)
        - Camera Field of View (camera_angle_x)
    > A Ray is Defined by 2 Things:
        - Origin (Camera Position)
        - Direction (ray pointing to the pixel position)
    So a Ray is Defined as,
        r(t) = o + t * d
    
    > To compute Direction:
        1. Convert pixel to Camera Co-ordinates by using focal length, image center and principal point
        2. Now convert this Camera Co-ordinates to world Camera Co-ordinates by using,
            - 4×4 Camera Pose Matrix
    
    > Now we have
        - Origin (world co-ordinates) [Shape = (H*W, 3)]
        - Direction (world co-ordinates) [Shape = (H*W, 3)]
        Per Pixel


    # Original paper
    --------------------------------------------------------------------------------------------
    x = (x, y, z)
    d = (θ, φ)
    ↓
    Implementation
    --------------------------------------------------------------------------------------------
    position  = (x, y, z)
    direction = (dx, dy, dz)


    So the Approach is simply:
        1. Compute Camera Intrinsics (This is helps us to convert from image pixel coordinates to camera coordinates)
        2. Generate Pixel Coordinates
        3. Convert Pixels to Camera Coordinates
        4. Normalize Directions
        5. Extract Camera Pose
        6. Rotate Directions into World Coordinates
        7. Generate Ray Origins
        8. Pair Every Origin with Every Direction
        9. Select Training Rays
    """
    def __init__(self):
        super().__init__()
        

    def forward(self, img_batch: dict, height = None, width = None):
        # Compute Camera Intrinsics
        # -------------------------------------------------------------------------------------------
        image = img_batch["image"] # Grab the image
        device = image.device
        dtype = image.dtype
        if height is None:
            H = image.shape[-2]
        else:
            H = height

        if width is None:
            W = image.shape[-1]
        else:
            W = width
        c_x, c_y = W / 2, H / 2 # image centers
        theta = img_batch["camera_angle_x"][0]
        f_x = W / ( 2 * torch.tan(theta / 2))
        f_y = f_x  # or we could use this, f_y = H / ( 2 * torch.tan(theta / 2))

        # Generate Pixel Coordinates
        # -------------------------------------------------------------------------------------------
        v, u = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij"
        )
        u = u.reshape(-1).float()
        v = v.reshape(-1).float()

        # Convert Pixels to Camera Coordinates, with the help of computed camera Intrinsics
        # -------------------------------------------------------------------------------------------
        x, y = (u - c_x) / f_x, -(v - c_y) / f_y
        z = -torch.ones_like(x) # z.shape == x.shape and z.shape == y.shape
        camera_dirs = torch.stack([x, y, z], dim = -1)

        # Normalize Directions
        # -------------------------------------------------------------------------------------------
        camera_dirs = camera_dirs / torch.norm(
            camera_dirs,
            dim=-1,
            keepdim = True
        )

        # Rotate Directions into World Coordinates: Camera Coordinates ----> World Coordinates
        # -------------------------------------------------------------------------------------------
        camera_pose = img_batch["camera_pose"]
        R = camera_pose[:, :3, :3]
        T = camera_pose[:, :3, 3]
        world_dirs = camera_dirs @ R.transpose(-1, -2)
        ray_origins = T.unsqueeze(1).expand_as(world_dirs)

        return world_dirs, ray_origins


import torch
import torch.nn as nn


class SpatialGridRaySampler(nn.Module):
    """
    Selects a spatially structured subset of rays.

    Instead of randomly selecting rays, this sampler
    selects a regular H_grid × W_grid subset from
    the original image.

    Example:

        Original image:
            800 × 800

        grid_size:
            32 × 32

        Output:
            1024 spatially ordered rays

    Input:
        directions : (B, H*W, 3)
        origins    : (B, H*W, 3)

    Output:
        directions : (B, H_grid*W_grid, 3)
        origins    : (B, H_grid*W_grid, 3)
        indices    : (H_grid*W_grid,)
    """

    def __init__(self, grid_size: int = 32):

        super().__init__()

        self.grid_size = grid_size

    def forward(
        self,
        directions,
        origins,
        height,
        width,
    ):

        B, n_rays, _ = directions.shape
        assert n_rays == height * width

        # Select spatially distributed pixel coordinates
        y = torch.linspace(
            0,
            height - 1,
            self.grid_size,
            device=directions.device,
        ).round().long()

        x = torch.linspace(
            0,
            width - 1,
            self.grid_size,
            device=directions.device,
        ).round().long()

        yy, xx = torch.meshgrid(
            y,
            x,
            indexing="ij"
        )

        # Convert 2D coordinates → flattened ray indices
        idx = (
            yy * width + xx
        ).reshape(-1)

        # Select rays
        directions = directions[:, idx]
        origins = origins[:, idx]
        return directions, origins, idx


import torch
import torch.nn as nn


class RandomRaySampler(nn.Module):
    """
    Randomly samples a subset of rays from the full set of image rays.

    Input:
        directions : (B, R, 3)
        origins    : (B, R, 3)

    Output:
        directions : (B, num_rays, 3)
        origins    : (B, num_rays, 3)
        indices    : (num_rays,)
    """

    def __init__(self, num_rays: int = 1024, fixed_indices = None):
        super().__init__()
        self.num_rays = num_rays
        self.fixed_indices = fixed_indices

    def forward(self, directions, origins, sample_all_rays = False):
        _, n_rays, _ = directions.shape

        if sample_all_rays:
            idx = torch.arange(
                n_rays,
                device=directions.device,
            )
        elif self.fixed_indices is not None:
            idx = self.fixed_indices.to(
                directions.device
            )
        else:
            idx = torch.randperm(
                n_rays,
                device=directions.device
            )[:self.num_rays]

        directions = directions[:, idx]
        origins = origins[:, idx]

        return directions, origins, idx

class StratifiedSampler(nn.Module):

    """
    > This Implements the Ray Sampler from the Original Paper: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
    > Instead of Querying the MLP on all 640,000 Rays we Query the MLP on self.N points sampled uniformly from [t_f, t_n] / N bins

    > Discussed in the Section-4: Volume Rendering with Radiance Fields
    """
    def __init__(self, t_n, t_f, N: int):
        super().__init__()
        self.t_n = t_n
        self.t_f = t_f
        self.N = N
    
    def forward(self, ray):
        direction, origin = ray[0], ray[1]
        batch, n_rays, dir_dim = direction.shape # batch, n_pixels, 3 (x, y, z) direction
        bins = torch.linspace(self.t_n, self.t_f, self.N + 1, device = origin.device)
        lower = bins[:-1]
        upper = bins[1:]

        lower = lower.unsqueeze(0).unsqueeze(0).expand(batch, n_rays, self.N)
        upper = upper.unsqueeze(0).unsqueeze(0).expand(batch, n_rays, self.N)
        u = torch.rand(batch, n_rays, self.N, device = origin.device)
        t_vals = lower + (upper - lower) * u

        points = (
            origin[:, :, None, :]
            + direction[:, :, None, :] * t_vals[..., :, None]
        )
        deltas = t_vals[:, :, 1:] - t_vals[:, :, :-1] # difference b/w each points # [B, R, N-1]
        last_delta = torch.full_like(deltas[:, :, :1], 1e10) # (B, R, 1)
        deltas = torch.cat([deltas, last_delta], dim=2) # (B, R, N)
        return points, t_vals, deltas # We need this for volume rendering

class PositionalEncodings(nn.Module):
    """
    > This Applies the Positional Encodings to the 'N' sampled points for all Rays before feeding into the MLP
    > Discussed in the Section-5.1: Positional encoding
    > We apply Positional Encodings because,
        Mapping the inputs to a higher
        dimensional space using high frequency functions before passing them to the MLP
        network enables better fitting of data that contains high frequency variation
    """
    def __init__(self, L: int):
        super().__init__()
        self.L = L
        self.register_buffer('freqs', torch.pow(2.0, torch.arange(self.L)))
        self.out_dim = 3 + 6 * self.L # because of concat (6L) and then attaching the original co-ordinates (+3)

    def forward(self, ray_samples):
        # ray_samples.shape = (B, R, N, 3)
        B, R, N, dir_dim = ray_samples.shape
        original_xyz = ray_samples
        ray_samples = ray_samples.unsqueeze(-2) # (B, R, N, 1, 3) 
        freqs = self.freqs.view(1, 1, 1, self.L, 1) # freq.shape = (1, 1, 1, self.L, 1)
        x = freqs * torch.pi * ray_samples # (B, R, N, L, 3)
        sin_pos = torch.sin(x) # (B, R, N, L, 3)
        cos_pos = torch.cos(x) # (B, R, N, L, 3)
        sin = sin_pos.view(B, R, N, self.L * dir_dim) # or we could also use [sin.flatten(start_dim = -2)]; same goes with cos
        cos = cos_pos.view(B, R, N, self.L * dir_dim)
        pos_encodings = torch.cat([original_xyz, sin, cos], dim = -1)
        return pos_encodings

class VolumeRenderer(nn.Module):
    """
    > This class implements the differentiable Volume Renderer proposed in
    "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
    (ECCV 2020).

    > The implementation follows the methodology described in:
        - Section 4: Volume Rendering with Radiance Fields

    > Responsibilities:
        - Compute accumulated transmittance along each ray.
        - Estimate contribution weights for every sampled point.
        - Integrate the predicted radiance field into the final RGB color.
        - Return the normalized weights required for hierarchical importance sampling.

    > Note:
        This module performs differentiable volume rendering and provides both
        the rendered pixel colors and the sampling weights used by the
        ImportanceSampler.
    """
    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, sigma, content_features, t_vals, deltas):
        # R = n_rays
        # N = n_points
        
        # ------------------------------------------------------------
        # deltas.shape = (B, R, N) | deltas.unsqueeze(-1).shape = (B, R, N, 1)
        # sigma.shape = (B, R, N, 1)
        # content_features.shape = (B, R, N, C)
        # ------------------------------------------------------------
        deltas = deltas.unsqueeze(-1)
        sigma_delta = sigma * deltas # sigma_delta.shape = (B, R, N, 1) * (B, R, N, 1) = (B, R, N, 1)
        cumulative_sum = torch.cumsum(sigma_delta, dim = 2) # (B, R, N, 1)
        cumulative_sum = cumulative_sum[:, :, :-1, :] # get all but the last dim; shape = (B, R, N - 1, 1)
        zeros = torch.zeros_like(cumulative_sum[:, :, :1, :]) # (B, R, 1, 1)
        cumulative  = torch.cat([zeros, cumulative_sum], dim  = 2) # (B, R, N, 1)

        # Now get alpha, transmittance and weights
        alpha = 1 - torch.exp(-sigma_delta) # (B, R, N, 1)
        transmittance = torch.exp(-cumulative) # (B, R, N, 1)
        weights = transmittance * alpha # weights.shape = (B, R, N, 1) | (B, 1024, 64, 1) # This weights will then be used for rendering the final RGB and for constructing the PDF for hierarchical importance sampling

        rendered_content_features = torch.sum(content_features * weights, dim  = 2)
        depth = torch.sum(weights * t_vals.unsqueeze(-1), dim = 2) # optional
        return {
            "rendered_content_features": rendered_content_features,
            "weights": weights,
            "alpha": alpha,
            "transmittance": transmittance,
            "depth": depth
        }

    def render_with_weights(self, content_features, weights, t_vals = None):
        """
        Stage-2 path: integrate features using PRE-COMPUTED weights
        (frozen geometry from Stage-1) instead of recomputing them
        from sigma/t_vals/deltas.

        content_features: (B, R, N, C)  -- e.g. SICT-stylized features
        weights:          (B, R, N, 1)  -- cached from Stage-1
        t_vals:           (B, R, N)     -- optional, only if you want depth too
        """
        rendered_content_features = torch.sum(content_features * weights, dim=2)

        out = {"rendered_content_features": rendered_content_features}

        if t_vals is not None:
            out["depth"] = torch.sum(weights * t_vals.unsqueeze(-1), dim=2)

        return out

def stage_01_outputs():
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from config import config
    import torch
    from stage1 import PreFeatureGridTrainingStagePipeline
    from load_dataset import NeRFSyntheticDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/stage_01/model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Rebuild model with the EXACT config the checkpoint was trained with
    config = checkpoint["config"]

    model = PreFeatureGridTrainingStagePipeline(config=config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, best_loss={checkpoint['best_loss']:.6f}")


    test_dataset = NeRFSyntheticDataset(
        root_dir="./data/nerf_synthetic/lego",
        split="test",
        image_size=config["img_H"]
    )

    test_loader = DataLoader(
        test_dataset,       
        batch_size=5,
        shuffle=True
    )

    test_batch = next(iter(test_loader))

    test_batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in test_batch.items()
    }

    model.eval()

    with torch.no_grad():

        # Render all 5 images
        rendered_features = model.feature_pipeline.infer(
            test_batch,
            chunk_size=64
        )["rendered_content_features"]

        # Decode all 5 feature maps → RGB
        reconstructed_imgs = model.decoder(
            rendered_features
        )

    # Move to CPU
    reconstructed_imgs = reconstructed_imgs.cpu()
    target_imgs = test_batch["target_rgb"].cpu()

    batch_size = reconstructed_imgs.shape[0]

    # Create figure + axes
    fig, axes = plt.subplots(
        2,
        batch_size,
        figsize=(4 * batch_size, 8)
    )

    for i in range(batch_size):

        # Target
        axes[0, i].imshow(
            target_imgs[i].permute(1, 2, 0).numpy()
        )
        axes[0, i].set_title(f"Target {i + 1}")
        axes[0, i].axis("off")

        # Reconstruction
        axes[1, i].imshow(
            reconstructed_imgs[i].permute(1, 2, 0).numpy()
        )
        axes[1, i].set_title(f"Recon {i + 1}")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Target", fontsize=14)
    axes[1, 0].set_ylabel("Recon", fontsize=14)

    plt.tight_layout()
    plt.show()