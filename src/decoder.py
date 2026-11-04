import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNDecoder(nn.Module):
    """
    Decode the 32-dimensional rendered SICT feature representation into RGB
    """
    def __init__(self, C: int, H: int, W: int, grid_size: int):
        super().__init__()
        assert H % grid_size == 0, "img_H should be divisible by grid_size for clean upsampling"
        total_factor = H // grid_size
        self.up1 = int(total_factor ** 0.5) or 1
        self.up2 = total_factor // self.up1

        self.C = C
        self.H = H
        self.W = W

        # Step 1 — Decide the decoder's contract.
        # This decoder is only ever valid on the grid_size × grid_size
        # feature grid it was trained on (e.g. 32×32). Treat this as an
        # invariant, not a convenience.
        self.expected_grid = grid_size

        self.conv1 = nn.Conv2d(self.C, 128, kernel_size = 3, padding = 1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=128)

        self.conv2 = nn.Conv2d(128, 64, kernel_size = 3, padding = 1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=64)

        self.conv3 = nn.Conv2d(64, 32, kernel_size = 3, padding = 1)
        self.norm3 = nn.GroupNorm(num_groups=8, num_channels=32)

        self.rgb = nn.Conv2d(32, 3, kernel_size = 3, padding = 1)
        # self.input_scale = nn.Parameter(torch.tensor(50.0))

        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        self._initialize_weights()


    def _initialize_weights(self):
        # Conv + ReLU layers
        for layer in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(
                layer.weight,
                mode="fan_in",
                nonlinearity="leaky_relu"
            )
            nn.init.zeros_(layer.bias)

        # Final RGB layer
        nn.init.xavier_uniform_(self.rgb.weight)
        nn.init.zeros_(self.rgb.bias)
        

    def forward(self, x: torch.Tensor):
        """
        Here x represents the Feature grid after the volume rendering
        The aim is to render the RBG Image from this representation

        Input: x.shape = (B, R, C) ---> rendered feature from the PreDSTPipeline
        output: I.shape = (B, 3, H, W) ---> Rendered RGB Image
        """

        B, R, C = x.shape

        h = w = int(math.sqrt(R))

        # Step 2 — Enforce the contract with an assert, not a hope.
        # This is what turns the old silent-garbage failure (any perfect
        # square R quietly accepted) into an immediate, loud crash the
        # moment train/eval resolutions drift apart.
        assert h * w == R, "R must be a perfect square"
        assert h == self.expected_grid, (
            f"CNNDecoder trained for {self.expected_grid}×{self.expected_grid}, "
            f"got {h}×{h}"
        )

        x = x.transpose(1, 2) # (B, R, C) -> (B, C, R)
        x = x.reshape(B, C, h, w) # (B, C, sqrt(R), sqrt(R))

        # Apply the input scaling
        # x = x * self.input_scale
        
        # Apply the CNN-decoder
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        # Step 3 — relative scale_factor instead of hard-coded absolute size.
        # Was: F.interpolate(x, size=(128, 128), ...)
        # 32×32 * 4 == 128×128, but now the ratio is explicit and travels
        # with whatever input resolution actually arrives.
        x = F.interpolate(
            x,
            scale_factor = self.up1,
            mode="bilinear",
            align_corners=False
        )

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        # Was: F.interpolate(x, size=(256, 256), ...)
        # 128×128 * 2 == 256×256, same reasoning as above.
        x = F.interpolate(
            x,
            scale_factor = self.up2,
            mode="bilinear",
            align_corners=False
        )

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        # Left as absolute size on purpose — this is the true final output
        # resolution (self.H, self.W), not an internal proxy resolution,
        # so it isn't part of the train/eval mismatch bug.
        x = F.interpolate(
            x,
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False
        )

        x = self.rgb(x)
        # print(f"RGB details: ")
        # print(f"rgb.min: {x.min()}")
        # print(f"rgb.max: {x.max()}")
        # print(f"rgb.mean(): {x.mean()}")
        return torch.sigmoid(x)