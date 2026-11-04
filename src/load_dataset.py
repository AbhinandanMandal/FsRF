import os
import random
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class NeRFSyntheticDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size = None):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        self.transform = transforms.ToTensor()
        json_path = os.path.join(root_dir, f"transforms_{split}.json")

        with open(json_path, "r") as f:
            metadata = json.load(f)

        self.camera_angle_x = metadata["camera_angle_x"]
        self.frames = metadata["frames"]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image_path = os.path.join(
            self.root_dir,
            frame["file_path"] + ".png"
        )
        image = Image.open(image_path).convert("RGBA")

        # Resize BEFORE converting to tensor
        if self.image_size is not None:
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR
            )
        
        image = self.transform(image)
        rgb = image[:3]
        alpha = image[3:4]

        # Composite onto a white background
        target_rgb = rgb * alpha + (1.0 - alpha)

        transform_matrix = torch.tensor(
            frame["transform_matrix"],
            dtype=torch.float32
        )

        sample = {
            "image": image,
            "target_rgb": target_rgb,
            "camera_angle_x": self.camera_angle_x,
            "camera_pose": transform_matrix,
        }
        return sample

class StyleImageDataset(Dataset):
    """
    Loads style images from a flat folder of images.
    Returns RGB tensors in [0, 1], shape (3, H, W).
    """
    VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    def __init__(self, style_dir, image_size=None):
        self.style_dir = style_dir
        self.image_size = image_size

        self.paths = sorted(
            os.path.join(style_dir, f)
            for f in os.listdir(style_dir)
            if f.lower().endswith(self.VALID_EXTS)
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No style images found in {style_dir}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")

        if self.image_size is not None:
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR,
            )

        return self.to_tensor(image)  # (3, H, W), values in [0,1]


class StylizedNeRFDataset(Dataset):
    """
    Pairs each NeRF content sample with a style image.

    mode="random": a new random style image per __getitem__ call (recommended
                    for training — the same content view sees many styles
                    over the course of training).
    mode="fixed":  every sample gets style_dataset[fixed_idx] (useful for
                    a controlled qualitative comparison / eval).
    """
    def __init__(self, content_dataset, style_dataset, mode="random", fixed_idx=0, seed=None):
        self.content_dataset = content_dataset
        self.style_dataset = style_dataset
        self.mode = mode
        self.fixed_idx = fixed_idx
        self.rng = random.Random(seed)

        assert mode in ("random", "fixed")

    def __len__(self):
        return len(self.content_dataset)

    def __getitem__(self, idx):
        sample = self.content_dataset[idx]

        if self.mode == "random":
            style_idx = self.rng.randrange(len(self.style_dataset))
        else:
            style_idx = self.fixed_idx

        sample["style_img"] = self.style_dataset[style_idx]
        return sample

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class StyleImageDataset(Dataset):
    """
    Loads style images from a flat folder of images.
    Returns RGB tensors in [0, 1], shape (3, H, W).
    """
    VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    def __init__(self, style_dir, image_size=None):
        self.style_dir = style_dir
        self.image_size = image_size

        self.paths = sorted(
            os.path.join(style_dir, f)
            for f in os.listdir(style_dir)
            if f.lower().endswith(self.VALID_EXTS)
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No style images found in {style_dir}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")

        if self.image_size is not None:
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR,
            )

        return self.to_tensor(image)  # (3, H, W), values in [0,1]


import random
from torch.utils.data import Dataset

class StylizedNeRFDataset(Dataset):
    """
    Pairs each NeRF content sample with a style image.

    mode="random": a new random style image per __getitem__ call (recommended
                    for training — the same content view sees many styles
                    over the course of training).
    mode="fixed":  every sample gets style_dataset[fixed_idx] (useful for
                    a controlled qualitative comparison / eval).
    """
    def __init__(self, content_dataset, style_dataset, mode="random", fixed_idx=0, seed=None):
        self.content_dataset = content_dataset
        self.style_dataset = style_dataset
        self.mode = mode
        self.fixed_idx = fixed_idx
        self.rng = random.Random(seed)

        assert mode in ("random", "fixed")

    def __len__(self):
        return len(self.content_dataset)

    def __getitem__(self, idx):
        sample = self.content_dataset[idx]

        if self.mode == "random":
            style_idx = self.rng.randrange(len(self.style_dataset))
        else:
            style_idx = self.fixed_idx

        sample["style_img"] = self.style_dataset[style_idx]
        return sample