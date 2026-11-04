import torch
from config import config as stage_01_config
from stage_02_config import config as stage_02_config
from stage1 import PreFeatureGridTrainingStagePipeline
from stage2 import FsRF

def load_stylerf_for_inference(
    config,
    stage1_model,
    stage1_checkpoint_path,
    stage2_checkpoint_path,
    device,
):
    """
    Load complete StyleRF model for inference.

    Required:
        Stage-1 checkpoint:
            - feature_pipeline
            - decoder

        Stage-2 checkpoint:
            - SICT
            - DST
            - trained decoder
    """

    device = torch.device(device)

    # ============================================================
    # 1. LOAD STAGE-1 MODEL
    # ============================================================

    stage1_model = stage1_model.to(device)

    stage1_checkpoint = torch.load(
        stage1_checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    stage1_model.load_state_dict(
        stage1_checkpoint["model_state_dict"]
    )

    stage1_model.eval()

    # Freeze Stage-1
    for param in stage1_model.parameters():
        param.requires_grad_(False)

    # ============================================================
    # 2. CREATE STAGE-2 StyleRF
    # ============================================================

    model = FsRF(
        config=config,
        device=device,
    ).to(device)

    # ============================================================
    # 3. LOAD STAGE-2 CHECKPOINT
    # ============================================================

    stage2_checkpoint = torch.load(
        stage2_checkpoint_path,
        map_location=device,
        weights_only = True
    )

    # SICT
    model.sict_pipeline.load_state_dict(
        stage2_checkpoint[
            "sict_pipeline_state_dict"
        ]
    )

    # DST
    model.dst_pipeline.load_state_dict(
        stage2_checkpoint[
            "dst_pipeline_state_dict"
        ]
    )

    # Trained Stage-2 decoder
    model.decoder.load_state_dict(
        stage2_checkpoint[
            "decoder_state_dict"
        ]
    )

    # ============================================================
    # 4. INFERENCE MODE
    # ============================================================

    model.eval()

    # Explicitly make sure these are eval
    model.sict_pipeline.eval()
    model.dst_pipeline.eval()
    model.decoder.eval()

    print("=" * 70)
    print("FsRF loaded successfully")
    print("=" * 70)

    print(f"Device       : {device}")
    print(f"Stage-1      : {stage1_checkpoint_path}")
    print(f"Stage-2      : {stage2_checkpoint_path}")
    print("=" * 70)
    return stage1_model, model

def tensor_to_numpy(tensor):
    # Remove batch dim, permute (3, H, W) -> (H, W, 3), convert to NumPy array
    img_np = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(img_np, 0, 1)


if __name__ == "__main__":
    from torch.utils.data import DataLoader, Subset
    from load_dataset import NeRFSyntheticDataset, StyleImageDataset, StylizedNeRFDataset
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    stage1_model = PreFeatureGridTrainingStagePipeline(
        config = stage_01_config
    ).to(device)

    stage1_model, stylerf = load_stylerf_for_inference(
        config = stage_02_config,
        stage1_model=stage1_model,
        stage1_checkpoint_path=
            "./checkpoints/stage_01/model.pth",
        stage2_checkpoint_path=
            "./checkpoints/stage_02/model.pth",
        device=device,
    )

    content_dataset = NeRFSyntheticDataset(
        root_dir="./data/nerf_synthetic/lego",
        split="test",
        image_size=stage_01_config["img_H"],
    )

    style_dataset = StyleImageDataset(
        style_dir="./data/style",
        image_size=stage_01_config["img_H"],
    )

    dataset = StylizedNeRFDataset(
        content_dataset,
        style_dataset,
        mode="fixed",
        fixed_idx=3,
    )

    loader = DataLoader(
        Subset(dataset, [67]),
        batch_size=1,
        shuffle=False,
    )

    batch = next(iter(loader))

    batch = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

    with torch.no_grad():

        features = stage1_model.feature_pipeline.infer(batch)

        output = stylerf(
            features=features,
            batch=batch,
        )

    stylized_img = output["rgb_img"]

    # plot the image
    content_np = tensor_to_numpy(batch["image"])
    style_np = tensor_to_numpy(batch["style_img"])
    stylized_np = tensor_to_numpy(output["rgb_img"])

    # Plot all three side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(content_np)
    axes[0].set_title("Original Content Image")
    axes[0].axis("off")

    axes[1].imshow(style_np)
    axes[1].set_title("Style Image")
    axes[1].axis("off")

    axes[2].imshow(stylized_np)
    axes[2].set_title("Generated Stylized Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()