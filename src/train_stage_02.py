import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg import VGGFeatureExtractor, normalize_vgg

class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [B, C, H, W]

        batch_size = x.size(0)
        h = x.size(2)
        w = x.size(3)

        if w == 1:
            count_h = x[:, :, 1:, :].numel()
            h_tv = torch.pow(
                x[:, :, 1:, :] - x[:, :, :h-1, :],
                2
            ).sum()

            return 2 * (h_tv / count_h) / batch_size

        if h == 1:
            count_w = x[:, :, :, 1:].numel()
            w_tv = torch.pow(
                x[:, :, :, 1:] - x[:, :, :, :w-1],
                2
            ).sum()

            return 2 * (w_tv / count_w) / batch_size

        count_h = x[:, :, 1:, :].numel()
        count_w = x[:, :, :, 1:].numel()

        h_tv = torch.pow(
            x[:, :, 1:, :] - x[:, :, :h-1, :],
            2
        ).sum()

        w_tv = torch.pow(
            x[:, :, :, 1:] - x[:, :, :, :w-1],
            2
        ).sum()

        return 2 * (
            h_tv / count_h +
            w_tv / count_w
        ) / batch_size
    
class StyleGridLoss(nn.Module):
    """
    Computes the two losses required for Style Grid Training (Stage-2):
        1. Content loss  — MSE(VGG(content_img).relu4_1, VGG(rendered_style_img).relu4_1)
        2. Style loss    — AdaIN-style mean/std matching between VGG(style_img) and
                            VGG(rendered_style_img), summed over multiple layers.
    """

    # Standard AdaIN layer set used for style statistics matching
    STYLE_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1")
    CONTENT_LAYER = "relu4_1"

    def __init__(self, device: torch.device, content_balancer_weight: float, style_balancer_weight: float, feature_tv_weight = 0.0, image_tv_weight = 0.0, gram_loss_weight: float = 1.0):
        super().__init__()
        self.device = device
        self.vgg_feature_extractor = VGGFeatureExtractor(self.device)
        self.content_balancer_weight = content_balancer_weight
        self.style_balancer_weight = style_balancer_weight  # lambda in the paper
        self.gram_loss_weight = gram_loss_weight

        self.feature_tv_weight = feature_tv_weight
        self.image_tv_weight = image_tv_weight
        self.tv_loss = TVLoss()

        self.eps = 1e-8

    def forward(self, content_img, style_img , feature_map, rendered_style_img):
        """
        content_img: Ground-truth content RGB image. Shape = [B, 3, H, W]

        style_img: Reference style RGB image. Shape = [B, 3, H, W]

        rendered_style_img:
            The FINAL decoded stylized RGB image — i.e. FsRF's
            `output["rgb_img"]` (decoder applied to F_cs), NOT the raw
            rendered feature tensor. Shape = [B, 3, H, W]
        
        feature_map:
            F_cs

        Returns:
            dict with "content_loss", "style_loss", "total_loss"
        """
        content_feats = self.vgg_feature_extractor(normalize_vgg(content_img))
        style_feats = self.vgg_feature_extractor(normalize_vgg(style_img))
        out_feats = self.vgg_feature_extractor(normalize_vgg(rendered_style_img))

        # Content loss — single layer, content vs. FINAL OUTPUT image
        content_loss = F.mse_loss(
            content_feats[self.CONTENT_LAYER],
            out_feats[self.CONTENT_LAYER],
        )

        adain_style_loss = 0.0
        gram_style_loss = 0.0
        for layer in self.STYLE_LAYERS:
            # Style Statistics
            s_mean, s_std = self.get_mean_std(style_feats[layer])
            o_mean, o_std = self.get_mean_std(out_feats[layer])

            # Adain style loss
            adain_layer_loss = F.mse_loss(s_mean, o_mean) + F.mse_loss(s_std, o_std)
            adain_style_loss = adain_style_loss + adain_layer_loss

            # Gram matrix loss
            s_gram = self.get_gram_matrix(style_feats[layer])
            o_gram = self.get_gram_matrix(out_feats[layer])
            gram_layer_loss = F.mse_loss(s_gram, o_gram)
            gram_style_loss = gram_style_loss + gram_layer_loss

        style_loss = adain_style_loss + self.gram_loss_weight * gram_style_loss

        feature_tv_loss = self.tv_loss(feature_map)
        feature_tv_loss = self.feature_tv_weight * feature_tv_loss

        image_tv_loss = self.tv_loss(rendered_style_img)
        image_tv_loss = self.image_tv_weight * image_tv_loss

        total_loss = self.content_balancer_weight * content_loss + self.style_balancer_weight * style_loss + feature_tv_loss + image_tv_loss
        
        return {
            "content_loss": content_loss,
            "adain_style_loss": adain_style_loss,
            "gram_style_loss": gram_style_loss,
            "style_loss": style_loss,
            "feature_tv_loss": feature_tv_loss,
            "image_tv_loss": image_tv_loss,
            "total_loss": total_loss,
        }

    def get_mean_std(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        std = std + self.eps
        return mean, std

    def get_gram_matrix(self, x):
        """
        Compute normalized Gram matrix.
        """
        B, C, H, W = x.shape
        features = x.reshape(B,C,H * W)
        gram = torch.bmm(features,features.transpose(1, 2))
        # Normalize by number of feature elements
        gram = gram / (C * H * W)
        return gram


import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from stage2 import FsRF

from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    SequentialLR,
)


class StyleGridTrainer:
    """
    Trainer for FsRF Stage-2 (SICT + DST) training.

    Trainable:
        - SICT
        - DST
        - Stage-1 initialized CNN decoder

    Frozen:
        - Stage-1 feature pipeline / learned 3D representation

    Responsibilities
    ----------------
    - Load frozen Stage-1 checkpoint
    - Freeze Stage-1 feature pipeline
    - Initialize Stage-2 decoder from Stage-1 decoder
    - Train SICT + DST + decoder
    - Build content/style/paired datasets + loader
    - Forward pass:
          frozen feature pipeline
              -> SICT
              -> volume rendering
              -> DST
              -> decoder
    - Stage-2 loss computation
    - Backpropagation
    - Optimizer step
    - Warmup + cosine LR scheduling
    - Checkpointing
    - Resume training
    - Style-image render plotting
    - Optional Weights & Biases logging
    - Gradient diagnostics
    """

    def __init__(
        self,
        stage1_model,
        stage1_checkpoint_path,
        loss_fn,
        config,
        device,
        content_dataset_cls,
        style_dataset_cls,
        paired_dataset_cls,
        content_root_dir,
        style_dir,
        use_wandb=False,
        resume_from=None,
    ):
        self.start_epoch = 1
        self.best_loss = float("inf")

        self.config = config
        self.device = device
        self.loss_fn = loss_fn
        self.use_wandb = use_wandb

        # ============================================================
        # AMP
        # ============================================================

        self.use_amp = config.get("use_amp", True)

        self.scaler = torch.amp.GradScaler(
            enabled=self.use_amp
        )

        # ============================================================
        # 1. LOAD STAGE-1 CHECKPOINT
        # ============================================================

        self.stage1_model = stage1_model.to(device)

        checkpoint = torch.load(
            stage1_checkpoint_path,
            map_location=device,
        )

        self.stage1_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.stage1_model.eval()

        # ------------------------------------------------------------
        # Freeze Stage-1 feature representation
        # ------------------------------------------------------------

        for param in self.stage1_model.feature_pipeline.parameters():
            param.requires_grad_(False)

        # ------------------------------------------------------------
        # Stage-1 decoder is only used as initialization.
        # It will NOT be trained directly.
        # ------------------------------------------------------------

        for param in self.stage1_model.decoder.parameters():
            param.requires_grad_(False)

        # ============================================================
        # 2. BUILD STAGE-2 MODEL
        # ============================================================

        self.model = FsRF(
            config=config,
            device=device,
        ).to(device)

        self.model.dst_pipeline.dst.match_decoder_distribution = False

        # ------------------------------------------------------------
        # Initialize Stage-2 decoder from Stage-1 decoder
        # ------------------------------------------------------------

        self.model.decoder.load_state_dict(
            self.stage1_model.decoder.state_dict()
        )

        # ------------------------------------------------------------
        # Stage-2 decoder IS trainable
        # ------------------------------------------------------------

        for param in self.model.decoder.parameters():
            param.requires_grad_(True)

        # SICT trainable
        for param in self.model.sict_pipeline.parameters():
            param.requires_grad_(True)

        # DST trainable
        for param in self.model.dst_pipeline.parameters():
            param.requires_grad_(True)

        # ============================================================
        # 3. DATASETS / DATALOADER
        # ============================================================

        content_dataset = content_dataset_cls(
            root_dir=content_root_dir,
            split="train",
            image_size = config["img_H"],
        )

        style_dataset = style_dataset_cls(
            style_dir=style_dir,
            image_size=config["img_H"],
        )

        self.train_dataset = paired_dataset_cls(
            content_dataset=content_dataset,
            style_dataset=style_dataset,
            mode=config.get(
                "style_pairing_mode",
                "random",
            ),
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
        )

        # ============================================================
        # 4. TRAINABLE PARAMETERS
        # ============================================================

        self.sict_params = [
            p
            for p in self.model.sict_pipeline.parameters()
            if p.requires_grad
        ]

        self.dst_params = [
            p
            for p in self.model.dst_pipeline.parameters()
            if p.requires_grad
        ]

        self.decoder_params = [
            p
            for p in self.model.decoder.parameters()
            if p.requires_grad
        ]

        assert len(self.sict_params) > 0, (
            "SICT has no trainable parameters!"
        )

        assert len(self.dst_params) > 0, (
            "DST has no trainable parameters!"
        )

        assert len(self.decoder_params) > 0, (
            "Decoder has no trainable parameters!"
        )

        # ============================================================
        # 5. OPTIMIZER
        # ============================================================

        self.stylizer_lr = config.get(
            "stylizer_lr",
            1e-4,
        )

        self.decoder_lr = config.get(
            "decoder_lr",
            1e-5,
        )

        self.optimizer = Adam(
            [
                {
                    "params": self.sict_params,
                    "lr": self.stylizer_lr,
                },
                {
                    "params": self.dst_params,
                    "lr": self.stylizer_lr,
                },
                {
                    "params": self.decoder_params,
                    "lr": self.decoder_lr,
                },
            ],
            betas=(0.9, 0.99),
            weight_decay=config.get(
                "weight_decay",
                0.0,
            ),
        )

        # ============================================================
        # 6. LR SCHEDULER
        #
        # Warmup:
        #     warmup_epochs
        #
        # After warmup:
        #     CosineAnnealingLR
        # ============================================================

        self.warmup_epochs = config.get(
            "warmup_epochs",
            0,
        )

        total_epochs = config["epochs"]

        cosine_epochs = max(
            1,
            total_epochs - self.warmup_epochs,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                ConstantLR(
                    self.optimizer,
                    factor=1.0,
                    total_iters=self.warmup_epochs,
                ),
                CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_epochs,
                    eta_min=config.get(
                        "lr_final",
                        5e-6,
                    ),
                ),
            ],
            milestones=[
                self.warmup_epochs
            ],
        )

        # ============================================================
        # 7. CHECKPOINT DIRECTORY
        # ============================================================

        self.save_dir = Path(
            config.get(
                "save_dir",
                "checkpoints/stage2",
            )
        )

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ============================================================
        # 8. RESUME
        # ============================================================

        if resume_from is not None:
            self.load_checkpoint(
                resume_from
            )

        # ============================================================
        # 9. WEIGHTS & BIASES
        # ============================================================

        if self.use_wandb:

            import wandb

            wandb.init(
                project=config.get(
                    "wandb_project",
                    "FsRF",
                ),
                name=config.get(
                    "run_name"
                ),
                config=config,
            )

            self.wandb = wandb

    # ================================================================
    # GRADIENT NORM
    # ================================================================

    def get_module_loss_grad_norm(
        self,
        loss,
        module,
    ):
        """
        Compute gradient norm of a module with respect to loss.

        Used for diagnostics only.
        Does not modify gradients.
        """

        params = [
            p
            for p in module.parameters()
            if p.requires_grad
        ]

        if len(params) == 0:
            return 0.0

        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )

        squared_norm = 0.0

        for grad in grads:

            if grad is not None:

                squared_norm += (
                    grad.detach()
                    .norm(2)
                    .item()
                    ** 2
                )

        return squared_norm ** 0.5

    # ================================================================
    # PARAMETER NORM
    # ================================================================

    @staticmethod
    def get_module_param_norm(module):
        """
        Compute L2 parameter norm.
        """

        squared_norm = 0.0

        for param in module.parameters():

            if param.requires_grad:

                squared_norm += (
                    param.detach()
                    .norm(2)
                    .item()
                    ** 2
                )

        return squared_norm ** 0.5

    # ================================================================
    # TRAIN ONE EPOCH
    # ================================================================

    def train_one_epoch(
        self,
        epoch,
    ):
        """
        Train one complete epoch.
        """

        # ------------------------------------------------------------
        # Explicitly restore training mode
        # ------------------------------------------------------------

        self.model.sict_pipeline.train()
        self.model.dst_pipeline.train()
        self.model.decoder.train()

        # ------------------------------------------------------------
        # Epoch metrics
        # ------------------------------------------------------------
        epoch_loss = 0.0
        epoch_content_loss = 0.0

        epoch_style_loss = 0.0
        epoch_adain_style_loss = 0.0
        epoch_gram_style_loss = 0.0

        epoch_feature_tv_loss = 0.0
        epoch_image_tv_loss = 0.0

        epoch_sict_grad_norm = 0.0
        epoch_dst_grad_norm = 0.0
        epoch_decoder_grad_norm = 0.0

        epoch_sict_param_norm = 0.0
        epoch_dst_param_norm = 0.0
        epoch_decoder_param_norm = 0.0

        # ============================================================
        # PROGRESS BAR
        # ============================================================

        progress = tqdm(
            self.train_loader,
            desc=(
                f"Epoch "
                f"[{epoch}/{self.config['epochs']}]"
            ),
            dynamic_ncols=True,
            leave=False,
            colour="magenta",
            bar_format=(
                "{l_bar}"
                "{bar:20}"
                "{r_bar}"
            ),
        )

        # ============================================================
        # BATCH LOOP
        # ============================================================

        for batch_idx, batch in enumerate(
            progress
        ):

            # --------------------------------------------------------
            # Move tensors to device
            # --------------------------------------------------------

            batch = {
                k: (
                    v.to(self.device)
                    if torch.is_tensor(v)
                    else v
                )
                for k, v in batch.items()
            }

            # --------------------------------------------------------
            # Zero gradients
            # --------------------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # ========================================================
            # STAGE-1 FEATURE EXTRACTION
            # ========================================================

            with torch.no_grad():

                features = (
                    self.stage1_model
                    .feature_pipeline
                    .infer(batch)
                )

            # ========================================================
            # STAGE-2 FORWARD
            # ========================================================

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):

                output = self.model(
                    features=features,
                    batch=batch,
                )

                # ----------------------------------------------------
                # F_cs -> feature map
                # ----------------------------------------------------
                B, R, C = (
                    output["F_cs"].shape
                )

                feature_H = self.config["grid_size"]
                feature_W = self.config["grid_size"]

                assert feature_H * feature_W == R, (
                    f"Feature ray count R={R} does not match "
                    f"grid size {feature_H}×{feature_W}"
                )

                feature_map = (
                    output["F_cs"]
                    .transpose(1, 2)
                    .reshape(B, C, feature_H, feature_W)
                )

                # ----------------------------------------------------
                # Loss
                # ---------------------------------------------------

                losses = self.loss_fn(
                    content_img=batch[
                        "target_rgb"
                    ],
                    style_img=batch[
                        "style_img"
                    ],
                    rendered_style_img=output[
                        "rgb_img"
                    ],
                    feature_map=feature_map,
                )

                content_loss = (
                    losses["content_loss"]
                )

                adain_style_loss = losses["adain_style_loss"]
                gram_style_loss = losses["gram_style_loss"]

                style_loss = (
                    losses["style_loss"]
                )

                feature_tv_loss = (
                    losses[
                        "feature_tv_loss"
                    ]
                )

                image_tv_loss = (
                    losses[
                        "image_tv_loss"
                    ]
                )

                total_loss = (
                    losses["total_loss"]
                )

            # ========================================================
            # BACKWARD
            # ========================================================

            self.scaler.scale(
                total_loss
            ).backward()

            # --------------------------------------------------------
            # IMPORTANT:
            #
            # Unscale before reading gradients.
            # Otherwise gradient diagnostics contain
            # AMP scaling.
            # --------------------------------------------------------

            if self.use_amp:

                self.scaler.unscale_(
                    self.optimizer
                )

            # ========================================================
            # GRADIENT DIAGNOSTICS
            # ========================================================

            sict_grad_norm = (
                self.get_module_grad_norm(
                    self.model.sict_pipeline
                )
            )

            dst_grad_norm = (
                self.get_module_grad_norm(
                    self.model.dst_pipeline
                )
            )

            decoder_grad_norm = (
                self.get_module_grad_norm(
                    self.model.decoder
                )
            )

            # --------------------------------------------------------
            # Parameter norms
            # --------------------------------------------------------

            sict_param_norm = (
                self.get_module_param_norm(
                    self.model.sict_pipeline
                )
            )

            dst_param_norm = (
                self.get_module_param_norm(
                    self.model.dst_pipeline
                )
            )

            decoder_param_norm = (
                self.get_module_param_norm(
                    self.model.decoder
                )
            )

            # ========================================================
            # OPTIMIZER STEP
            # ========================================================

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            # ========================================================
            # ACCUMULATE METRICS
            # ========================================================

            epoch_loss += total_loss.item()
            epoch_content_loss += content_loss.item()
            epoch_style_loss += style_loss.item()
            epoch_adain_style_loss += adain_style_loss.item()
            epoch_gram_style_loss += gram_style_loss.item()
            epoch_feature_tv_loss += feature_tv_loss.item()
            epoch_image_tv_loss += image_tv_loss.item()


            epoch_sict_grad_norm += (
                sict_grad_norm
            )

            epoch_dst_grad_norm += (
                dst_grad_norm
            )

            epoch_decoder_grad_norm += (
                decoder_grad_norm
            )

            epoch_sict_param_norm += (
                sict_param_norm
            )

            epoch_dst_param_norm += (
                dst_param_norm
            )

            epoch_decoder_param_norm += (
                decoder_param_norm
            )

            # ========================================================
            # PROGRESS BAR
            # ========================================================

            progress.set_postfix(
                loss=f"{total_loss.item():.6f}",
                content=f"{content_loss.item():.6f}",
                style=f"{style_loss.item():.6f}",
                dec_grad=f"{decoder_grad_norm:.2e}",
            )

            # ========================================================
            # W&B
            # ========================================================

            if self.use_wandb:

                self.wandb.log(
                    {
                        "train/lr_sict": (
                            self.optimizer
                            .param_groups[0]
                            ["lr"]
                        ),

                        "train/lr_dst": (
                            self.optimizer
                            .param_groups[1]
                            ["lr"]
                        ),

                        "train/lr_decoder": (
                            self.optimizer
                            .param_groups[2]
                            ["lr"]
                        ),

                        "train/loss": (
                            total_loss.item()
                        ),

                        "train/content_loss": (
                            content_loss.item()
                        ),

                        "train/style_loss": (
                            style_loss.item()
                        ),

                        "train/adain_style_loss": (
                            adain_style_loss.item()
                        ),

                        "train/gram_style_loss": (
                            gram_style_loss.item()
                        ),

                        "train/feature_tv_loss": (
                            feature_tv_loss.item()
                        ),

                        "train/image_tv_loss": (
                            image_tv_loss.item()
                        ),

                        "train/sict_grad_norm": (
                            sict_grad_norm
                        ),

                        "train/dst_grad_norm": (
                            dst_grad_norm
                        ),

                        "train/decoder_grad_norm": (
                            decoder_grad_norm
                        ),

                        "train/sict_param_norm": (
                            sict_param_norm
                        ),

                        "train/dst_param_norm": (
                            dst_param_norm
                        ),

                        "train/decoder_param_norm": (
                            decoder_param_norm
                        ),
                    }
                )

        # ============================================================
        # AVERAGE METRICS
        # ============================================================

        num_batches = len(
            self.train_loader
        )

        return {

            "loss": (
                epoch_loss
                / num_batches
            ),

            "content_loss": (
                epoch_content_loss
                / num_batches
            ),

            "style_loss": (
                epoch_style_loss
                / num_batches
            ),

            "adain_style_loss": (
                epoch_adain_style_loss
                / num_batches
            ),

            "gram_style_loss": (
                epoch_gram_style_loss
                / num_batches
            ),

            "feature_tv_loss": (
                epoch_feature_tv_loss
                / num_batches
            ),

            "image_tv_loss": (
                epoch_image_tv_loss
                / num_batches
            ),

            "sict_grad_norm": (
                epoch_sict_grad_norm
                / num_batches
            ),

            "dst_grad_norm": (
                epoch_dst_grad_norm
                / num_batches
            ),

            "decoder_grad_norm": (
                epoch_decoder_grad_norm
                / num_batches
            ),

            "sict_param_norm": (
                epoch_sict_param_norm
                / num_batches
            ),

            "dst_param_norm": (
                epoch_dst_param_norm
                / num_batches
            ),

            "decoder_param_norm": (
                epoch_decoder_param_norm
                / num_batches
            ),
        }

    # ================================================================
    # GRADIENT NORM FROM .grad
    # ================================================================

    @staticmethod
    def get_module_grad_norm(
        module
    ):
        """
        Compute gradient norm from gradients
        already produced by backward().
        """

        squared_norm = 0.0

        for param in module.parameters():

            if (
                param.requires_grad
                and param.grad is not None
            ):

                squared_norm += (
                    param.grad.detach()
                    .norm(2)
                    .item()
                    ** 2
                )

        return squared_norm ** 0.5

    # ================================================================
    # CHECKPOINT
    # ================================================================

    def save_checkpoint(
        self,
        epoch,
        best_loss,
    ):
        """
        Save all trainable Stage-2 state.

        Important:
        The decoder is included because it is now
        trainable during Stage-2.
        """

        torch.save(
            {
                "epoch": epoch,

                "best_loss": best_loss,

                # ----------------------------------------------------
                # Trainable modules
                # ----------------------------------------------------

                "sict_pipeline_state_dict": (
                    self.model
                    .sict_pipeline
                    .state_dict()
                ),

                "dst_pipeline_state_dict": (
                    self.model
                    .dst_pipeline
                    .state_dict()
                ),

                "decoder_state_dict": (
                    self.model
                    .decoder
                    .state_dict()
                ),

                # ----------------------------------------------------
                # Training state
                # ----------------------------------------------------

                "optimizer_state_dict": (
                    self.optimizer.state_dict()
                ),

                "scheduler_state_dict": (
                    self.scheduler.state_dict()
                ),

                "scaler_state_dict": (
                    self.scaler.state_dict()
                ),

                # ----------------------------------------------------
                # Configuration
                # ----------------------------------------------------

                "config": self.config,
            },

            self.save_dir
            / f"epoch_{epoch}.pth",
        )

    # ================================================================
    # LOAD CHECKPOINT
    # ================================================================

    def load_checkpoint(
        self,
        checkpoint_path,
    ):
        """
        Resume Stage-2 training.

        Supports checkpoints generated by
        this trainer.
        """

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        # ------------------------------------------------------------
        # SICT
        # ------------------------------------------------------------

        self.model.sict_pipeline.load_state_dict(
            checkpoint[
                "sict_pipeline_state_dict"
            ]
        )

        # ------------------------------------------------------------
        # DST
        # ------------------------------------------------------------

        self.model.dst_pipeline.load_state_dict(
            checkpoint[
                "dst_pipeline_state_dict"
            ]
        )

        # ------------------------------------------------------------
        # Decoder
        #
        # Required because decoder is trainable.
        # ------------------------------------------------------------

        if "decoder_state_dict" in checkpoint:

            self.model.decoder.load_state_dict(
                checkpoint[
                    "decoder_state_dict"
                ]
            )

        # ------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        # ------------------------------------------------------------
        # Scheduler
        #
        # IMPORTANT:
        # Do not recreate scheduler after loading.
        # Its state contains the current warmup/cosine position.
        # ------------------------------------------------------------

        if (
            "scheduler_state_dict"
            in checkpoint
        ):

            self.scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        # ------------------------------------------------------------
        # AMP scaler
        # ------------------------------------------------------------

        if (
            "scaler_state_dict"
            in checkpoint
        ):

            self.scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        # ------------------------------------------------------------
        # Epoch / best loss
        # ------------------------------------------------------------

        self.start_epoch = (
            checkpoint["epoch"] + 1
        )

        self.best_loss = checkpoint.get(
            "best_loss",
            float("inf"),
        )

        # ------------------------------------------------------------
        # Ensure correct training modes
        # ------------------------------------------------------------

        self.model.sict_pipeline.train()
        self.model.dst_pipeline.train()
        self.model.decoder.train()

        print("-" * 70)
        print(
            f"Resumed Stage-2 training from: "
            f"{checkpoint_path}"
        )

        print(
            f"Starting Epoch : "
            f"{self.start_epoch}"
        )

        print(
            f"Best Loss      : "
            f"{self.best_loss:.6f}"
        )

        print("-" * 70)

    # ================================================================
    # FIT
    # ================================================================

    def fit(self):

        print("-" * 80)
        print(
            "FsRF Stage-2 "
            "SICT/DST + Decoder Training"
        )
        print("-" * 80)

        print(
            f"Device         : "
            f"{self.device}"
        )

        print(
            f"Run Name       : "
            f"{self.config.get('run_name', 'N/A')}"
        )

        print(
            f"Checkpoint Dir : "
            f"{self.config.get('save_dir', 'checkpoints/stage2')}"
        )

        # ============================================================
        # FEATURE DIMS
        # ============================================================

        print("\n[Feature Dims]")

        print(
            f"  Content Feature Dim : "
            f"{self.config['content_feature_dim']}"
        )

        print(
            f"  Embed Dim (SICT)    : "
            f"{self.config['embed_dim']}"
        )

        print(
            f"  T Size (DST)        : "
            f"{self.config['T_size']}"
        )

        # ============================================================
        # OPTIMIZATION
        # ============================================================

        print("\n[Optimization]")

        print(
            f"  Epochs              : "
            f"{self.config['epochs']}"
        )

        print(
            f"  SICT/DST LR         : "
            f"{self.stylizer_lr:.1e}"
        )

        print(
            f"  Decoder LR          : "
            f"{self.decoder_lr:.1e}"
        )

        print(
            f"  Warmup Epochs       : "
            f"{self.warmup_epochs}"
        )

        print(
            f"  Final LR            : "
            f"{self.config.get('lr_final', 5e-6):.1e}"
        )

        print(
            f"  AMP                 : "
            f"{self.use_amp}"
        )

        print("=" * 80)

        # ============================================================
        # TRAINING
        # ============================================================

        start_time = time.time()

        best_loss = self.best_loss

        for epoch in range(
            self.start_epoch,
            self.config["epochs"] + 1,
        ):

            metrics = (
                self.train_one_epoch(
                    epoch
                )
            )

            train_loss = metrics[
                "loss"
            ]

            # --------------------------------------------------------
            # LR before scheduler step
            # --------------------------------------------------------

            sict_lr = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            dst_lr = (
                self.optimizer
                .param_groups[1]["lr"]
            )

            decoder_lr = (
                self.optimizer
                .param_groups[2]["lr"]
            )

            # ========================================================
            # EPOCH LOG
            # ========================================================

            print(
                f"[{epoch:03d}/"
                f"{self.config['epochs']}] "
                f"Loss: "
                f"{train_loss:.6f} | "
                f"Content: "
                f"{metrics['content_loss']:.6f} | "
                f"Style: "
                f"{metrics['style_loss']:.6f} | "
                f"AdaIN: "
                f"{metrics['adain_style_loss']:.6f} | "
                f"Gram: "
                f"{metrics['gram_style_loss']:.6f} | "
                f"Feature TV: "
                f"{metrics['feature_tv_loss']:.6f} | "
                f"Image TV: "
                f"{metrics['image_tv_loss']:.6f}"
            )

            print(
                f"    Grad Norms -> "
                f"SICT: "
                f"{metrics['sict_grad_norm']:.3e} | "
                f"DST: "
                f"{metrics['dst_grad_norm']:.3e} | "
                f"Decoder: "
                f"{metrics['decoder_grad_norm']:.3e}"
            )

            print(
                f"    Param Norms -> "
                f"SICT: "
                f"{metrics['sict_param_norm']:.3e} | "
                f"DST: "
                f"{metrics['dst_param_norm']:.3e} | "
                f"Decoder: "
                f"{metrics['decoder_param_norm']:.3e}"
            )

            print(
                f"    LR -> "
                f"SICT: "
                f"{sict_lr:.2e} | "
                f"DST: "
                f"{dst_lr:.2e} | "
                f"Decoder: "
                f"{decoder_lr:.2e}"
            )

            # ========================================================
            # W&B EPOCH METRICS
            # ========================================================

            if self.use_wandb:

                self.wandb.log(
                    {
                        "epoch": epoch,

                        "epoch/loss": train_loss,

                        "epoch/content_loss": (
                            metrics[
                                "content_loss"
                            ]
                        ),

                        "epoch/style_loss": (
                            metrics[
                                "style_loss"
                            ]
                        ),

                        "epoch/adain_style_loss": (
                            metrics[
                                "adain_style_loss"
                            ]
                        ),

                        "epoch/gram_style_loss": (
                            metrics[
                                "gram_style_loss"
                            ]
                        ),

                        "epoch/feature_tv_loss": (
                            metrics[
                                "feature_tv_loss"
                            ]
                        ),

                        "epoch/image_tv_loss": (
                            metrics[
                                "image_tv_loss"
                            ]
                        ),

                        "epoch/sict_grad_norm": (
                            metrics[
                                "sict_grad_norm"
                            ]
                        ),

                        "epoch/dst_grad_norm": (
                            metrics[
                                "dst_grad_norm"
                            ]
                        ),

                        "epoch/decoder_grad_norm": (
                            metrics[
                                "decoder_grad_norm"
                            ]
                        ),

                        "epoch/sict_param_norm": (
                            metrics[
                                "sict_param_norm"
                            ]
                        ),

                        "epoch/dst_param_norm": (
                            metrics[
                                "dst_param_norm"
                            ]
                        ),

                        "epoch/decoder_param_norm": (
                            metrics[
                                "decoder_param_norm"
                            ]
                        ),

                        "epoch/lr_sict": sict_lr,
                        "epoch/lr_dst": dst_lr,
                        "epoch/lr_decoder": (
                            decoder_lr
                        ),
                    }
                )

            # ========================================================
            # SCHEDULER
            # ========================================================

            self.scheduler.step()

            # ========================================================
            # BEST CHECKPOINT
            # ========================================================

            if train_loss < best_loss:

                best_loss = train_loss

                self.save_checkpoint(
                    epoch,
                    best_loss,
                )

                print(
                    f"  -> New best model saved "
                    f"(loss={best_loss:.6f})"
                )

            # ========================================================
            # STYLE RENDER
            # ========================================================

            if (
                epoch
                % self.config[
                    "style_img_plot_interval"
                ]
                == 0
            ):

                self.plot_style_render(
                    epoch=epoch,
                    chunk_size=self.config.get(
                        "render_chunk_size",
                        512,
                    ),
                )

        # ============================================================
        # FINISHED
        # ============================================================

        elapsed = (
            time.time()
            - start_time
        ) / 60

        print("=" * 70)

        print(
            f"Stage-2 Training Finished "
            f"({elapsed:.2f} min)"
        )

        print(
            f"Best Loss : "
            f"{best_loss:.6f}"
        )

        print("=" * 70)

        if self.use_wandb:
            self.wandb.finish()

    # ================================================================
    # STYLE RENDER
    # ================================================================

    @torch.no_grad()
    def plot_style_render(
        self,
        epoch,
        chunk_size=512,
    ):

        # ------------------------------------------------------------
        # Save current training state
        # ------------------------------------------------------------

        self.model.sict_pipeline.eval()
        self.model.dst_pipeline.eval()
        self.model.decoder.eval()

        # ------------------------------------------------------------
        # Get one batch
        # ------------------------------------------------------------

        batch = next(
            iter(self.train_loader)
        )

        batch = {
            k: (
                v.to(self.device)
                if torch.is_tensor(v)
                else v
            )
            for k, v in batch.items()
        }

        # ------------------------------------------------------------
        # Frozen feature extraction
        # ------------------------------------------------------------

        features = (
            self.stage1_model
            .feature_pipeline
            .infer(
                batch,
                chunk_size=chunk_size,
            )
        )

        # ------------------------------------------------------------
        # Stage-2 forward
        # ------------------------------------------------------------
        output = self.model(
            features=features,
            batch=batch,
        )

        rgb_img = output[
            "rgb_img"
        ]

        # ------------------------------------------------------------
        # Images
        # ------------------------------------------------------------

        content_img = (
            batch["target_rgb"][0]
            .detach()
            .cpu()
        )

        style_img = (
            batch["style_img"][0]
            .detach()
            .cpu()
        )

        stylized_img = (
            rgb_img[0]
            .detach()
            .cpu()
        )

        # ============================================================
        # SAVE RENDER
        # ============================================================

        render_dir = (
            self.save_dir
            / "renders"
        )

        render_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            1,
            3,
            figsize=(15, 5),
        )

        ax[0].imshow(
            content_img.permute(
                1,
                2,
                0,
            )
        )

        ax[0].set_title(
            "Content"
        )

        ax[0].axis("off")

        ax[1].imshow(
            style_img.permute(
                1,
                2,
                0,
            )
        )

        ax[1].set_title(
            "Style"
        )

        ax[1].axis("off")

        ax[2].imshow(
            stylized_img.permute(
                1,
                2,
                0,
            )
        )

        ax[2].set_title(
            f"Stylized — Epoch {epoch}"
        )

        ax[2].axis("off")

        plt.tight_layout()

        save_path = (
            render_dir
            / f"epoch_{epoch:04d}.png"
        )

        plt.savefig(
            save_path,
            dpi=150,
        )

        # ------------------------------------------------------------
        # W&B
        # ------------------------------------------------------------

        if self.use_wandb:

            self.wandb.log(
                {
                    "style_render":
                        self.wandb.Image(
                            str(save_path)
                        ),
                    "epoch": epoch,
                }
            )

        plt.close(fig)

        # IMPORTANT:
        self.model.sict_pipeline.train()
        self.model.dst_pipeline.train()
        self.model.decoder.train()


if __name__ == "__main__":
    from stage1 import PreFeatureGridTrainingStagePipeline
    from stage_02_config import config
    from load_dataset import NeRFSyntheticDataset, StyleImageDataset, StylizedNeRFDataset

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    stage1_model = PreFeatureGridTrainingStagePipeline(
        config=config
    ).to(device)

    loss_fn = StyleGridLoss(
        device=device,
        content_balancer_weight=config.get(
            "content_balancer_weight",
            5.0
        ),
        style_balancer_weight=config.get(
            "style_balancer_weight",
            10.0
        ),
        feature_tv_weight=config[
            "feature_tv_weight"
        ],
        image_tv_weight=config[
            "image_tv_weight"
        ],
    )

    trainer = StyleGridTrainer(
        stage1_model = stage1_model,
        stage1_checkpoint_path = config[
            "stage1_ckpt_path"
        ],
        loss_fn=loss_fn,
        config=config,
        device=device,
        content_dataset_cls=NeRFSyntheticDataset,
        style_dataset_cls=StyleImageDataset,
        paired_dataset_cls=StylizedNeRFDataset,
        content_root_dir=(
            "./data/nerf_synthetic/lego"
        ),
        style_dir="./data/style",
        use_wandb=False
    )

    trainer.fit()