import time
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, SequentialLR

class FeatureGridTrainer:
    """
    Trainer for StyleRF Stage-1 Feature Grid Training.

    Responsibilities
    ----------------
    - Forward pass
    - Stage-1 loss computation
    - Backpropagation
    - Optimizer step
    - Learning-rate scheduling
    - Checkpointing
    - Optional Weights & Biases logging
    """

    def __init__(
        self,
        model,
        train_loader,
        loss_fn,
        config,
        device,
        use_wandb=False,
        resume_from=None,
    ):
        self.start_epoch = 1
        self.best_loss = float("inf")

        self.model = model.to(device)
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.use_wandb = use_wandb

        self.use_amp = config.get("use_amp", True)
        self.scaler = torch.amp.GradScaler(enabled = self.use_amp)

        decoder_lr_mult = config.get("decoder_lr_mult", 10.0)  # decoder gets 10x base lr

        # Optimizer
        # self.optimizer = Adam(
        #     self.model.parameters(),
        #     lr=config["lr"],
        #     weight_decay=config.get("weight_decay", 0.0),
        # )

        self.optimizer = Adam(
            [
                {
                    "params": self.model.feature_pipeline.parameters(),
                    "lr": config["lr"],
                },
                {
                    "params": self.model.decoder.parameters(),
                    "lr": config["lr"] * decoder_lr_mult,
                },
            ],
            weight_decay=config.get("weight_decay", 0.0),
        )

        warmup_epochs = config["warmup_epochs"]

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                ConstantLR(self.optimizer, factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(
                    self.optimizer,
                    T_max=config["epochs"] - warmup_epochs,
                    eta_min=config.get("lr_final", 5e-5),
                ),
            ],
            milestones=[warmup_epochs],
        )

        # Checkpoint directory
        self.save_dir = Path(
            config.get("save_dir", "checkpoints")
        )
        self.save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Resume
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[
                    ConstantLR(self.optimizer, factor=1.0, total_iters=warmup_epochs),
                    CosineAnnealingLR(
                        self.optimizer,
                        T_max=config["epochs"] - warmup_epochs,
                        eta_min=config.get("lr_final", 5e-5),
                    ),
                ],
                milestones=[warmup_epochs],
            )
        # Weights & Biases
        if self.use_wandb:
            import wandb

            wandb.init(
                project=config.get(
                    "wandb_project",
                    "StyleRF"
                ),
                name=config.get("run_name"),
                config=config,
            )

            self.wandb = wandb

    # Train One Epoch
    def train_one_epoch(self, epoch):
        self.model.train()

        epoch_loss = 0.0
        epoch_feature_loss = 0.0
        epoch_rgb_loss = 0.0
        epoch_perceptual_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch [{epoch}/{self.config['epochs']}]",
            dynamic_ncols=True,
            leave=False,
            colour="cyan",
            bar_format="{l_bar}{bar:20}{r_bar}",
        )

        for batch_idx, batch in enumerate(progress):

            # Move batch to device
            batch = {
                k: v.to(self.device)
                if torch.is_tensor(v)
                else v
                for k, v in batch.items()
            }

            self.optimizer.zero_grad(set_to_none = True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Forward
                output = self.model(batch)

                # Compute losses
                losses = self.loss_fn(
                    batch=batch,
                    output=output,
                )

                feature_loss = losses["feature_loss"]
                rgb_loss = losses["rgb_loss"]
                perceptual_loss = losses["perceptual_loss"]

                w_feat = 0.1
                w_rgb = 1.0
                w_perc = 0.01
        
                total_loss = (
                    w_feat * losses["feature_loss"]
                    + w_rgb * losses["rgb_loss"]
                    + w_perc * losses["perceptual_loss"]
                )

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_feature_loss += feature_loss.item()
            epoch_rgb_loss += rgb_loss.item()
            epoch_perceptual_loss += perceptual_loss.item()

            # Progress bar
            progress.set_postfix(
                loss=f"{total_loss.item():.6f}",
                feature=f"{feature_loss.item():.6f}",
                rgb=f"{rgb_loss.item():.6f}",
                perceptual=f"{perceptual_loss.item():.6f}",
            )

            # W&B logging
            if self.use_wandb:

                self.wandb.log({
                    "train/lr_feature_pipeline": self.optimizer.param_groups[0]["lr"],
                    "train/lr_decoder": self.optimizer.param_groups[1]["lr"],

                    "train/loss": total_loss.item(),
                    "train/feature_loss": feature_loss.item(),
                    "train/rgb_loss": rgb_loss.item(),
                    "train/perceptual_loss": perceptual_loss.item(),

                    "render/weights_mean":
                        output["weights"].mean().item(),

                    "render/weights_max":
                        output["weights"].max().item(),

                    "render/alpha_mean":
                        output["alpha"].mean().item(),

                    "render/alpha_max":
                        output["alpha"].max().item(),

                    "render/depth_mean":
                        output["depth"].mean().item(),
                })

        # Epoch averages
        num_batches = len(self.train_loader)

        return {
            "loss": epoch_loss / num_batches,
            "feature_loss":
                epoch_feature_loss / num_batches,
            "rgb_loss":
                epoch_rgb_loss / num_batches,
            "perceptual_loss":
                epoch_perceptual_loss / num_batches,
        }

    def get_module_loss_grad_norm(self, loss, module):
        params = [
            p for p in module.parameters()
            if p.requires_grad
        ]

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
                    grad.detach().norm(2).item() ** 2
                )

        return squared_norm ** 0.5

    # Checkpoint
    def save_checkpoint(self, epoch, best_loss):
        torch.save(
            {
                "epoch": epoch,
                "best_loss": best_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "config": self.config,
            },
            self.save_dir / f"epoch_{epoch}.pth",
        )

    # Load Checkpoint
    def load_checkpoint(self, checkpoint_path):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.start_epoch = (
            checkpoint["epoch"] + 1
        )

        self.best_loss = checkpoint.get(
            "best_loss",
            float("inf")
        )

        print("-" * 70)
        print(
            f"Resumed training from: "
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

    # Fit
    def fit(self):

        print("-" * 80)
        print("StyleRF Stage-1 Feature Grid Training")
        print("-" * 80)

        print(f"Device         : {self.device}")
        print(
            f"Run Name       : "
            f"{self.config.get('run_name', 'N/A')}"
        )
        print(
            f"Checkpoint Dir : "
            f"{self.config.get('save_dir', 'checkpoints')}"
        )

        print("\n[Feature Grid]")
        print(
            f"  Feature Dim  : "
            f"{self.config['content_feature_dim']}"
        )
        print(
            f"  Embedding Dim: "
            f"{self.config['embed_dim']}"
        )
        print(
            f"  Method       : "
            f"{self.config['method']}"
        )

        print("\n[Sampling]")
        print(
            f"  Near Plane   : "
            f"{self.config['t_n']}"
        )
        print(
            f"  Far Plane    : "
            f"{self.config['t_f']}"
        )
        print(
            f"  Samples/Ray  : "
            f"{self.config['n_points']}"
        )
        print(
            f"  Rays/Batch   : "
            f"{self.config['num_rays']}"
        )

        print("\n[Optimization]")
        print(
            f"  Epochs       : "
            f"{self.config['epochs']}"
        )
        print(
            f"  Learning Rate: "
            f"{self.config['lr']:.1e}"
        )

        print("=" * 80)

        start_time = time.time()

        best_loss = self.best_loss

        for epoch in range(
            self.start_epoch,
            self.config["epochs"] + 1,
        ):

            metrics = self.train_one_epoch(
                epoch
            )

            train_loss = metrics["loss"]

            current_lr = (
                self.scheduler
                .get_last_lr()[0]
            )

            print(
                f"[{epoch:03d}/"
                f"{self.config['epochs']}] "
                f"Loss: {train_loss:.6f} | "
                f"Feature: "
                f"{metrics['feature_loss']:.6f} | "
                f"RGB: "
                f"{metrics['rgb_loss']:.6f} | "
                f"Perceptual: "
                f"{metrics['perceptual_loss']:.6f} | "
                f"LR: {current_lr:.2e}"
            )

            # Scheduler
            self.scheduler.step()

            # Best checkpoint
            if train_loss < best_loss:
                best_loss = train_loss
                self.save_checkpoint(
                    epoch,
                    best_loss
                )
                print(
                    f"  → New best model saved "
                    f"(loss={best_loss:.6f})"
                )

            # Full-resolution validation render
            if epoch % self.config["render_every"] == 0:
                self.render_validation_image(
                    epoch = epoch,
                    chunk_size = self.config.get(
                        "render_chunk_size",
                        512
                    )
        )

        elapsed = (
            time.time() - start_time
        ) / 60

        print("=" * 70)
        print(
            f"Training Finished "
            f"({elapsed:.2f} min)"
        )
        print(
            f"Best Loss : "
            f"{best_loss:.6f}"
        )
        print("=" * 70)

        if self.use_wandb:
            self.wandb.finish()

    def get_loss_grad_stats(self, loss):
        """
        Compute gradient statistics produced by one loss component.

        Returns:
            total_grad_norm
            mean_grad_mag
            max_grad_mag
            num_params_with_grad
            num_params_without_grad
        """

        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            retain_graph=True,
            allow_unused=True,
        )

        grad_norms = []
        grad_magnitudes = []

        params_with_grad = 0
        params_without_grad = 0

        for grad in grads:

            if grad is None:
                params_without_grad += 1
                continue
            params_with_grad += 1
            grad_norms.append(
                grad.norm(2).item()
            )
            grad_magnitudes.append(
                grad.abs().mean().item()
            )

        if not grad_norms:
            return {
                "norm": 0.0,
                "mean_mag": 0.0,
                "max_mag": 0.0,
                "with_grad": 0,
                "without_grad": params_without_grad,
            }

        # Global L2 norm across parameter gradients
        total_norm = (
            sum(x ** 2 for x in grad_norms) ** 0.5
        )

        return {
            "norm": total_norm,
            "mean_mag": sum(grad_magnitudes) / len(grad_magnitudes),
            "max_mag": max(grad_magnitudes),
            "with_grad": params_with_grad,
            "without_grad": params_without_grad,
        }

    @torch.no_grad()
    def render_validation_image(self, epoch, chunk_size = 512):
        self.model.eval()

        # Take one batch
        batch = next(iter(self.train_loader))

        batch = {
            k: v.to(self.device)
            if torch.is_tensor(v)
            else v
            for k, v in batch.items()
        }

        # Full-resolution feature rendering
        rendered_features = self.model.feature_pipeline.infer(
            batch,
            chunk_size=chunk_size
        )

        # rendered_features:
        # (B, H*W, C)

        B, R, C = rendered_features.shape
        expected_R = self.config["grid_size"] ** 2
        assert R == expected_R, f"Expected {expected_R} rays (grid_size²), got {R}"

        # Decode features → RGB
        rendered_img = self.model.decoder(
            rendered_features
        )

        # CNNDecoder expects:
        # (B, R, C)
        # and internally converts to:
        # (B, C, H, W)

        # Ground truth
        target_img = batch["target_rgb"]

        # Save first image
        pred = rendered_img[0].detach().cpu()
        gt = target_img[0].detach().cpu()

        render_dir = (
            self.save_dir / "renders"
        )

        render_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(
            1,
            2,
            figsize=(10, 5)
        )

        ax[0].imshow(
            gt.permute(1, 2, 0)
        )
        ax[0].set_title("Ground Truth")
        ax[0].axis("off")

        ax[1].imshow(
            pred.permute(1, 2, 0)
        )
        ax[1].set_title(
            f"Prediction — Epoch {epoch}"
        )
        ax[1].axis("off")

        plt.tight_layout()

        plt.savefig(
            render_dir / f"epoch_{epoch:04d}.png",
            dpi=150
        )

        plt.close(fig)
        self.model.train()


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from load_dataset import NeRFSyntheticDataset
    from config import config
    from stage1 import PreFeatureGridTrainingStagePipeline, FeatureGridLoss


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = NeRFSyntheticDataset(
        root_dir="./data/nerf_synthetic/lego",
        split="train",
        image_size=config["img_H"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],          
        shuffle=True
    )

    batch = next(iter(train_loader))
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    model = PreFeatureGridTrainingStagePipeline(
        config=config
    )

    feature_grid_objective = FeatureGridLoss(
        device=device,
        H=config["img_H"],
        W=config["img_W"],
        grid_size=config["grid_size"]
    )

    trainer = FeatureGridTrainer(
        model = model,
        train_loader = train_loader,
        loss_fn = feature_grid_objective,
        config = config,
        device = device,
        use_wandb = False
    )
    trainer.fit()