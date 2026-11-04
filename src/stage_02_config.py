config = {
    # NeRF / Feature Field
    "t_n": 2.0,
    "t_f": 6.0,
    "n_points": 64,
    "num_rays": 128 * 128,
    "grid_size": 128,
    "position_L": 10,
    "direction_L": 4,
    "hidden_dim": 256,
    "method": "nerf",
    "batch_size": 1,
    "warmup_epochs": 40,
    "train_chunk_size": 512,

    # Feature Representation
    "content_feature_dim": 256,
    "embed_dim": 32,
    "T_size": 32,

    # Rendering
    "volume_renderer_eps": 1e-10,
    "importance_sampler_eps": 1e-5,
    "n_importance": 64,

    # Image
    "img_H": 512, # changed from 128 to 512
    "img_W": 512, # changed from 128 to 512

    # Stage-1 Training
    "epochs": 1000,
    "lr_final": 5e-7,
    "weight_decay": 0.0,
    "render_every": 5,
    "render_chunk_size": 128,

    "stylizer_lr": 1e-4,
    "decoder_lr": 1e-5,

    "style_img_plot_interval": 2,

    # Checkpoints
    "stage1_ckpt_path": "checkpoints/stage_01/model.pth",

    "save_dir": "checkpoints/stage_02",

    # Logging
    "wandb_project": "FsRF",
    "run_name": "fsrf-stage2-training",

    "content_balancer_weight": 5, # changed from 5 to 3
    "style_balancer_weight": 10, # changed from 19 to 100
    "gram_loss_weight": 1e5, # changed from 0 to 1e4

    "feature_tv_weight": 1e-6,
    "image_tv_weight" : 1e-5,
    "use_amp": False,
}

assert config["num_rays"] == config["grid_size"] ** 2, (
    "num_rays must equal grid_size²"
)