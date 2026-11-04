config = {
    # NeRF / Feature Field
    "t_n": 2.0,
    "t_f": 6.0,
    "n_points": 64,
    "num_rays": 16384, # changed from 64*64 to 128*128
    "grid_size": 128, # changed from 64 to 128
    "position_L": 10,
    "direction_L": 4,
    "hidden_dim": 256,
    "method": "nerf",
    "batch_size": 1,
    "warmup_epochs": 50,
    "train_chunk_size": 512,
    "use_amp": False,

    # Feature Representation
    "content_feature_dim": 256,
    "embed_dim": 128, # changed from 32 to 128
    "T_size": 128,

    # Rendering
    "volume_renderer_eps": 1e-10,
    "importance_sampler_eps": 1e-5,
    "n_importance": 64,

    # Image
    "img_H": 512, # changed from 128 to 512
    "img_W": 512, # changed from 128 to 512

    # Stage-1 Training
    "epochs": 500,
    "lr": 7e-4,
    "lr_final": 5e-6,
    "weight_decay": 0.0,
    "render_every": 5,
    "render_chunk_size": 128,

    # Checkpoints
    "save_dir": "./checkpoints/stage_01",

    # Logging
    "wandb_project": "FsRF",
    "run_name": "stage1-feature-grid",
}
assert config["num_rays"] == config["grid_size"] ** 2, (
    "num_rays must equal grid_size²"
)