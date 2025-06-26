class Config:
    # Paths
    lr_dir = "data/LR"
    hr_dir = "data/HR"
    captions_path = "captions/filtered.csv"
    checkpoint_dir = "checkpoints"
    
    # Model
    text_dim = 512  # CLIP text embedding size
    text_proj_dim = 256
    img_channels = 3
    scale_factor = 4
    num_res_blocks = 16
    
    # Training
    batch_size = 16
    num_workers = 4
    lr = 2e-4
    betas = (0.9, 0.999)
    epochs = 100
    val_interval = 2
    
    # Loss weights
    λ_pixel = 1.0
    λ_perc = 0.1
    λ_adv = 0.01
    λ_clip = 0.2