from dataclasses import dataclass


@dataclass
class Config:
    # model-readable images
    image_height: int = 160
    image_width: int = 800

    # public carrier image
    carrier_height: int = 180
    carrier_width: int = 800

    # dataset sizes
    train_size: int = 10000
    val_size: int = 2000

    # sequence lengths
    min_values: int = 18
    max_values: int = 40

    # training: safe for laptop 16 GB RAM
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-3
    num_workers: int = 2
    prefetch_factor: int = 2

    use_compile: bool = False
    use_amp: bool = True
    use_channels_last: bool = True
    grad_clip: float = 2.0

    compute_exact_acc_every: int = 2
    compute_token_acc_every: int = 1
    save_last_every: int = 1
    early_stopping_patience: int = 8

    train_dir: str = "data/train"
    val_dir: str = "data/val"
    train_labels: str = "data/train_labels.jsonl"
    val_labels: str = "data/val_labels.jsonl"
    checkpoint_dir: str = "checkpoints"

    seed: int = 1337
