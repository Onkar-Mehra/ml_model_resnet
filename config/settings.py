"""
Palm Vein Biometric System Configuration (IMPROVED)
====================================================
All hyperparameters, paths, and settings for the biometric system.
Optimized for maximum accuracy with 1453+ people dataset.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class ImageConfig:
    """Image processing configuration."""
    # Input image size (height, width)
    input_size: Tuple[int, int] = (224, 224)
    
    # ROI extraction settings
    roi_size: Tuple[int, int] = (200, 200)
    
    # Preprocessing
    clahe_clip_limit: float = 3.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    
    # Gabor filter parameters for vein extraction
    gabor_ksize: int = 31
    gabor_sigma: float = 4.0
    gabor_theta_count: int = 8  # Number of orientations
    gabor_lambd: float = 10.0
    gabor_gamma: float = 0.5
    
    # Morphological operations
    morph_kernel_size: int = 3
    
    # Image normalization
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # CHANGED: Using resnet50 for better feature extraction
    backbone: str = "resnet50"  # Options: custom, resnet50, efficientnet_b3, vgg19
    
    # CHANGED: Increased embedding dimension for more capacity
    embedding_dim: int = 1024
    
    # Attention mechanism
    use_attention: bool = True
    attention_heads: int = 8
    
    # Fusion strategy for RGB + IR
    fusion_type: str = "attention"  # Options: concat, attention, bilinear
    
    # CHANGED: Reduced dropout for larger dataset
    dropout_rate: float = 0.2
    
    # CHANGED: Using pretrained weights
    pretrained: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    # CHANGED: Increased batch size
    batch_size: int = 64
    
    # CHANGED: Learning rate with warmup
    initial_lr: float = 1e-4
    min_lr: float = 1e-7
    warmup_epochs: int = 5
    
    # CHANGED: Increased epochs
    num_epochs: int = 200
    
    # Early stopping
    early_stopping_patience: int = 25
    
    # CHANGED: Using cosine annealing with warm restarts
    lr_scheduler: str = "cosine_warmup"  # Options: step, cosine, plateau, cosine_warmup
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    
    # CHANGED: Combined loss for better embeddings
    loss_type: str = "combined"  # Options: triplet, arcface, contrastive, combined
    
    # ArcFace parameters - CHANGED: Optimized values
    arcface_s: float = 64.0  # Increased scale
    arcface_m: float = 0.5
    
    # Triplet loss parameters
    triplet_margin: float = 0.3
    triplet_weight: float = 0.3  # NEW: Weight for triplet loss in combined
    
    # Center loss parameters - NEW
    center_loss_weight: float = 0.01
    
    # Optimizer
    optimizer: str = "adamw"  # Options: adam, adamw, sgd
    weight_decay: float = 1e-4
    
    # Data augmentation strength
    augmentation_strength: str = "strong"  # Options: none, light, strong
    
    # Mixed precision training
    use_amp: bool = True
    
    # Gradient clipping
    gradient_clip_val: float = 1.0
    
    # NEW: Label smoothing
    label_smoothing: float = 0.1


@dataclass
class VerificationConfig:
    """Verification/matching configuration."""
    # Similarity threshold for verification
    similarity_threshold: float = 0.75
    
    # Distance metric
    distance_metric: str = "cosine"  # Options: cosine, euclidean, manhattan
    
    # Multi-scale matching
    use_multi_scale: bool = True
    scales: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    
    # Template matching
    num_templates_per_user: int = 1  # Can be increased with augmentation
    
    # Matching strategy
    matching_strategy: str = "weighted_fusion"  # Options: max, average, weighted_fusion
    
    # Weights for RGB vs IR features
    rgb_weight: float = 0.4
    ir_weight: float = 0.6  # IR typically more reliable for vein patterns


@dataclass
class DatabaseConfig:
    """Database configuration for storing embeddings."""
    # Storage type
    storage_type: str = "faiss"  # Options: faiss, sqlite, numpy
    
    # FAISS index type
    faiss_index_type: str = "IVFFlat"  # Options: Flat, IVFFlat, IVFPQ
    
    # Number of clusters for IVF
    faiss_nlist: int = 100
    
    # Number of probes during search
    faiss_nprobe: int = 10
    
    # Database path
    db_path: str = "data/embeddings_db"


@dataclass
class PathConfig:
    """Path configuration."""
    # Base directory
    base_dir: Path = field(default_factory=lambda: Path("/home/claude/palm_vein_biometric"))
    
    # Data directories
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    
    # Model directories
    model_dir: Path = field(default_factory=lambda: Path("models/saved"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("models/checkpoints"))
    
    # Logs
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # Database
    database_dir: Path = field(default_factory=lambda: Path("data/database"))
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Path) and attr_name.endswith('_dir'):
                full_path = self.base_dir / attr if not attr.is_absolute() else attr
                full_path.mkdir(parents=True, exist_ok=True)


@dataclass
class SystemConfig:
    """Complete system configuration."""
    image: ImageConfig = field(default_factory=ImageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Device configuration
    device: str = "cuda"  # Options: cuda, cpu, mps
    num_workers: int = 0  # Set to 0 for Windows compatibility
    seed: int = 42
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SystemConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()
