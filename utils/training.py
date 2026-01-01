"""
Training Utilities for Palm Vein Recognition (IMPROVED)
=======================================================
Training loops, schedulers, and utilities optimized for maximum accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
from datetime import datetime
from tqdm import tqdm
import math

from config.settings import SystemConfig, DEFAULT_CONFIG
from models.networks import PalmVeinTrainingModel, PalmVeinNet
from models.losses import CombinedLoss, TripletLoss, LabelSmoothingCrossEntropy
from data.dataset import create_dataloaders, PalmVeinDataset

logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
        warmup_start_lr: float = 1e-7
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """Early stopping with patience."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """Trainer class for palm vein recognition model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: SystemConfig,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Get number of classes from model
        if hasattr(model, 'arcface'):
            self.num_classes = model.arcface.weight.size(0)
        else:
            self.num_classes = 1000  # Default
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = self._create_criterion()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.use_amp = config.training.use_amp and device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode='min'
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': [],
            'loss_breakdown': []
        }
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if self.config.training.optimizer == 'adamw':
            return optim.AdamW(
                params,
                lr=self.config.training.initial_lr,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adam':
            return optim.Adam(
                params,
                lr=self.config.training.initial_lr,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'sgd':
            return optim.SGD(
                params,
                lr=self.config.training.initial_lr,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_criterion(self):
        """Create loss function based on config."""
        if self.config.training.loss_type == 'combined':
            return CombinedLoss(
                num_classes=self.num_classes,
                embedding_dim=self.config.model.embedding_dim,
                arcface_weight=1.0,
                triplet_weight=self.config.training.triplet_weight,
                center_weight=self.config.training.center_loss_weight,
                triplet_margin=self.config.training.triplet_margin,
                label_smoothing=self.config.training.label_smoothing
            )
        elif self.config.training.loss_type == 'arcface':
            return LabelSmoothingCrossEntropy(
                smoothing=self.config.training.label_smoothing
            )
        elif self.config.training.loss_type == 'triplet':
            return TripletLoss(
                margin=self.config.training.triplet_margin,
                mining_strategy='hard'
            )
        else:
            return nn.CrossEntropyLoss()
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.training.lr_scheduler == 'cosine_warmup':
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config.training.warmup_epochs,
                total_epochs=self.config.training.num_epochs,
                min_lr=self.config.training.min_lr
            )
        elif self.config.training.lr_scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.lr_step_size,
                gamma=self.config.training.lr_gamma
            )
        elif self.config.training.lr_scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.min_lr
            )
        elif self.config.training.lr_scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.config.training.min_lr
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        loss_breakdown = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch in pbar:
            rgb = batch['rgb'].to(self.device)
            ir = batch['ir'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    logits, embeddings = self.model(rgb, ir, labels)
                    
                    if isinstance(self.criterion, CombinedLoss):
                        loss, breakdown = self.criterion(logits, embeddings, labels)
                    else:
                        loss = self.criterion(logits, labels)
                        breakdown = {'total': loss.item()}
                
                self.scaler.scale(loss).backward()
                
                if self.config.training.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_val
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, embeddings = self.model(rgb, ir, labels)
                
                if isinstance(self.criterion, CombinedLoss):
                    loss, breakdown = self.criterion(logits, embeddings, labels)
                else:
                    loss = self.criterion(logits, labels)
                    breakdown = {'total': loss.item()}
                
                loss.backward()
                
                if self.config.training.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_val
                    )
                
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * rgb.size(0)
            _, predicted = logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += rgb.size(0)
            
            # Accumulate loss breakdown
            for key, value in breakdown.items():
                if key not in loss_breakdown:
                    loss_breakdown[key] = 0.0
                loss_breakdown[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{total_correct/total_samples:.4f}"
            })
        
        # Average metrics
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        for key in loss_breakdown:
            loss_breakdown[key] /= len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'acc': avg_acc,
            'breakdown': loss_breakdown
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_embeddings = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            rgb = batch['rgb'].to(self.device)
            ir = batch['ir'].to(self.device)
            labels = batch['label'].to(self.device)
            
            logits, embeddings = self.model(rgb, ir, labels)
            
            if isinstance(self.criterion, CombinedLoss):
                loss, _ = self.criterion(logits, embeddings, labels)
            else:
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * rgb.size(0)
            _, predicted = logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += rgb.size(0)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{total_correct/total_samples:.4f}"
            })
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        # Calculate EER
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        eer = self._calculate_eer(all_embeddings, all_labels)
        
        return {
            'loss': avg_loss,
            'acc': avg_acc,
            'eer': eer
        }
    
    def _calculate_eer(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Calculate Equal Error Rate."""
        embeddings = embeddings.numpy()
        labels = labels.numpy()
        
        # Compute cosine similarities
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(embeddings_norm, embeddings_norm.T)
        
        # Get genuine and impostor scores
        genuine_scores = []
        impostor_scores = []
        
        n = len(labels)
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    genuine_scores.append(similarities[i, j])
                else:
                    impostor_scores.append(similarities[i, j])
        
        if len(genuine_scores) == 0 or len(impostor_scores) == 0:
            return 0.0
        
        # Calculate FAR and FRR at different thresholds
        thresholds = np.linspace(0, 1, 100)
        
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        min_diff = float('inf')
        eer = 0.0
        
        for thresh in thresholds:
            far = np.mean(impostor_scores >= thresh)
            frr = np.mean(genuine_scores < thresh)
            
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                eer = (far + frr) / 2
        
        return eer
    
    def train(
        self,
        num_epochs: int,
        save_dir: Union[str, Path],
        save_best: bool = True
    ) -> Dict[str, List]:
        """Full training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        
        for epoch in range(num_epochs):
            # Update learning rate
            if isinstance(self.scheduler, WarmupCosineScheduler):
                self.scheduler.step(epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            elif self.scheduler is not None and not isinstance(self.scheduler, WarmupCosineScheduler):
                self.scheduler.step()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['acc']:.4f}, "
                f"EER: {val_metrics['eer']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['lr'].append(current_lr)
            self.history['loss_breakdown'].append(train_metrics.get('breakdown', {}))
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                
                if save_best:
                    self._save_checkpoint(
                        save_dir / 'best_model.pth',
                        epoch,
                        val_metrics
                    )
                    logger.info(f"Saved best model at epoch {epoch+1}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    save_dir / f'checkpoint_epoch_{epoch+1}.pth',
                    epoch,
                    val_metrics
                )
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save final model
        self._save_checkpoint(
            save_dir / 'final_model.pth',
            epoch,
            val_metrics
        )
        
        # Save training history
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Training complete. Best model at epoch {best_epoch+1}")
        
        return self.history
    
    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': metrics['loss'],
            'val_acc': metrics['acc'],
            'config': {
                'embedding_dim': self.config.model.embedding_dim,
                'backbone': self.config.model.backbone,
                'fusion_type': self.config.model.fusion_type,
                'num_classes': self.num_classes
            }
        }, path)


def train_model(
    data_dir: Union[str, Path],
    save_dir: Union[str, Path],
    config: Optional[SystemConfig] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None
) -> Tuple[nn.Module, Dict]:
    """Train a palm vein recognition model."""
    config = config or DEFAULT_CONFIG
    
    if num_epochs:
        config.training.num_epochs = num_epochs
    if batch_size:
        config.training.batch_size = batch_size
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, dataset = create_dataloaders(
        data_dir=data_dir,
        batch_size=config.training.batch_size,
        target_size=config.image.input_size,
        augmentation_strength=config.training.augmentation_strength,
        val_split=0.2,
        num_workers=0,  # Windows compatibility
        seed=config.seed
    )
    
    num_classes = dataset.get_num_classes()
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Save metadata
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    dataset.save_metadata(save_path / 'dataset_metadata.json')
    
    # Create model
    logger.info("Creating model...")
    model = PalmVeinTrainingModel(
        num_classes=num_classes,
        backbone=config.model.backbone,
        pretrained=config.model.pretrained,
        embedding_dim=config.model.embedding_dim,
        fusion_type=config.model.fusion_type,
        use_attention=config.model.use_attention,
        dropout=config.model.dropout_rate,
        arcface_s=config.training.arcface_s,
        arcface_m=config.training.arcface_m
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    history = trainer.train(
        num_epochs=config.training.num_epochs,
        save_dir=save_dir,
        save_best=True
    )
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {min(history['val_loss']):.4f}")
    logger.info(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    
    return model, history
