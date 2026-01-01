"""
SERVER TRAINING SCRIPT - With Detailed Logging
===============================================
Optimized for: 16 CPU, 32GB RAM, Ubuntu 24.04
Includes: Progress logging, checkpoints, email alerts (optional)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Set CPU threads BEFORE importing torch
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

sys.path.insert(0, str(Path(__file__).parent))

import torch
torch.set_num_threads(16)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm
import json
import random
import logging

# ============================================
# CONFIGURATION
# ============================================
DATA_DIR = "final_folder"
SAVE_DIR = "models"
LOG_DIR = "logs"
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_CPU_CORES = 16
DEVICE = "cpu"

# Logging interval
LOG_EVERY_N_BATCHES = 10
SAVE_CHECKPOINT_EVERY = 10  # epochs
# ============================================

# Create directories
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# Setup logging
log_filename = f"{LOG_DIR}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Also create a simple progress file for quick checking
progress_file = f"{LOG_DIR}/progress.txt"


def log_progress(epoch, epochs, train_loss, train_acc, val_loss, val_acc, epoch_time, best_acc):
    """Write progress to a simple file for quick checking."""
    with open(progress_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("PALM VEIN TRAINING PROGRESS\n")
        f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Current Epoch: {epoch}/{epochs}\n")
        f.write(f"Progress: {100*epoch/epochs:.1f}%\n")
        f.write(f"{'█' * int(50*epoch/epochs)}{'░' * (50-int(50*epoch/epochs))}\n\n")
        f.write(f"Latest Metrics:\n")
        f.write(f"  Train Loss: {train_loss:.4f}\n")
        f.write(f"  Train Acc:  {train_acc:.2f}%\n")
        f.write(f"  Val Loss:   {val_loss:.4f}\n")
        f.write(f"  Val Acc:    {val_acc:.2f}%\n\n")
        f.write(f"Best Val Accuracy: {best_acc:.2f}%\n")
        f.write(f"Epoch Time: {epoch_time:.1f} minutes\n")
        f.write(f"Estimated Remaining: {(epochs-epoch)*epoch_time/60:.1f} hours\n")
        f.write("="*60 + "\n")


from data.dataset import PalmVeinDataset, DataAugmentation
from models.networks import PalmVeinNet, PalmVeinTrainingModel
from models.losses import CombinedLoss


def create_dataloaders(data_dir, batch_size, val_split=0.2):
    """Create train and val dataloaders."""
    logger.info("Creating dataloaders...")
    
    train_aug = DataAugmentation(strength="strong")
    
    train_dataset = PalmVeinDataset(
        data_dir=data_dir,
        target_size=(224, 224),
        augmentation=train_aug,
        mode="train"
    )
    
    val_dataset = PalmVeinDataset(
        data_dir=data_dir,
        target_size=(224, 224),
        augmentation=None,
        mode="val"
    )
    
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    split_idx = int(len(indices) * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_labels = [train_dataset.labels[i] for i in train_indices]
    class_counts = np.bincount(train_labels, minlength=train_dataset.get_num_classes())
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
    
    logger.info(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")
    
    return train_loader, val_loader, train_dataset


def train():
    start_time = datetime.now()
    
    logger.info("="*60)
    logger.info("PALM VEIN BIOMETRIC TRAINING")
    logger.info("="*60)
    logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Server: 16 CPU, 32GB RAM")
    logger.info(f"Dataset: {DATA_DIR}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"CPU Cores: {NUM_CPU_CORES}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Log File: {log_filename}")
    logger.info("="*60)
    
    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_loader, val_loader, dataset = create_dataloaders(DATA_DIR, BATCH_SIZE)
    num_classes = dataset.get_num_classes()
    
    logger.info(f"Number of classes (people): {num_classes}")
    logger.info(f"Training batches per epoch: {len(train_loader)}")
    logger.info(f"Validation batches per epoch: {len(val_loader)}")
    
    # Save metadata
    dataset.save_metadata(save_path / "dataset_metadata.json")
    
    # Create model
    logger.info("Creating model: ResNet50 + ArcFace + Combined Loss")
    model = PalmVeinTrainingModel(
        num_classes=num_classes,
        backbone="resnet50",
        pretrained=True,
        embedding_dim=512,
        fusion_type="attention",
        use_attention=True,
        dropout=0.3,
        arcface_s=64.0,
        arcface_m=0.5
    )
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=512,
        arcface_weight=1.0,
        triplet_weight=0.3,
        center_weight=0.01,
        triplet_margin=0.3,
        label_smoothing=0.1
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # Scheduler - Cosine Annealing with Warmup
    warmup_epochs = 5
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': [], 'epoch_time': []
    }
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    
    logger.info("="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # ==================== TRAINING ====================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", 
                    file=sys.stdout, ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            rgb = batch['rgb'].to(DEVICE)
            ir = batch['ir'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(rgb, ir, labels)
            logits = output['logits']
            embeddings = output['embeddings']
            
            # Compute loss
            loss, loss_dict = criterion(logits, embeddings, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * rgb.size(0)
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
            batch_losses.append(loss.item())
            
            # Update progress bar
            current_acc = 100. * train_correct / train_total
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{current_acc:.2f}%"
            })
            
            # Log every N batches
            if (batch_idx + 1) % LOG_EVERY_N_BATCHES == 0:
                logger.debug(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)} - "
                           f"Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        
        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]",
                       file=sys.stdout, ncols=100)
            
            for batch in pbar:
                rgb = batch['rgb'].to(DEVICE)
                ir = batch['ir'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                output = model(rgb, ir, labels)
                logits = output['logits']
                embeddings = output['embeddings']
                
                loss, _ = criterion(logits, embeddings, labels)
                
                val_loss += loss.item() * rgb.size(0)
                _, predicted = logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.*val_correct/val_total:.2f}%"
                })
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Epoch time
        epoch_time = (time.time() - epoch_start) / 60  # minutes
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        
        # Log epoch summary
        logger.info("="*60)
        logger.info(f"EPOCH {epoch+1}/{EPOCHS} COMPLETED")
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        logger.info(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f} min")
        logger.info(f"  Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
        
        # Estimate remaining time
        avg_epoch_time = np.mean(history['epoch_time'])
        remaining_epochs = EPOCHS - epoch - 1
        remaining_time = remaining_epochs * avg_epoch_time / 60  # hours
        logger.info(f"  Estimated Remaining: {remaining_time:.1f} hours")
        logger.info("="*60)
        
        # Update progress file
        log_progress(epoch+1, EPOCHS, train_loss, train_acc, val_loss, val_acc, 
                    epoch_time, best_val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'num_classes': num_classes,
                'config': {
                    'backbone': 'resnet50',
                    'embedding_dim': 512,
                    'fusion_type': 'attention',
                    'num_classes': num_classes
                }
            }, save_path / "best_model.pth")
            
            logger.info(f"*** NEW BEST MODEL SAVED! Val Acc: {val_acc:.2f}% ***")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % SAVE_CHECKPOINT_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes
            }, save_path / f"checkpoint_epoch_{epoch+1}.pth")
            logger.info(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")
        
        # Save history after each epoch
        with open(save_path / "history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    # ==================== TRAINING COMPLETE ====================
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 3600  # hours
    
    # Save final model
    torch.save({
        'epoch': EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'config': {
            'backbone': 'resnet50',
            'embedding_dim': 512,
            'fusion_type': 'attention',
            'num_classes': num_classes
        }
    }, save_path / "final_model.pth")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED!")
    logger.info("="*60)
    logger.info(f"Total Training Time: {total_time:.2f} hours")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"Final Model: {save_path}/final_model.pth")
    logger.info(f"Best Model: {save_path}/best_model.pth")
    logger.info(f"Training Log: {log_filename}")
    logger.info("="*60)
    
    # Final progress update
    log_progress(EPOCHS, EPOCHS, train_loss, train_acc, val_loss, val_acc, 
                epoch_time, best_val_acc)


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        logger.info("\n*** Training interrupted by user ***")
    except Exception as e:
        logger.error(f"\n*** Training failed with error: {e} ***")
        raise