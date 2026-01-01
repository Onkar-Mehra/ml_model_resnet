"""
Dataset Module for Palm Vein Biometric System (Fixed for Windows)
==================================================================
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union, Callable
import logging
from collections import defaultdict
import random

from preprocessing.image_processor import (
    PalmPreprocessor,
    DataAugmentation,
    load_image
)

logger = logging.getLogger(__name__)


class PalmVeinDataset(Dataset):
    """Dataset for palm vein images with RGB and IR modalities."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        target_size: Tuple[int, int] = (224, 224),
        augmentation: Optional[DataAugmentation] = None,
        preprocess: bool = True,
        cache_preprocessed: bool = False,
        transform: Optional[Callable] = None,
        mode: str = "train"
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.augmentation = augmentation
        self.preprocess = preprocess
        self.cache_preprocessed = cache_preprocessed
        self.transform = transform
        self.mode = mode
        
        # Store parameters for lazy initialization (Windows pickle fix)
        self._preprocessor = None
        
        self.samples = []
        self.labels = []
        self.label_to_name = {}
        self.name_to_label = {}
        
        self.cache = {} if cache_preprocessed else None
        
        self._load_dataset()
    
    @property
    def preprocessor(self):
        """Lazy initialization of preprocessor."""
        if self._preprocessor is None:
            self._preprocessor = PalmPreprocessor(target_size=self.target_size)
        return self._preprocessor
    
    def _load_dataset(self):
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            self._load_nested_structure(subdirs)
        else:
            self._load_flat_structure()
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.label_to_name)} identities")
    
    def _load_nested_structure(self, subdirs: List[Path]):
        for idx, person_dir in enumerate(sorted(subdirs)):
            person_name = person_dir.name
            self.label_to_name[idx] = person_name
            self.name_to_label[person_name] = idx
            
            rgb_images = list(person_dir.glob("*rgb*")) + list(person_dir.glob("*RGB*"))
            ir_images = list(person_dir.glob("*ir*")) + list(person_dir.glob("*IR*"))
            
            if not rgb_images or not ir_images:
                logger.warning(f"Missing images for {person_name}")
                continue
            
            for rgb_path, ir_path in zip(sorted(rgb_images), sorted(ir_images)):
                self.samples.append({
                    'rgb_path': rgb_path,
                    'ir_path': ir_path,
                    'label': idx,
                    'name': person_name
                })
                self.labels.append(idx)
    
    def _load_flat_structure(self):
        person_files = defaultdict(dict)
        
        for file_path in self.data_dir.glob("*"):
            if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            filename = file_path.stem.lower()
            
            if '_rgb' in filename or '_ir' in filename:
                parts = filename.rsplit('_', 1)
                person_id = parts[0]
                modality = parts[1]
            elif 'rgb' in filename:
                person_id = filename.replace('rgb', '').strip('_')
                modality = 'rgb'
            elif 'ir' in filename:
                person_id = filename.replace('ir', '').strip('_')
                modality = 'ir'
            else:
                continue
            
            person_files[person_id][modality] = file_path
        
        for idx, (person_id, files) in enumerate(sorted(person_files.items())):
            if 'rgb' not in files or 'ir' not in files:
                logger.warning(f"Missing modality for {person_id}")
                continue
            
            self.label_to_name[idx] = person_id
            self.name_to_label[person_id] = idx
            
            self.samples.append({
                'rgb_path': files['rgb'],
                'ir_path': files['ir'],
                'label': idx,
                'name': person_id
            })
            self.labels.append(idx)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        if self.cache is not None and idx in self.cache:
            cached = self.cache[idx]
            if self.mode == "train" and self.augmentation:
                rgb = self.augmentation.augment(cached['rgb_processed'])
                ir = self.augmentation.augment(cached['ir_processed'])
            else:
                rgb = cached['rgb_processed']
                ir = cached['ir_processed']
        else:
            rgb_image = load_image(sample['rgb_path'])
            ir_image = load_image(sample['ir_path'])
            
            if self.preprocess:
                rgb_result = self.preprocessor.process(rgb_image, is_ir=False)
                ir_result = self.preprocessor.process(ir_image, is_ir=True)
                
                rgb = rgb_result['enhanced_multiscale']
                ir = ir_result['enhanced_multiscale']
            else:
                rgb = cv2.resize(rgb_image, self.target_size)
                ir = cv2.resize(ir_image, self.target_size)
                if len(ir.shape) == 3:
                    ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
            
            if self.cache is not None:
                self.cache[idx] = {
                    'rgb_processed': rgb.copy(),
                    'ir_processed': ir.copy()
                }
            
            if self.mode == "train" and self.augmentation:
                seed = random.randint(0, 2**32 - 1)
                rgb = self.augmentation.augment(rgb, seed=seed)
                ir = self.augmentation.augment(ir, seed=seed)
        
        rgb_tensor = self._to_tensor(rgb, channels=3)
        ir_tensor = self._to_tensor(ir, channels=1)
        
        if self.transform:
            rgb_tensor = self.transform(rgb_tensor)
            ir_tensor = self.transform(ir_tensor)
        
        return {
            'rgb': rgb_tensor,
            'ir': ir_tensor,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'name': sample['name']
        }
    
    def _to_tensor(self, image: np.ndarray, channels: int) -> torch.Tensor:
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        
        if channels == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels == 1 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image)
    
    def get_num_classes(self) -> int:
        return len(self.label_to_name)
    
    def get_name_by_label(self, label: int) -> str:
        return self.label_to_name.get(label, "Unknown")
    
    def get_label_by_name(self, name: str) -> int:
        return self.name_to_label.get(name, -1)
    
    def save_metadata(self, path: Union[str, Path]):
        metadata = {
            'label_to_name': {str(k): v for k, v in self.label_to_name.items()},
            'name_to_label': self.name_to_label,
            'num_samples': len(self.samples),
            'num_classes': len(self.label_to_name)
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load_metadata(cls, path: Union[str, Path]) -> dict:
        with open(path, 'r') as f:
            metadata = json.load(f)
        metadata['label_to_name'] = {int(k): v for k, v in metadata['label_to_name'].items()}
        return metadata


class InferenceDataset(Dataset):
    """Simple dataset for inference on new images."""
    
    def __init__(
        self,
        rgb_paths: List[Union[str, Path]],
        ir_paths: List[Union[str, Path]],
        target_size: Tuple[int, int] = (224, 224)
    ):
        assert len(rgb_paths) == len(ir_paths)
        
        self.rgb_paths = rgb_paths
        self.ir_paths = ir_paths
        self.target_size = target_size
        self._preprocessor = None
    
    @property
    def preprocessor(self):
        if self._preprocessor is None:
            self._preprocessor = PalmPreprocessor(target_size=self.target_size)
        return self._preprocessor
    
    def __len__(self) -> int:
        return len(self.rgb_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_image = load_image(self.rgb_paths[idx])
        ir_image = load_image(self.ir_paths[idx])
        
        rgb_result = self.preprocessor.process(rgb_image, is_ir=False)
        ir_result = self.preprocessor.process(ir_image, is_ir=True)
        
        rgb = rgb_result['enhanced_multiscale']
        ir = ir_result['enhanced_multiscale']
        
        rgb_tensor = self._to_tensor(rgb, channels=3)
        ir_tensor = self._to_tensor(ir, channels=1)
        
        return {
            'rgb': rgb_tensor,
            'ir': ir_tensor,
            'rgb_path': str(self.rgb_paths[idx]),
            'ir_path': str(self.ir_paths[idx])
        }
    
    def _to_tensor(self, image: np.ndarray, channels: int) -> torch.Tensor:
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        
        if channels == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels == 1 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image)


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    augmentation_strength: str = "strong",
    val_split: float = 0.2,
    num_workers: int = 0,  # SET TO 0 FOR WINDOWS
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, PalmVeinDataset]:
    """Create training and validation dataloaders."""
    random.seed(seed)
    np.random.seed(seed)
    
    train_augmentation = DataAugmentation(strength=augmentation_strength)
    
    train_dataset = PalmVeinDataset(
        data_dir=data_dir,
        target_size=target_size,
        augmentation=train_augmentation,
        mode="train"
    )
    
    val_dataset = PalmVeinDataset(
        data_dir=data_dir,
        target_size=target_size,
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
    
    # num_workers=0 for Windows compatibility
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,  # Disabled for CPU
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False  # Disabled for CPU
    )
    
    return train_loader, val_loader, train_dataset