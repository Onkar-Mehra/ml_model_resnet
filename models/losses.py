"""
Loss Functions for Palm Vein Recognition (IMPROVED)
====================================================
Various loss functions for metric learning with combined loss support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class TripletLoss(nn.Module):
    """Triplet loss with online mining strategies."""
    
    def __init__(
        self,
        margin: float = 0.3,
        mining_strategy: str = "hard"
    ):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        # Compute pairwise distances
        distances = self._pairwise_distances(embeddings)
        
        if self.mining_strategy == "hard":
            return self._hard_mining_loss(distances, labels)
        elif self.mining_strategy == "semi_hard":
            return self._semi_hard_mining_loss(distances, labels)
        else:
            return self._all_triplets_loss(distances, labels)
    
    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0)
        return torch.sqrt(distances + 1e-16)
    
    def _hard_mining_loss(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        batch_size = labels.size(0)
        
        # Get positive and negative masks
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # Mask out diagonal
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        positive_mask = labels_equal & ~eye_mask
        negative_mask = labels_not_equal
        
        # Hard positive: maximum distance among positives
        hard_positive = (distances * positive_mask.float()).max(dim=1)[0]
        
        # Hard negative: minimum distance among negatives
        max_dist = distances.max()
        hard_negative = (distances + max_dist * (~negative_mask).float()).min(dim=1)[0]
        
        # Triplet loss
        loss = F.relu(hard_positive - hard_negative + self.margin)
        
        return loss.mean()
    
    def _semi_hard_mining_loss(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        batch_size = labels.size(0)
        
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        positive_mask = labels_equal & ~eye_mask
        negative_mask = ~labels_equal
        
        # For each anchor, get positive distance
        anchor_positive_dist = (distances * positive_mask.float()).max(dim=1, keepdim=True)[0]
        
        # Semi-hard negatives: farther than positive but within margin
        semi_hard_mask = negative_mask & (distances > anchor_positive_dist) & \
                         (distances < anchor_positive_dist + self.margin)
        
        # Use hard negatives if no semi-hard available
        max_dist = distances.max()
        semi_hard_dist = distances + max_dist * (~semi_hard_mask).float()
        hard_negative = semi_hard_dist.min(dim=1)[0]
        
        loss = F.relu(anchor_positive_dist.squeeze() - hard_negative + self.margin)
        
        return loss.mean()
    
    def _all_triplets_loss(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        batch_size = labels.size(0)
        
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        
        # Anchor-positive distances
        anchor_positive_dist = distances.unsqueeze(2)
        
        # Anchor-negative distances
        anchor_negative_dist = distances.unsqueeze(1)
        
        # Triplet loss for all valid triplets
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        
        # Mask for valid triplets
        positive_mask = labels_equal & ~eye_mask
        negative_mask = ~labels_equal
        valid_mask = positive_mask.unsqueeze(2) & negative_mask.unsqueeze(1)
        
        triplet_loss = triplet_loss * valid_mask.float()
        triplet_loss = F.relu(triplet_loss)
        
        # Average over valid triplets
        num_valid = valid_mask.sum()
        if num_valid > 0:
            loss = triplet_loss.sum() / num_valid
        else:
            loss = torch.tensor(0.0, device=distances.device)
        
        return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for pair-based learning."""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor  # 1 for same class, 0 for different
    ) -> torch.Tensor:
        distances = F.pairwise_distance(embeddings1, embeddings2)
        
        # Loss for positive pairs
        positive_loss = labels * distances.pow(2)
        
        # Loss for negative pairs
        negative_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        loss = (positive_loss + negative_loss).mean()
        
        return loss


class CenterLoss(nn.Module):
    """Center loss for intra-class variation minimization."""
    
    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        batch_centers = self.centers[labels]
        loss = F.mse_loss(embeddings, batch_centers)
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        n_classes = logits.size(-1)
        
        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(logits)
            smooth_labels.fill_(self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - self.smoothing)
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = (-smooth_labels * log_probs).sum(dim=-1).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """Combined loss function for maximum accuracy."""
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 1024,
        arcface_weight: float = 1.0,
        triplet_weight: float = 0.3,
        center_weight: float = 0.01,
        triplet_margin: float = 0.3,
        label_smoothing: float = 0.1,
        use_focal: bool = False,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.arcface_weight = arcface_weight
        self.triplet_weight = triplet_weight
        self.center_weight = center_weight
        
        # Primary classification loss
        if use_focal:
            self.classification_loss = FocalLoss(gamma=focal_gamma)
        elif label_smoothing > 0:
            self.classification_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.classification_loss = nn.CrossEntropyLoss()
        
        # Triplet loss for embedding quality
        self.triplet_loss = TripletLoss(margin=triplet_margin, mining_strategy="hard")
        
        # Center loss for intra-class compactness
        self.center_loss = CenterLoss(num_classes, embedding_dim)
    
    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Classification loss (with ArcFace logits)
        cls_loss = self.classification_loss(logits, labels)
        
        # Triplet loss
        tri_loss = self.triplet_loss(embeddings, labels)
        
        # Center loss
        ctr_loss = self.center_loss(embeddings, labels)
        
        # Combined loss
        total_loss = (
            self.arcface_weight * cls_loss +
            self.triplet_weight * tri_loss +
            self.center_weight * ctr_loss
        )
        
        # Loss breakdown for logging
        loss_dict = {
            'total': total_loss.item(),
            'arcface': cls_loss.item(),
            'triplet': tri_loss.item(),
            'center': ctr_loss.item()
        }
        
        return total_loss, loss_dict


class AdaptiveCombinedLoss(nn.Module):
    """Adaptive combined loss that adjusts weights during training."""
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 1024,
        triplet_margin: float = 0.3,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        # Learnable loss weights
        self.log_arcface_weight = nn.Parameter(torch.tensor(0.0))
        self.log_triplet_weight = nn.Parameter(torch.tensor(-1.2))  # ~0.3
        self.log_center_weight = nn.Parameter(torch.tensor(-4.6))   # ~0.01
        
        # Loss functions
        self.classification_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.triplet_loss = TripletLoss(margin=triplet_margin, mining_strategy="hard")
        self.center_loss = CenterLoss(num_classes, embedding_dim)
    
    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Get weights
        arcface_weight = torch.exp(self.log_arcface_weight)
        triplet_weight = torch.exp(self.log_triplet_weight)
        center_weight = torch.exp(self.log_center_weight)
        
        # Compute losses
        cls_loss = self.classification_loss(logits, labels)
        tri_loss = self.triplet_loss(embeddings, labels)
        ctr_loss = self.center_loss(embeddings, labels)
        
        # Weighted sum with regularization
        total_loss = (
            arcface_weight * cls_loss +
            triplet_weight * tri_loss +
            center_weight * ctr_loss +
            self.log_arcface_weight + self.log_triplet_weight + self.log_center_weight
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'arcface': cls_loss.item(),
            'triplet': tri_loss.item(),
            'center': ctr_loss.item(),
            'w_arcface': arcface_weight.item(),
            'w_triplet': triplet_weight.item(),
            'w_center': center_weight.item()
        }
        
        return total_loss, loss_dict
