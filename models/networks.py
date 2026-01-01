"""
Palm Vein Recognition Neural Network Models (IMPROVED)
======================================================
Deep learning architectures for palm vein feature extraction and matching.
Optimized for maximum accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, List, Dict
import math


class ChannelAttention(nn.Module):
    """Channel attention module for feature recalibration."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on important regions."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ConvBlock(nn.Module):
    """Basic convolutional block with batch norm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_attention: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.attention(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with optional attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_attention: bool = False
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.attention(out)
        out += identity
        out = self.relu(out)
        
        return out


class BilinearFusion(nn.Module):
    """Bilinear fusion for combining RGB and IR features."""
    
    def __init__(self, feature_dim: int, output_dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(feature_dim, feature_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, rgb_features: torch.Tensor, ir_features: torch.Tensor) -> torch.Tensor:
        fused = self.bilinear(rgb_features, ir_features)
        fused = self.bn(fused)
        fused = self.relu(fused)
        return fused


class CustomCNNBackbone(nn.Module):
    """Custom CNN backbone optimized for palm vein patterns."""
    
    def __init__(self, in_channels: int = 3, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Feature extraction blocks
        self.block1 = self._make_block(64, 128, 2)
        self.block2 = self._make_block(128, 256, 2)
        self.block3 = self._make_block(256, 512, 2)
        self.block4 = self._make_block(512, 512, 1)
        
        # Attention modules
        if use_attention:
            self.cbam1 = CBAM(128)
            self.cbam2 = CBAM(256)
            self.cbam3 = CBAM(512)
            self.cbam4 = CBAM(512)
        
        self.output_channels = 512
    
    def _make_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        
        x = self.block1(x)
        if self.use_attention:
            x = self.cbam1(x)
        
        x = self.block2(x)
        if self.use_attention:
            x = self.cbam2(x)
        
        x = self.block3(x)
        if self.use_attention:
            x = self.cbam3(x)
        
        x = self.block4(x)
        if self.use_attention:
            x = self.cbam4(x)
        
        return x


class FeatureExtractor(nn.Module):
    """Feature extractor with multiple backbone options."""
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        in_channels: int = 3,
        use_attention: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == "custom":
            self.backbone = CustomCNNBackbone(in_channels, use_attention)
            self.output_channels = 512
        
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base_model = models.resnet50(weights=weights)
            
            # Modify first layer if needed
            if in_channels != 3:
                base_model.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            
            # Remove final layers
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])
            self.output_channels = 2048
            
            # Add attention
            if use_attention:
                self.attention = CBAM(self.output_channels)
            else:
                self.attention = None
        
        elif backbone == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.efficientnet_b3(weights=weights)
            
            if in_channels != 3:
                base_model.features[0][0] = nn.Conv2d(
                    in_channels, 40, kernel_size=3, stride=2, padding=1, bias=False
                )
            
            self.backbone = base_model.features
            self.output_channels = 1536
            
            if use_attention:
                self.attention = CBAM(self.output_channels)
            else:
                self.attention = None
        
        elif backbone == "vgg19":
            weights = models.VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.vgg19_bn(weights=weights)
            
            if in_channels != 3:
                base_model.features[0] = nn.Conv2d(
                    in_channels, 64, kernel_size=3, stride=1, padding=1
                )
            
            self.backbone = base_model.features
            self.output_channels = 512
            
            if use_attention:
                self.attention = CBAM(self.output_channels)
            else:
                self.attention = None
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        
        if hasattr(self, 'attention') and self.attention is not None:
            x = self.attention(x)
        
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal attention for RGB-IR feature fusion."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query_rgb = nn.Conv2d(channels, channels, 1)
        self.key_ir = nn.Conv2d(channels, channels, 1)
        self.value_ir = nn.Conv2d(channels, channels, 1)
        
        self.query_ir = nn.Conv2d(channels, channels, 1)
        self.key_rgb = nn.Conv2d(channels, channels, 1)
        self.value_rgb = nn.Conv2d(channels, channels, 1)
        
        self.proj_rgb = nn.Conv2d(channels, channels, 1)
        self.proj_ir = nn.Conv2d(channels, channels, 1)
        
        self.norm_rgb = nn.LayerNorm(channels)
        self.norm_ir = nn.LayerNorm(channels)
    
    def forward(
        self,
        rgb_features: torch.Tensor,
        ir_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = rgb_features.shape
        
        # RGB attends to IR
        q_rgb = self.query_rgb(rgb_features).view(b, self.num_heads, self.head_dim, h * w)
        k_ir = self.key_ir(ir_features).view(b, self.num_heads, self.head_dim, h * w)
        v_ir = self.value_ir(ir_features).view(b, self.num_heads, self.head_dim, h * w)
        
        attn_rgb = torch.matmul(q_rgb.transpose(-2, -1), k_ir) * self.scale
        attn_rgb = F.softmax(attn_rgb, dim=-1)
        rgb_attended = torch.matmul(v_ir, attn_rgb.transpose(-2, -1))
        rgb_attended = rgb_attended.view(b, c, h, w)
        rgb_attended = self.proj_rgb(rgb_attended)
        rgb_out = rgb_features + rgb_attended
        
        # IR attends to RGB
        q_ir = self.query_ir(ir_features).view(b, self.num_heads, self.head_dim, h * w)
        k_rgb = self.key_rgb(rgb_features).view(b, self.num_heads, self.head_dim, h * w)
        v_rgb = self.value_rgb(rgb_features).view(b, self.num_heads, self.head_dim, h * w)
        
        attn_ir = torch.matmul(q_ir.transpose(-2, -1), k_rgb) * self.scale
        attn_ir = F.softmax(attn_ir, dim=-1)
        ir_attended = torch.matmul(v_rgb, attn_ir.transpose(-2, -1))
        ir_attended = ir_attended.view(b, c, h, w)
        ir_attended = self.proj_ir(ir_attended)
        ir_out = ir_features + ir_attended
        
        return rgb_out, ir_out


class MultiModalFusion(nn.Module):
    """Multi-modal fusion module for combining RGB and IR features."""
    
    def __init__(
        self,
        channels: int,
        fusion_type: str = "attention",
        num_heads: int = 8
    ):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        
        elif fusion_type == "attention":
            self.cross_attention = CrossModalAttention(channels, num_heads)
            self.fusion = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        
        elif fusion_type == "bilinear":
            self.bilinear = nn.Bilinear(channels, channels, channels)
            self.pool = nn.AdaptiveAvgPool2d(1)
        
        elif fusion_type == "weighted":
            self.rgb_weight = nn.Parameter(torch.tensor(0.4))
            self.ir_weight = nn.Parameter(torch.tensor(0.6))
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        rgb_features: torch.Tensor,
        ir_features: torch.Tensor
    ) -> torch.Tensor:
        if self.fusion_type == "concat":
            fused = torch.cat([rgb_features, ir_features], dim=1)
            return self.fusion(fused)
        
        elif self.fusion_type == "attention":
            rgb_att, ir_att = self.cross_attention(rgb_features, ir_features)
            fused = torch.cat([rgb_att, ir_att], dim=1)
            return self.fusion(fused)
        
        elif self.fusion_type == "bilinear":
            b, c, h, w = rgb_features.shape
            rgb_pooled = self.pool(rgb_features).view(b, c)
            ir_pooled = self.pool(ir_features).view(b, c)
            fused = self.bilinear(rgb_pooled, ir_pooled)
            return fused.unsqueeze(-1).unsqueeze(-1).expand(b, c, h, w)
        
        elif self.fusion_type == "weighted":
            rgb_w = torch.sigmoid(self.rgb_weight)
            ir_w = torch.sigmoid(self.ir_weight)
            total = rgb_w + ir_w
            return (rgb_w / total) * rgb_features + (ir_w / total) * ir_features


class EmbeddingHead(nn.Module):
    """Embedding head for generating normalized feature vectors."""
    
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 1024,
        dropout: float = 0.2
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class ArcFaceHead(nn.Module):
    """ArcFace classification head for metric learning."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.5,
        easy_margin: bool = False
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Normalize weight
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weight_norm)
        
        if labels is None:
            return cosine * self.s
        
        # ArcFace margin
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


class PalmVeinNet(nn.Module):
    """Complete palm vein recognition network."""
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        embedding_dim: int = 1024,
        fusion_type: str = "attention",
        use_attention: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # RGB feature extractor
        self.rgb_extractor = FeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
            in_channels=3,
            use_attention=use_attention
        )
        
        # IR feature extractor
        self.ir_extractor = FeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
            in_channels=1,
            use_attention=use_attention
        )
        
        # Feature fusion
        feature_channels = self.rgb_extractor.output_channels
        self.fusion = MultiModalFusion(
            channels=feature_channels,
            fusion_type=fusion_type,
            num_heads=8
        )
        
        # Embedding head
        self.embedding_head = EmbeddingHead(
            in_channels=feature_channels,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
    
    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor
    ) -> torch.Tensor:
        # Extract features
        rgb_features = self.rgb_extractor(rgb)
        ir_features = self.ir_extractor(ir)
        
        # Fuse features
        fused_features = self.fusion(rgb_features, ir_features)
        
        # Generate embedding
        embedding = self.embedding_head(fused_features)
        
        return embedding
    
    def extract_features(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract individual and fused features."""
        rgb_features = self.rgb_extractor(rgb)
        ir_features = self.ir_extractor(ir)
        fused_features = self.fusion(rgb_features, ir_features)
        
        return rgb_features, ir_features, fused_features


class PalmVeinTrainingModel(nn.Module):
    """Training model with ArcFace head."""
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        embedding_dim: int = 1024,
        fusion_type: str = "attention",
        use_attention: bool = True,
        dropout: float = 0.2,
        arcface_s: float = 64.0,
        arcface_m: float = 0.5
    ):
        super().__init__()
        
        # Base network
        self.backbone = PalmVeinNet(
            backbone=backbone,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            fusion_type=fusion_type,
            use_attention=use_attention,
            dropout=dropout
        )
        
        # ArcFace head
        self.arcface = ArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            s=arcface_s,
            m=arcface_m
        )
    
    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.backbone(rgb, ir)
        logits = self.arcface(embeddings, labels)
        
        return logits, embeddings
    
    def get_embedding(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor
    ) -> torch.Tensor:
        """Get embeddings without classification."""
        return self.backbone(rgb, ir)
