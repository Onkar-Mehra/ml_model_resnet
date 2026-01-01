"""
Palm Vein Image Preprocessing Module (Fixed for Windows)
=========================================================
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PalmROIExtractor:
    """Extracts the Region of Interest (ROI) from palm images."""
    
    def __init__(self, target_size: Tuple[int, int] = (200, 200)):
        self.target_size = target_size
    
    def extract(self, image: np.ndarray, is_ir: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if is_ir:
            _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        else:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._center_crop(image), np.ones(self.target_size, dtype=np.uint8) * 255
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        padding = int(min(w, h) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        if len(image.shape) == 3:
            roi = image[y:y+h, x:x+w]
        else:
            roi = gray[y:y+h, x:x+w]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = largest_contour - np.array([x, y])
        cv2.drawContours(mask, [shifted_contour], -1, 255, -1)
        
        roi = cv2.resize(roi, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        return roi, mask
    
    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        th, tw = self.target_size
        
        start_y = max(0, (h - th) // 2)
        start_x = max(0, (w - tw) // 2)
        
        if len(image.shape) == 3:
            crop = image[start_y:start_y+th, start_x:start_x+tw]
        else:
            crop = image[start_y:start_y+th, start_x:start_x+tw]
        
        return cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LANCZOS4)


class ImageEnhancer:
    """Enhances palm images for better vein visibility."""
    
    def __init__(
        self,
        clahe_clip_limit: float = 3.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        # Store parameters instead of CLAHE object (fixes Windows pickle error)
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
    
    def _get_clahe(self):
        """Create CLAHE object on demand (not stored as instance variable)."""
        return cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )
    
    def enhance(self, image: np.ndarray, is_ir: bool = False) -> np.ndarray:
        if len(image.shape) == 3:
            if is_ir:
                gray = image[:, :, 1]
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create CLAHE on demand
        clahe = self._get_clahe()
        enhanced = clahe.apply(gray)
        
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return enhanced
    
    def enhance_multiscale(self, image: np.ndarray, is_ir: bool = False) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if not is_ir else image[:, :, 1]
        else:
            gray = image.copy()
        
        scales = [15, 80, 250]
        retinex = np.zeros_like(gray, dtype=np.float64)
        
        for scale in scales:
            blur = cv2.GaussianBlur(gray.astype(np.float64), (0, 0), scale)
            retinex += np.log10(gray.astype(np.float64) + 1) - np.log10(blur + 1)
        
        retinex = retinex / len(scales)
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create CLAHE on demand
        clahe = self._get_clahe()
        enhanced = clahe.apply(retinex)
        
        return enhanced


class VeinExtractor:
    """Extracts vein patterns from palm images using Gabor filters."""
    
    def __init__(
        self,
        ksize: int = 31,
        sigma: float = 4.0,
        theta_count: int = 8,
        lambd: float = 10.0,
        gamma: float = 0.5
    ):
        self.ksize = ksize
        self.sigma = sigma
        self.theta_count = theta_count
        self.lambd = lambd
        self.gamma = gamma
        self.gabor_kernels = None  # Create on demand
    
    def _get_gabor_kernels(self) -> List[np.ndarray]:
        """Create Gabor kernels on demand."""
        if self.gabor_kernels is None:
            self.gabor_kernels = []
            for i in range(self.theta_count):
                theta = i * np.pi / self.theta_count
                kernel = cv2.getGaborKernel(
                    (self.ksize, self.ksize),
                    self.sigma,
                    theta,
                    self.lambd,
                    self.gamma,
                    0,
                    ktype=cv2.CV_64F
                )
                self.gabor_kernels.append(kernel)
        return self.gabor_kernels
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        kernels = self._get_gabor_kernels()
        responses = []
        for kernel in kernels:
            response = cv2.filter2D(image.astype(np.float64), cv2.CV_64F, kernel)
            responses.append(np.abs(response))
        
        vein_pattern = np.max(responses, axis=0)
        vein_pattern = cv2.normalize(vein_pattern, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vein_pattern = cv2.morphologyEx(vein_pattern, cv2.MORPH_CLOSE, kernel)
        
        return vein_pattern
    
    def extract_skeleton(self, vein_pattern: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(vein_pattern, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(binary, opened)
            eroded = cv2.erode(binary, element)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = eroded.copy()
            
            if cv2.countNonZero(binary) == 0:
                break
        
        return skeleton


class PalmPreprocessor:
    """Complete preprocessing pipeline for palm vein images."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        clahe_clip_limit: float = 3.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        gabor_params: Optional[dict] = None
    ):
        self.target_size = target_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.gabor_params = gabor_params or {}
        
        # Don't create objects here - create on demand to avoid pickle issues
        self._roi_extractor = None
        self._enhancer = None
        self._vein_extractor = None
    
    @property
    def roi_extractor(self):
        if self._roi_extractor is None:
            self._roi_extractor = PalmROIExtractor(self.target_size)
        return self._roi_extractor
    
    @property
    def enhancer(self):
        if self._enhancer is None:
            self._enhancer = ImageEnhancer(self.clahe_clip_limit, self.clahe_tile_grid_size)
        return self._enhancer
    
    @property
    def vein_extractor(self):
        if self._vein_extractor is None:
            self._vein_extractor = VeinExtractor(**self.gabor_params)
        return self._vein_extractor
    
    def process(
        self,
        image: np.ndarray,
        is_ir: bool = False,
        extract_veins: bool = True
    ) -> dict:
        result = {}
        
        roi, mask = self.roi_extractor.extract(image, is_ir)
        result['roi'] = roi
        result['mask'] = mask
        
        enhanced = self.enhancer.enhance(roi, is_ir)
        result['enhanced'] = enhanced
        
        enhanced_ms = self.enhancer.enhance_multiscale(roi, is_ir)
        result['enhanced_multiscale'] = enhanced_ms
        
        if extract_veins:
            vein_pattern = self.vein_extractor.extract(enhanced_ms)
            result['vein_pattern'] = vein_pattern
            
            skeleton = self.vein_extractor.extract_skeleton(vein_pattern)
            result['skeleton'] = skeleton
        
        return result
    
    def process_pair(
        self,
        rgb_image: np.ndarray,
        ir_image: np.ndarray
    ) -> dict:
        rgb_result = self.process(rgb_image, is_ir=False)
        ir_result = self.process(ir_image, is_ir=True)
        
        return {
            'rgb': rgb_result,
            'ir': ir_result
        }


class DataAugmentation:
    """Data augmentation for palm vein images."""
    
    def __init__(self, strength: str = "strong"):
        self.strength = strength
        
        if strength == "none":
            self.params = {}
        elif strength == "light":
            self.params = {
                'rotation_range': 10,
                'scale_range': (0.95, 1.05),
                'translate_range': 0.05,
                'brightness_range': (0.9, 1.1),
                'flip_prob': 0.0
            }
        else:
            self.params = {
                'rotation_range': 20,
                'scale_range': (0.85, 1.15),
                'translate_range': 0.1,
                'brightness_range': (0.8, 1.2),
                'flip_prob': 0.5,
                'gaussian_noise_std': 10,
                'blur_prob': 0.3
            }
    
    def augment(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        if self.strength == "none":
            return image
        
        if seed is not None:
            np.random.seed(seed)
        
        h, w = image.shape[:2]
        augmented = image.copy()
        
        if 'rotation_range' in self.params:
            angle = np.random.uniform(-self.params['rotation_range'], self.params['rotation_range'])
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            augmented = cv2.warpAffine(augmented, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        if 'scale_range' in self.params:
            scale = np.random.uniform(*self.params['scale_range'])
            new_w, new_h = int(w * scale), int(h * scale)
            augmented = cv2.resize(augmented, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            if scale > 1:
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                augmented = augmented[start_y:start_y+h, start_x:start_x+w]
            else:
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                augmented = cv2.copyMakeBorder(augmented, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT)
        
        if 'translate_range' in self.params:
            tx = np.random.uniform(-self.params['translate_range'], self.params['translate_range']) * w
            ty = np.random.uniform(-self.params['translate_range'], self.params['translate_range']) * h
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            augmented = cv2.warpAffine(augmented, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        if 'brightness_range' in self.params:
            factor = np.random.uniform(*self.params['brightness_range'])
            augmented = np.clip(augmented * factor, 0, 255).astype(np.uint8)
        
        if 'flip_prob' in self.params and np.random.random() < self.params['flip_prob']:
            augmented = cv2.flip(augmented, 1)
        
        if 'gaussian_noise_std' in self.params:
            noise = np.random.normal(0, self.params['gaussian_noise_std'], augmented.shape)
            augmented = np.clip(augmented + noise, 0, 255).astype(np.uint8)
        
        if 'blur_prob' in self.params and np.random.random() < self.params['blur_prob']:
            ksize = np.random.choice([3, 5])
            augmented = cv2.GaussianBlur(augmented, (ksize, ksize), 0)
        
        return augmented


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from path."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image from {path}")
    return image


def save_image(image: np.ndarray, path: Union[str, Path]):
    """Save an image to path."""
    cv2.imwrite(str(path), image)