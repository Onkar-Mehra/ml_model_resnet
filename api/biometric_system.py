"""
Palm Vein Biometric System API
==============================
Main API for enrollment and verification of palm vein biometrics.
This is the primary interface for using the system.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
import logging
from datetime import datetime
import json

from preprocessing.image_processor import PalmPreprocessor, load_image
from models.networks import PalmVeinNet, PalmVeinTrainingModel
from matching.database import EmbeddingDatabase, MultiTemplateDatabase
from config.settings import SystemConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class PalmVeinBiometricSystem:
    """
    Complete palm vein biometric system for enrollment and verification.
    
    This class provides the main interface for:
    1. Enrolling new users with their RGB and IR palm images
    2. Verifying users against enrolled templates
    3. Identifying unknown users
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        database_path: Optional[Union[str, Path]] = None,
        config: Optional[SystemConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the biometric system.
        
        Args:
            model_path: Path to trained model checkpoint
            database_path: Path to embedding database
            config: System configuration
            device: Device to use (cuda/cpu)
        """
        self.config = config or DEFAULT_CONFIG
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize preprocessor
        self.preprocessor = PalmPreprocessor(
            target_size=self.config.image.input_size,
            clahe_clip_limit=self.config.image.clahe_clip_limit,
            clahe_tile_grid_size=self.config.image.clahe_tile_grid_size
        )
        
        # Initialize model
        self.model = None
        if model_path:
            self.load_model(model_path)
        
        # Initialize database
        self.database = MultiTemplateDatabase(
            embedding_dim=self.config.model.embedding_dim,
            distance_metric=self.config.verification.distance_metric,
            db_path=database_path,
            use_faiss=True
        )
        
        # Verification threshold
        self.threshold = self.config.verification.similarity_threshold
        
        logger.info(f"Biometric system initialized on {self.device}")
    
    def load_model(self, model_path: Union[str, Path]):
        """Load a trained model from checkpoint."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model config from checkpoint
        model_config = checkpoint.get('config', {})
        embedding_dim = model_config.get('embedding_dim', self.config.model.embedding_dim)
        
        # Create model architecture
        self.model = PalmVeinNet(
            backbone=self.config.model.backbone,
            pretrained=False,  # Don't need pretrained for inference
            embedding_dim=embedding_dim,
            fusion_type=self.config.model.fusion_type,
            use_attention=self.config.model.use_attention,
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        # Load weights
        # Handle case where checkpoint has full training model
        state_dict = checkpoint['model_state_dict']
        
        # Filter out ArcFace head weights if present
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                model_state_dict[new_key] = value
            elif not key.startswith('arcface.'):
                model_state_dict[key] = value
        
        # Try loading filtered state dict first
        try:
            self.model.load_state_dict(model_state_dict, strict=False)
        except Exception as e:
            logger.warning(f"Partial model loading: {e}")
            # Try loading original state dict
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
    
    def preprocess_images(
        self,
        rgb_image: Union[np.ndarray, str, Path],
        ir_image: Union[np.ndarray, str, Path]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess RGB and IR images for the model.
        
        Args:
            rgb_image: RGB image (array or path)
            ir_image: IR image (array or path)
            
        Returns:
            Preprocessed RGB and IR tensors
        """
        # Load images if paths provided
        if isinstance(rgb_image, (str, Path)):
            rgb_image = load_image(rgb_image)
        if isinstance(ir_image, (str, Path)):
            ir_image = load_image(ir_image)
        
        # Preprocess
        rgb_result = self.preprocessor.process(rgb_image, is_ir=False)
        ir_result = self.preprocessor.process(ir_image, is_ir=True)
        
        # Get enhanced images
        rgb_processed = rgb_result['enhanced_multiscale']
        ir_processed = ir_result['enhanced_multiscale']
        
        # Convert to tensors
        rgb_tensor = self._to_tensor(rgb_processed, channels=3)
        ir_tensor = self._to_tensor(ir_processed, channels=1)
        
        return rgb_tensor, ir_tensor
    
    def _to_tensor(self, image: np.ndarray, channels: int) -> torch.Tensor:
        """Convert image to normalized tensor."""
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        
        if channels == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels == 1 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image)
    
    @torch.no_grad()
    def extract_embedding(
        self,
        rgb_image: Union[np.ndarray, str, Path],
        ir_image: Union[np.ndarray, str, Path]
    ) -> np.ndarray:
        """
        Extract embedding from RGB and IR images.
        
        Args:
            rgb_image: RGB palm image
            ir_image: IR palm image
            
        Returns:
            Normalized embedding vector
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        rgb_tensor, ir_tensor = self.preprocess_images(rgb_image, ir_image)
        
        # Add batch dimension
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        ir_tensor = ir_tensor.unsqueeze(0).to(self.device)
        
        # Extract embedding
        embedding = self.model(rgb_tensor, ir_tensor)
        
        return embedding.cpu().numpy().flatten()
    
    def enroll(
        self,
        rgb_image: Union[np.ndarray, str, Path],
        ir_image: Union[np.ndarray, str, Path],
        name: str,
        additional_info: Optional[Dict] = None
    ) -> Tuple[bool, int, str]:
        """
        Enroll a new user with their palm vein images.
        
        This function takes 2 images (RGB and IR) of the same hand
        and stores the extracted features for future verification.
        
        Args:
            rgb_image: RGB image of the palm
            ir_image: Infrared image of the palm
            name: User's name/identifier
            additional_info: Additional metadata to store
            
        Returns:
            (success, user_id, message)
        """
        try:
            # Extract embedding
            embedding = self.extract_embedding(rgb_image, ir_image)
            
            # Check if user already exists
            existing_ids = self.database.name_to_ids.get(name, [])
            if existing_ids:
                logger.warning(f"User {name} already enrolled. Adding additional template.")
            
            # Enroll in database
            user_id = self.database.enroll(
                embedding=embedding,
                name=name,
                additional_info=additional_info
            )
            
            # Save database
            if self.database.db_path:
                self.database.save()
            
            message = f"Successfully enrolled {name} with ID {user_id}"
            logger.info(message)
            
            return True, user_id, message
            
        except Exception as e:
            message = f"Enrollment failed: {str(e)}"
            logger.error(message)
            return False, -1, message
    
    def verify(
        self,
        rgb_image: Union[np.ndarray, str, Path],
        ir_image: Union[np.ndarray, str, Path],
        claimed_name: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Verify if the palm images match the claimed identity.
        
        Args:
            rgb_image: RGB image of the palm
            ir_image: Infrared image of the palm
            claimed_name: The identity being claimed
            threshold: Verification threshold (optional, uses default if not provided)
            
        Returns:
            (is_verified, similarity_score, message)
        """
        threshold = threshold or self.threshold
        
        try:
            # Extract embedding
            query_embedding = self.extract_embedding(rgb_image, ir_image)
            
            # Verify against claimed identity
            is_verified, similarity, match_info = self.database.verify_multi_template(
                query_embedding=query_embedding,
                claimed_name=claimed_name,
                threshold=threshold,
                aggregation="max"
            )
            
            if is_verified:
                message = f"Verification SUCCESSFUL for {claimed_name} (similarity: {similarity:.4f})"
            else:
                message = f"Verification FAILED for {claimed_name} (similarity: {similarity:.4f})"
            
            logger.info(message)
            
            return is_verified, similarity, message
            
        except Exception as e:
            message = f"Verification error: {str(e)}"
            logger.error(message)
            return False, 0.0, message
    
    def identify(
        self,
        rgb_image: Union[np.ndarray, str, Path],
        ir_image: Union[np.ndarray, str, Path],
        threshold: Optional[float] = None,
        top_k: int = 5
    ) -> Tuple[Optional[str], float, List[Dict], str]:
        """
        Identify the person from their palm images.
        
        This function searches the database for the best matching identity.
        
        Args:
            rgb_image: RGB image of the palm
            ir_image: Infrared image of the palm
            threshold: Identification threshold
            top_k: Number of top matches to return
            
        Returns:
            (identified_name, similarity, top_matches, message)
        """
        threshold = threshold or self.threshold
        
        try:
            # Extract embedding
            query_embedding = self.extract_embedding(rgb_image, ir_image)
            
            # Search database
            results = self.database.search(
                query_embedding=query_embedding,
                k=top_k,
                threshold=threshold
            )
            
            if results:
                top_match = results[0]
                identified_name = top_match['name']
                similarity = top_match['similarity']
                
                if similarity >= threshold:
                    message = f"Identified as {identified_name} (similarity: {similarity:.4f})"
                else:
                    identified_name = None
                    message = f"No confident match found (best: {results[0]['name']} at {similarity:.4f})"
            else:
                identified_name = None
                similarity = 0.0
                message = "No matches found in database"
            
            logger.info(message)
            
            return identified_name, similarity, results, message
            
        except Exception as e:
            message = f"Identification error: {str(e)}"
            logger.error(message)
            return None, 0.0, [], message
    
    def delete_user(self, name: str) -> Tuple[bool, str]:
        """
        Delete a user from the database.
        
        Args:
            name: User's name to delete
            
        Returns:
            (success, message)
        """
        try:
            count = self.database.delete_by_name(name)
            
            if count > 0:
                if self.database.db_path:
                    self.database.save()
                message = f"Deleted {count} template(s) for {name}"
                logger.info(message)
                return True, message
            else:
                message = f"No templates found for {name}"
                logger.warning(message)
                return False, message
                
        except Exception as e:
            message = f"Deletion error: {str(e)}"
            logger.error(message)
            return False, message
    
    def get_enrolled_users(self) -> List[str]:
        """Get list of all enrolled users."""
        return self.database.get_all_names()
    
    def get_enrollment_count(self) -> int:
        """Get total number of enrollments."""
        return self.database.get_enrollment_count()
    
    def save_database(self, path: Optional[Union[str, Path]] = None):
        """Save the database to disk."""
        self.database.save(path)
        logger.info(f"Database saved to {path or self.database.db_path}")
    
    def load_database(self, path: Union[str, Path]):
        """Load database from disk."""
        self.database.load(path)
        logger.info(f"Database loaded from {path}")
    
    def set_threshold(self, threshold: float):
        """Set the verification threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
        logger.info(f"Threshold set to {threshold}")
    
    def get_system_info(self) -> Dict:
        """Get system information."""
        return {
            'device': self.device,
            'model_loaded': self.model is not None,
            'embedding_dim': self.config.model.embedding_dim,
            'threshold': self.threshold,
            'enrolled_users': len(self.get_enrolled_users()),
            'total_enrollments': self.get_enrollment_count()
        }


# Convenience functions for quick usage

def enroll_user(
    rgb_image_path: str,
    ir_image_path: str,
    name: str,
    model_path: str,
    database_path: str
) -> Tuple[bool, int, str]:
    """
    Quick function to enroll a user.
    
    Args:
        rgb_image_path: Path to RGB image
        ir_image_path: Path to IR image
        name: User's name
        model_path: Path to trained model
        database_path: Path to database
        
    Returns:
        (success, user_id, message)
    """
    system = PalmVeinBiometricSystem(
        model_path=model_path,
        database_path=database_path
    )
    
    return system.enroll(rgb_image_path, ir_image_path, name)


def verify_user(
    rgb_image_path: str,
    ir_image_path: str,
    claimed_name: str,
    model_path: str,
    database_path: str,
    threshold: float = 0.75
) -> Tuple[bool, float, str]:
    """
    Quick function to verify a user.
    
    Args:
        rgb_image_path: Path to RGB image
        ir_image_path: Path to IR image
        claimed_name: Claimed identity
        model_path: Path to trained model
        database_path: Path to database
        threshold: Verification threshold
        
    Returns:
        (is_verified, similarity, message)
    """
    system = PalmVeinBiometricSystem(
        model_path=model_path,
        database_path=database_path
    )
    
    return system.verify(rgb_image_path, ir_image_path, claimed_name, threshold)


def identify_user(
    rgb_image_path: str,
    ir_image_path: str,
    model_path: str,
    database_path: str,
    threshold: float = 0.75
) -> Tuple[Optional[str], float, str]:
    """
    Quick function to identify a user.
    
    Args:
        rgb_image_path: Path to RGB image
        ir_image_path: Path to IR image
        model_path: Path to trained model
        database_path: Path to database
        threshold: Identification threshold
        
    Returns:
        (identified_name, similarity, message)
    """
    system = PalmVeinBiometricSystem(
        model_path=model_path,
        database_path=database_path
    )
    
    name, similarity, results, message = system.identify(
        rgb_image_path, ir_image_path, threshold
    )
    
    return name, similarity, message
