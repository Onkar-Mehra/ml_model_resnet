#!/usr/bin/env python3
"""
Palm Vein Biometric System - Usage Examples
============================================

This script demonstrates how to use the palm vein biometric system
for training, enrollment, verification, and identification.

The system expects:
- RGB images: Regular color photos of palms
- IR images: Infrared photos of palms (shows vein patterns clearly)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def example_organize_dataset():
    """
    Example: How to organize your dataset.
    
    Your dataset should be organized in one of these formats:
    
    Format 1 - Nested (recommended):
    --------------------------------
    data/
        person_001/
            rgb.jpg (or rgb_001.jpg)
            ir.jpg (or ir_001.jpg)
        person_002/
            rgb.jpg
            ir.jpg
        ...
    
    Format 2 - Flat:
    ----------------
    data/
        person_001_rgb.jpg
        person_001_ir.jpg
        person_002_rgb.jpg
        person_002_ir.jpg
        ...
    
    The naming convention must include 'rgb' and 'ir' to identify modalities.
    """
    print("""
    Dataset Organization Example:
    
    For 1000 people with 2 images each (1 RGB, 1 IR):
    
    data/
    ├── person_001/
    │   ├── rgb.jpg
    │   └── ir.jpg
    ├── person_002/
    │   ├── rgb.jpg
    │   └── ir.jpg
    ...
    └── person_1000/
        ├── rgb.jpg
        └── ir.jpg
    """)


def example_training():
    """
    Example: Training the model.
    """
    from config.settings import SystemConfig
    from utils.training import train_model
    
    print("\n" + "="*60)
    print("TRAINING EXAMPLE")
    print("="*60)
    
    # Configuration
    config = SystemConfig()
    
    # Customize training parameters
    config.training.num_epochs = 100
    config.training.batch_size = 32
    config.training.initial_lr = 1e-4
    config.training.loss_type = "arcface"  # Best for biometrics
    config.training.augmentation_strength = "strong"  # For better generalization
    
    # Model configuration
    config.model.backbone = "custom"  # Use custom CNN optimized for vein patterns
    config.model.embedding_dim = 512
    config.model.fusion_type = "attention"  # Cross-modal attention fusion
    
    print(f"""
    Training Configuration:
    - Epochs: {config.training.num_epochs}
    - Batch Size: {config.training.batch_size}
    - Learning Rate: {config.training.initial_lr}
    - Loss: {config.training.loss_type}
    - Backbone: {config.model.backbone}
    - Embedding Dimension: {config.model.embedding_dim}
    - Fusion Type: {config.model.fusion_type}
    
    To train:
    
    python main.py train \\
        --data_dir /path/to/your/data \\
        --save_dir /path/to/save/models \\
        --epochs 100 \\
        --batch_size 32
    
    Or programmatically:
    
    model, history = train_model(
        data_dir='/path/to/your/data',
        save_dir='/path/to/save/models',
        config=config
    )
    """)


def example_enrollment():
    """
    Example: Enrolling a user.
    """
    from api.biometric_system import PalmVeinBiometricSystem
    
    print("\n" + "="*60)
    print("ENROLLMENT EXAMPLE")
    print("="*60)
    
    print("""
    # Initialize the system
    system = PalmVeinBiometricSystem(
        model_path='models/best_model.pth',
        database_path='data/database'
    )
    
    # Enroll a user with their palm images
    success, user_id, message = system.enroll(
        rgb_image='path/to/john_rgb.jpg',
        ir_image='path/to/john_ir.jpg',
        name='John Doe',
        additional_info={
            'department': 'Engineering',
            'employee_id': 'EMP001'
        }
    )
    
    if success:
        print(f"User enrolled successfully with ID: {user_id}")
    else:
        print(f"Enrollment failed: {message}")
    
    # CLI equivalent:
    python main.py enroll \\
        --rgb john_rgb.jpg \\
        --ir john_ir.jpg \\
        --name "John Doe" \\
        --model models/best_model.pth \\
        --database data/database
    """)


def example_verification():
    """
    Example: Verifying a user.
    """
    from api.biometric_system import PalmVeinBiometricSystem
    
    print("\n" + "="*60)
    print("VERIFICATION EXAMPLE")
    print("="*60)
    
    print("""
    # Initialize the system
    system = PalmVeinBiometricSystem(
        model_path='models/best_model.pth',
        database_path='data/database'
    )
    
    # Verify a claimed identity
    is_verified, similarity, message = system.verify(
        rgb_image='path/to/test_rgb.jpg',
        ir_image='path/to/test_ir.jpg',
        claimed_name='John Doe',
        threshold=0.75  # Adjust based on security requirements
    )
    
    if is_verified:
        print(f"Identity verified! Similarity: {similarity:.4f}")
    else:
        print(f"Verification failed. Similarity: {similarity:.4f}")
    
    # CLI equivalent:
    python main.py verify \\
        --rgb test_rgb.jpg \\
        --ir test_ir.jpg \\
        --name "John Doe" \\
        --model models/best_model.pth \\
        --database data/database \\
        --threshold 0.75
    """)


def example_identification():
    """
    Example: Identifying an unknown user.
    """
    from api.biometric_system import PalmVeinBiometricSystem
    
    print("\n" + "="*60)
    print("IDENTIFICATION EXAMPLE")
    print("="*60)
    
    print("""
    # Initialize the system
    system = PalmVeinBiometricSystem(
        model_path='models/best_model.pth',
        database_path='data/database'
    )
    
    # Identify from palm images
    identified_name, similarity, top_matches, message = system.identify(
        rgb_image='path/to/unknown_rgb.jpg',
        ir_image='path/to/unknown_ir.jpg',
        threshold=0.75,
        top_k=5  # Return top 5 matches
    )
    
    if identified_name:
        print(f"Identified as: {identified_name}")
        print(f"Confidence: {similarity:.4f}")
    else:
        print("Could not identify user")
    
    # Show all top matches
    for match in top_matches:
        print(f"  {match['name']}: {match['similarity']:.4f}")
    
    # CLI equivalent:
    python main.py identify \\
        --rgb unknown_rgb.jpg \\
        --ir unknown_ir.jpg \\
        --model models/best_model.pth \\
        --database data/database \\
        --threshold 0.75 \\
        --top_k 5 \\
        --show_top
    """)


def example_batch_enrollment():
    """
    Example: Batch enrolling multiple users.
    """
    print("\n" + "="*60)
    print("BATCH ENROLLMENT EXAMPLE")
    print("="*60)
    
    print("""
    from api.biometric_system import PalmVeinBiometricSystem
    from pathlib import Path
    
    # Initialize the system
    system = PalmVeinBiometricSystem(
        model_path='models/best_model.pth',
        database_path='data/database'
    )
    
    # Directory containing user images
    data_dir = Path('path/to/enrollment_data')
    
    # Enroll all users
    for user_dir in data_dir.iterdir():
        if user_dir.is_dir():
            name = user_dir.name
            
            # Find RGB and IR images
            rgb_image = next(user_dir.glob('*rgb*'), None)
            ir_image = next(user_dir.glob('*ir*'), None)
            
            if rgb_image and ir_image:
                success, user_id, message = system.enroll(
                    rgb_image=str(rgb_image),
                    ir_image=str(ir_image),
                    name=name
                )
                print(f"{name}: {message}")
    
    # Save the database
    system.save_database()
    """)


def example_threshold_tuning():
    """
    Example: Tuning the verification threshold.
    """
    print("\n" + "="*60)
    print("THRESHOLD TUNING EXAMPLE")
    print("="*60)
    
    print("""
    Threshold Guidelines:
    
    The threshold determines the trade-off between:
    - False Acceptance Rate (FAR): Incorrectly accepting imposters
    - False Rejection Rate (FRR): Incorrectly rejecting genuine users
    
    Recommended thresholds:
    ┌────────────────────┬───────────┬────────────────────────┐
    │ Security Level     │ Threshold │ Use Case               │
    ├────────────────────┼───────────┼────────────────────────┤
    │ Low Security       │ 0.60-0.70 │ Convenience access     │
    │ Medium Security    │ 0.70-0.80 │ Office buildings       │
    │ High Security      │ 0.80-0.90 │ Financial institutions │
    │ Very High Security │ 0.90-0.95 │ Government/Military    │
    └────────────────────┴───────────┴────────────────────────┘
    
    To find the optimal threshold:
    1. Test with known genuine pairs -> Calculate FRR at each threshold
    2. Test with known impostor pairs -> Calculate FAR at each threshold
    3. Find Equal Error Rate (EER) or choose based on requirements
    
    system.set_threshold(0.80)  # Update threshold
    """)


def example_environment_handling():
    """
    Example: Handling different environmental conditions.
    """
    print("\n" + "="*60)
    print("ENVIRONMENT HANDLING EXAMPLE")
    print("="*60)
    
    print("""
    The system is designed to handle various environmental conditions:
    
    1. ROTATION HANDLING:
       - The preprocessing pipeline extracts ROI (Region of Interest)
       - Data augmentation during training includes rotation up to ±20°
       - The model learns rotation-invariant features
    
    2. DIFFERENT BACKGROUNDS:
       - ROI extraction isolates the palm from background
       - The model focuses on vein patterns, not background
    
    3. LIGHTING VARIATIONS:
       - CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes lighting
       - Multi-scale enhancement improves contrast
       - IR images are less affected by ambient lighting
    
    4. SCALE VARIATIONS:
       - ROI is resized to a standard size
       - Multi-scale matching is available for verification
    
    5. TILTED HANDS:
       - Strong augmentation includes affine transformations
       - The deep learning model learns to be invariant to tilt
    
    Tips for best results:
    - Use consistent IR lighting during capture
    - Ensure palm fills most of the frame
    - Capture both RGB and IR simultaneously if possible
    - Train with diverse examples including various poses
    """)


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("PALM VEIN BIOMETRIC SYSTEM - USAGE EXAMPLES")
    print("="*60)
    
    example_organize_dataset()
    example_training()
    example_enrollment()
    example_verification()
    example_identification()
    example_batch_enrollment()
    example_threshold_tuning()
    example_environment_handling()
    
    print("\n" + "="*60)
    print("For more information, see the documentation or run:")
    print("  python main.py --help")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
