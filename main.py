#!/usr/bin/env python3
"""
Palm Vein Biometric System - Main Entry Point
==============================================

This is the main entry point for the palm vein biometric system.
It provides CLI commands for:
- Training the model
- Enrolling users
- Verifying users
- Identifying users

Usage:
    python main.py train --data_dir ./data --save_dir ./models
    python main.py enroll --rgb ./rgb.jpg --ir ./ir.jpg --name "John Doe"
    python main.py verify --rgb ./rgb.jpg --ir ./ir.jpg --name "John Doe"
    python main.py identify --rgb ./rgb.jpg --ir ./ir.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SystemConfig, DEFAULT_CONFIG
from api.biometric_system import PalmVeinBiometricSystem
from utils.training import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_command(args):
    """Train the model."""
    logger.info("Starting model training...")
    
    config = DEFAULT_CONFIG
    
    # Override config with command line arguments
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.initial_lr = args.lr
    
    model, history = train_model(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        config=config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {min(history['val_loss']):.4f}")
    logger.info(f"Best validation accuracy: {max(history['val_acc']):.4f}")


def enroll_command(args):
    """Enroll a new user."""
    logger.info(f"Enrolling user: {args.name}")
    
    system = PalmVeinBiometricSystem(
        model_path=args.model,
        database_path=args.database
    )
    
    success, user_id, message = system.enroll(
        rgb_image=args.rgb,
        ir_image=args.ir,
        name=args.name
    )
    
    if success:
        logger.info(f"✓ {message}")
    else:
        logger.error(f"✗ {message}")
        sys.exit(1)


def verify_command(args):
    """Verify a user."""
    logger.info(f"Verifying user: {args.name}")
    
    system = PalmVeinBiometricSystem(
        model_path=args.model,
        database_path=args.database
    )
    
    is_verified, similarity, message = system.verify(
        rgb_image=args.rgb,
        ir_image=args.ir,
        claimed_name=args.name,
        threshold=args.threshold
    )
    
    if is_verified:
        logger.info(f"✓ VERIFIED: {message}")
        print(f"\n{'='*50}")
        print(f"  VERIFICATION RESULT: SUCCESS")
        print(f"  User: {args.name}")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Threshold: {args.threshold}")
        print(f"{'='*50}\n")
    else:
        logger.warning(f"✗ NOT VERIFIED: {message}")
        print(f"\n{'='*50}")
        print(f"  VERIFICATION RESULT: FAILED")
        print(f"  User: {args.name}")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Threshold: {args.threshold}")
        print(f"{'='*50}\n")
        sys.exit(1)


def identify_command(args):
    """Identify a user."""
    logger.info("Identifying user from palm images...")
    
    system = PalmVeinBiometricSystem(
        model_path=args.model,
        database_path=args.database
    )
    
    identified_name, similarity, results, message = system.identify(
        rgb_image=args.rgb,
        ir_image=args.ir,
        threshold=args.threshold,
        top_k=args.top_k
    )
    
    print(f"\n{'='*50}")
    print(f"  IDENTIFICATION RESULTS")
    print(f"{'='*50}")
    
    if identified_name:
        print(f"  ✓ Identified as: {identified_name}")
        print(f"  Similarity: {similarity:.4f}")
    else:
        print(f"  ✗ Unable to identify")
        print(f"  Best match similarity: {similarity:.4f}")
    
    if results and args.show_top:
        print(f"\n  Top {len(results)} matches:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. {result['name']}: {result['similarity']:.4f}")
    
    print(f"{'='*50}\n")


def list_users_command(args):
    """List all enrolled users."""
    system = PalmVeinBiometricSystem(
        model_path=args.model,
        database_path=args.database
    )
    
    users = system.get_enrolled_users()
    count = system.get_enrollment_count()
    
    print(f"\n{'='*50}")
    print(f"  ENROLLED USERS ({count} total enrollments)")
    print(f"{'='*50}")
    
    for user in sorted(users):
        print(f"  - {user}")
    
    print(f"{'='*50}\n")


def delete_command(args):
    """Delete a user."""
    logger.info(f"Deleting user: {args.name}")
    
    system = PalmVeinBiometricSystem(
        model_path=args.model,
        database_path=args.database
    )
    
    success, message = system.delete_user(args.name)
    
    if success:
        logger.info(f"✓ {message}")
    else:
        logger.error(f"✗ {message}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Palm Vein Biometric System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_dir', type=str, required=True,
                             help='Path to training data directory')
    train_parser.add_argument('--save_dir', type=str, required=True,
                             help='Path to save model checkpoints')
    train_parser.add_argument('--epochs', type=int, default=None,
                             help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=None,
                             help='Batch size')
    train_parser.add_argument('--lr', type=float, default=None,
                             help='Initial learning rate')
    train_parser.add_argument('--device', type=str, default=None,
                             help='Device to use (cuda/cpu)')
    
    # Enroll command
    enroll_parser = subparsers.add_parser('enroll', help='Enroll a new user')
    enroll_parser.add_argument('--rgb', type=str, required=True,
                              help='Path to RGB palm image')
    enroll_parser.add_argument('--ir', type=str, required=True,
                              help='Path to IR palm image')
    enroll_parser.add_argument('--name', type=str, required=True,
                              help='User name/identifier')
    enroll_parser.add_argument('--model', type=str, required=True,
                              help='Path to trained model')
    enroll_parser.add_argument('--database', type=str, required=True,
                              help='Path to database directory')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify a user')
    verify_parser.add_argument('--rgb', type=str, required=True,
                              help='Path to RGB palm image')
    verify_parser.add_argument('--ir', type=str, required=True,
                              help='Path to IR palm image')
    verify_parser.add_argument('--name', type=str, required=True,
                              help='Claimed user name')
    verify_parser.add_argument('--model', type=str, required=True,
                              help='Path to trained model')
    verify_parser.add_argument('--database', type=str, required=True,
                              help='Path to database directory')
    verify_parser.add_argument('--threshold', type=float, default=0.75,
                              help='Verification threshold')
    
    # Identify command
    identify_parser = subparsers.add_parser('identify', help='Identify a user')
    identify_parser.add_argument('--rgb', type=str, required=True,
                                help='Path to RGB palm image')
    identify_parser.add_argument('--ir', type=str, required=True,
                                help='Path to IR palm image')
    identify_parser.add_argument('--model', type=str, required=True,
                                help='Path to trained model')
    identify_parser.add_argument('--database', type=str, required=True,
                                help='Path to database directory')
    identify_parser.add_argument('--threshold', type=float, default=0.75,
                                help='Identification threshold')
    identify_parser.add_argument('--top_k', type=int, default=5,
                                help='Number of top matches to return')
    identify_parser.add_argument('--show_top', action='store_true',
                                help='Show top matches')
    
    # List users command
    list_parser = subparsers.add_parser('list', help='List enrolled users')
    list_parser.add_argument('--model', type=str, required=True,
                            help='Path to trained model')
    list_parser.add_argument('--database', type=str, required=True,
                            help='Path to database directory')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a user')
    delete_parser.add_argument('--name', type=str, required=True,
                              help='User name to delete')
    delete_parser.add_argument('--model', type=str, required=True,
                              help='Path to trained model')
    delete_parser.add_argument('--database', type=str, required=True,
                              help='Path to database directory')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    commands = {
        'train': train_command,
        'enroll': enroll_command,
        'verify': verify_command,
        'identify': identify_command,
        'list': list_users_command,
        'delete': delete_command
    }
    
    commands[args.command](args)


if __name__ == '__main__':
    main()
