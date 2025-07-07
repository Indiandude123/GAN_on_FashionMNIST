import argparse
import os
import sys

# Add the project root to the Python path to allow importing modules
# This assumes main.py is run from the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import the main functions from your training and generation scripts
from train import run_training
from generate import run_generation

def main():
    """
    Main entry point for the GAN project.
    Uses argparse to handle different commands (train, generate).
    """
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST GAN Training and Generation CLI.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Train command ---
    train_parser = subparsers.add_parser(
        "train",
        help="Train the GAN model.",
        description="""
        Train the Generative Adversarial Network (GAN) on the Fashion-MNIST dataset.
        Example:
            python main.py train --epochs 200 --mode one_one
            python main.py train --epochs 150 --mode five_gen_one_disc
        """
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs. Default: 200"
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training. Default: 128"
    )
    train_parser.add_argument(
        "--mode",
        type=str,
        default="one_one",
        choices=["one_one", "five_gen_one_disc", "five_disc_one_gen"],
        help="Training mode:\n"
             "  one_one: 1 Generator step, 1 Discriminator step (default)\n"
             "  five_gen_one_disc: 5 Generator steps, 1 Discriminator step\n"
             "  five_disc_one_gen: 1 Generator step, 5 Discriminator steps"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=0.000086,
        help="Learning rate for Adam optimizer. Default: 0.000086"
    )
    train_parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.00001,
        help="Weight decay for Adam optimizer. Default: 0.00001"
    )

    # --- Generate command ---
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate samples from a trained GAN.",
        description="""
        Generate and display images using a trained Generator model.
        Example:
            python main.py generate
            python main.py generate --model_path checkpoints/generator_epoch_100.pth
        """
    )
    generate_parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("checkpoints", "generator_final.pth"),
        help="Path to the trained generator model (.pth file). "
             "Default: checkpoints/generator_final.pth"
    )
    generate_parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to generate and display. Default: 16"
    )

    args = parser.parse_args()

    if args.command == "train":
        print(f"Starting GAN training for {args.epochs} epochs in '{args.mode}' mode...")
        run_training(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            mode=args.mode,
            learning_rate=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.command == "generate":
        print(f"Generating {args.num_samples} samples using model: {args.model_path}...")
        run_generation(
            model_path=args.model_path,
            num_samples=args.num_samples
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

