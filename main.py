"""
Main entry point for the SAR Oil Spill Detection Pipeline.
Provides a unified CLI to train, evaluate, and run inference.
"""
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def main():
    parser = argparse.ArgumentParser(
        description="SAR Oil Spill Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train              Train the model (100 epochs)
  python main.py evaluate           Evaluate on test set
  python main.py analyze <path>     Analyze a single SAR image
  python main.py all                Train + Evaluate + Demo analysis
        """
    )

    parser.add_argument('mode', choices=['train', 'evaluate', 'analyze', 'all'],
                        help='Operation mode')
    parser.add_argument('image_path', nargs='?', default=None,
                        help='Path to SAR image (required for analyze mode)')

    args = parser.parse_args()

    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"{'║'} {'SAR Oil Spill Detection Pipeline':^56} {'║'}")
    print(f"{'║'} {'MobileNetV2 + Grad-CAM + Area Estimation':^56} {'║'}")
    print(f"{'╚'+'═'*58+'╝'}\n")

    if args.mode == 'train':
        from train import train
        train()

    elif args.mode == 'evaluate':
        from evaluate import evaluate
        evaluate()

    elif args.mode == 'analyze':
        from inference import analyze_sar
        if args.image_path is None:
            # Use first available image as demo
            demo_images = os.listdir(config.OIL_SPILL_DIR)
            if demo_images:
                args.image_path = os.path.join(config.OIL_SPILL_DIR, demo_images[0])
                print(f"[Info] No image specified. Using demo image: {args.image_path}")
            else:
                print("[Error] No image path provided and no demo images found.")
                sys.exit(1)
        results = analyze_sar(args.image_path)

    elif args.mode == 'all':
        print("[Step 1/3] Training model...")
        from train import train
        model = train()

        print("\n[Step 2/3] Evaluating model...")
        from evaluate import evaluate
        metrics = evaluate()

        print("\n[Step 3/3] Running demo analysis...")
        from inference import analyze_sar
        demo_images = os.listdir(config.OIL_SPILL_DIR)
        if demo_images:
            demo_path = os.path.join(config.OIL_SPILL_DIR, demo_images[0])
            results = analyze_sar(demo_path)

            # Also test a lookalike
            lookalike_images = os.listdir(config.LOOKALIKE_DIR)
            if lookalike_images:
                demo_path2 = os.path.join(config.LOOKALIKE_DIR, lookalike_images[0])
                results2 = analyze_sar(demo_path2)

    print("\n[Done] Pipeline complete! Check the output/ directory for results.")


if __name__ == "__main__":
    main()
