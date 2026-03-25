"""
Inference module — the main analyze_sar() function.
Takes a single SAR image and returns classification, heatmap, contour, and area.
"""
import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from model import load_model
from dataset_loader import get_transforms
from gradcam import GradCAM, generate_heatmap_image, generate_contour_image


def analyze_sar(image_path):
    """
    Analyzes a single SAR image for oil spill detection.

    Args:
        image_path: Path to the input SAR image (JPG or PNG)

    Returns:
        dict with:
            - class_name:     Predicted class ("Oil Spill" or "Lookalike")
            - confidence:     Confidence score in percentage
            - heatmap_path:   Path to saved Grad-CAM heatmap image
            - contour_path:   Path to saved contour boundary image
            - area_km2:       Estimated area in km²
            - pixel_count:    Number of pixels inside detected contour
    """
    # ─── Setup ───────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  SAR Image Analysis")
    print(f"{'='*60}")
    print(f"[Input] {image_path}")
    print(f"[Device] {device}")

    # Validate input
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ─── Load Model ──────────────────────────────────────────
    model = load_model(config.MODEL_SAVE_PATH, device)
    grad_cam = GradCAM(model)

    # ─── Load & Preprocess Image ─────────────────────────────
    # Load original image for visualization
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Preprocess for model
    pil_image = Image.open(image_path).convert("L")
    transform = get_transforms(is_training=False)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # ─── Classification ──────────────────────────────────────
    cam, predicted_class, confidence = grad_cam.generate(input_tensor)
    class_name = config.CLASS_NAMES[predicted_class]
    confidence_pct = confidence * 100

    print(f"\n[Result] Class: {class_name}")
    print(f"[Result] Confidence: {confidence_pct:.2f}%")

    # ─── Generate Output Images ──────────────────────────────
    # Create output filename base
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1. Heatmap image
    heatmap_overlay, _ = generate_heatmap_image(original_image, cam)
    heatmap_path = os.path.join(config.RESULTS_DIR, f"{base_name}_heatmap.jpg")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original SAR Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(heatmap_overlay)
    axes[1].set_title(f'Grad-CAM Heatmap\n{class_name} ({confidence_pct:.1f}%)',
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.suptitle('SAR Oil Spill Detection — Grad-CAM Analysis',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight', quality=95)
    plt.close()
    print(f"[Saved] Heatmap: {heatmap_path}")

    # 2. Contour image
    contour_image, pixel_count = generate_contour_image(
        original_image, cam, threshold=config.GRADCAM_THRESHOLD)
    contour_path = os.path.join(config.RESULTS_DIR, f"{base_name}_contour.jpg")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original SAR Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(contour_image)
    axes[1].set_title(f'Detected Region Contour\n{class_name}',
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.suptitle('SAR Oil Spill Detection — Contour Boundary',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(contour_path, dpi=150, bbox_inches='tight', quality=95)
    plt.close()
    print(f"[Saved] Contour: {contour_path}")

    # ─── Area Estimation ─────────────────────────────────────
    area_km2 = pixel_count * config.PIXEL_AREA_KM2

    print(f"\n[Area] Pixels inside contour: {pixel_count:,}")
    print(f"[Area] Resolution: {config.PIXEL_RESOLUTION_M}m/pixel (Sentinel-1)")
    print(f"[Area] Estimated area: {area_km2:.4f} km²")

    # ─── Summary ─────────────────────────────────────────────
    results = {
        'class_name': class_name,
        'confidence': round(confidence_pct, 2),
        'heatmap_path': heatmap_path,
        'contour_path': contour_path,
        'area_km2': round(area_km2, 4),
        'pixel_count': pixel_count
    }

    print(f"\n{'='*60}")
    print(f"  Analysis Complete")
    print(f"{'='*60}")
    print(f"  Class:       {results['class_name']}")
    print(f"  Confidence:  {results['confidence']}%")
    print(f"  Area:        {results['area_km2']} km²")
    print(f"  Heatmap:     {results['heatmap_path']}")
    print(f"  Contour:     {results['contour_path']}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Default: pick first image from Oil_spill folder
        test_images = os.listdir(config.OIL_SPILL_DIR)
        if test_images:
            test_path = os.path.join(config.OIL_SPILL_DIR, test_images[0])
            print(f"[Demo] No image path provided. Using: {test_path}")
            results = analyze_sar(test_path)
        else:
            print("Usage: python inference.py <image_path>")
            print("  or place images in dataset/Oil_spill/")
    else:
        results = analyze_sar(sys.argv[1])
