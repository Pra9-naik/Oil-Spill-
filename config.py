"""
Configuration settings for SAR Oil Spill Detection Pipeline.
All hyperparameters and paths are centralized here.
"""
import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OIL_SPILL_DIR = os.path.join(DATASET_DIR, "Oil_spill")
LOOKALIKE_DIR = os.path.join(DATASET_DIR, "Look_alikes")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "oil_spill_model.pth")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# MODEL & TRAINING HYPERPARAMETERS
# ============================================================
IMAGE_SIZE = 224               # MobileNetV2 input size
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
TRAIN_SPLIT = 0.8              # 80/20 train-test split
NUM_CLASSES = 2
RANDOM_SEED = 42

# ============================================================
# CLASS NAMES
# ============================================================
CLASS_NAMES = ["Lookalike", "Oil Spill"]

# ============================================================
# SENTINEL-1 SAR RESOLUTION
# ============================================================
PIXEL_RESOLUTION_M = 10        # 10 meters per pixel (Sentinel-1)
PIXEL_AREA_KM2 = (PIXEL_RESOLUTION_M ** 2) / 1e6  # Area per pixel in km²

# ============================================================
# GRAD-CAM SETTINGS
# ============================================================
GRADCAM_THRESHOLD = 0.5        # Threshold for contour extraction
GRADCAM_TARGET_LAYER = "features.18.0"  # Last conv layer of MobileNetV2

# ============================================================
# DATA AUGMENTATION SETTINGS
# ============================================================
SPECKLE_NOISE_STD = 0.1        # Standard deviation for speckle noise
