"""
Custom Dataset and Data Augmentation for SAR Oil Spill Detection.
Includes speckle noise augmentation specific to SAR imagery.
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import config


class SpeckleNoise:
    """
    Adds multiplicative speckle noise to simulate SAR-specific noise.
    Speckle noise: output = image + image * noise
    """
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + tensor * noise, 0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"


class SARDataset(Dataset):
    """
    Custom PyTorch Dataset for SAR images.
    Labels: 0 = Lookalike, 1 = Oil Spill
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image as grayscale
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_image_paths_and_labels():
    """
    Scans the dataset directory and returns lists of image paths and labels.
    """
    image_paths = []
    labels = []

    # Oil Spill images (label = 1)
    if os.path.exists(config.OIL_SPILL_DIR):
        for fname in os.listdir(config.OIL_SPILL_DIR):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(config.OIL_SPILL_DIR, fname))
                labels.append(1)

    # Lookalike images (label = 0)
    if os.path.exists(config.LOOKALIKE_DIR):
        for fname in os.listdir(config.LOOKALIKE_DIR):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(config.LOOKALIKE_DIR, fname))
                labels.append(0)

    print(f"[Dataset] Found {labels.count(1)} Oil Spill images")
    print(f"[Dataset] Found {labels.count(0)} Lookalike images")
    print(f"[Dataset] Total: {len(image_paths)} images")

    return image_paths, labels


def get_transforms(is_training=True):
    """
    Returns data transforms with SAR-specific augmentations for training.
    Grayscale images are replicated to 3 channels for MobileNetV2.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            # Replicate grayscale to 3 channels for MobileNetV2
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            SpeckleNoise(std=config.SPECKLE_NOISE_STD),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_dataloaders():
    """
    Creates train and test DataLoaders with 80/20 split.
    """
    image_paths, labels = get_image_paths_and_labels()

    # Create full dataset with training transforms initially
    full_dataset_train = SARDataset(image_paths, labels, transform=get_transforms(is_training=True))
    full_dataset_test = SARDataset(image_paths, labels, transform=get_transforms(is_training=False))

    # Calculate split sizes
    total_size = len(full_dataset_train)
    train_size = int(config.TRAIN_SPLIT * total_size)
    test_size = total_size - train_size

    # Split with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_indices, test_indices = random_split(
        range(total_size), [train_size, test_size], generator=generator
    )

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset_train, train_indices.indices)
    test_dataset = torch.utils.data.Subset(full_dataset_test, test_indices.indices)

    print(f"[Split] Training samples: {len(train_dataset)}")
    print(f"[Split] Testing samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader
