"""
MobileNetV2 model with transfer learning for binary SAR classification.
Frozen feature extractor with custom classifier head.
"""
import torch
import torch.nn as nn
from torchvision import models
import config


def create_model(pretrained=True, freeze_features=True):
    """
    Creates a MobileNetV2 model adapted for binary SAR classification.

    Args:
        pretrained: Whether to load ImageNet pretrained weights
        freeze_features: Whether to freeze the feature extractor layers

    Returns:
        model: Modified MobileNetV2 model
    """
    # Load MobileNetV2 with pretrained weights
    if pretrained:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2(weights=None)

    # Freeze feature extractor if specified
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
        print("[Model] Feature extractor layers frozen")

    # Replace classifier head for binary classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, config.NUM_CLASSES)
    )

    print(f"[Model] MobileNetV2 created with {config.NUM_CLASSES} output classes")
    print(f"[Model] Classifier: {model.classifier}")

    return model


def load_model(model_path, device='cpu'):
    """
    Loads a saved model from disk.

    Args:
        model_path: Path to the saved .pth file
        device: Device to load the model on

    Returns:
        model: Loaded model in eval mode
    """
    model = create_model(pretrained=False, freeze_features=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"[Model] Loaded from {model_path}")
    return model
