"""
Grad-CAM implementation for MobileNetV2.
Generates heatmaps showing which regions the model focused on.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for MobileNetV2.
    Hooks into the last convolutional layer to capture gradients and activations.
    """

    def __init__(self, model, target_layer_name="features"):
        """
        Args:
            model: The trained MobileNetV2 model
            target_layer_name: Name of the target layer for Grad-CAM
        """
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Hook into the last convolutional block of MobileNetV2 features
        # MobileNetV2.features[-1] is the last InvertedResidual + final conv
        target_layer = model.features[-1]

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Captures the forward activations."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Captures the backward gradients."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generates a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed input image tensor (1, 3, H, W)
            target_class: Class index to generate heatmap for (None = predicted class)

        Returns:
            heatmap: Normalized Grad-CAM heatmap (H, W) in range [0, 1]
            predicted_class: The predicted class index
            confidence: Confidence score for the predicted class
        """
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass for the target class
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        # Compute Grad-CAM
        # Global average pool the gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Apply ReLU

        # Resize to input size
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
                            mode='bilinear', align_corners=False)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, target_class, confidence


def generate_heatmap_image(original_image, cam, colormap=cv2.COLORMAP_JET):
    """
    Overlays the Grad-CAM heatmap on the original image.

    Args:
        original_image: Original image as numpy array (H, W) or (H, W, 3)
        cam: Grad-CAM heatmap normalized to [0, 1]
        colormap: OpenCV colormap to use

    Returns:
        heatmap_overlay: Colored heatmap overlaid on original image
        heatmap_only: The colored heatmap alone
    """
    # Resize CAM to match original image
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Create colored heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert original to 3-channel if grayscale
    if len(original_image.shape) == 2:
        original_3ch = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        original_3ch = original_image.copy()

    # Overlay heatmap on original
    heatmap_overlay = cv2.addWeighted(original_3ch, 0.6, heatmap, 0.4, 0)

    return heatmap_overlay, heatmap


def generate_contour_image(original_image, cam, threshold=0.5):
    """
    Draws red contour boundaries on the original image based on Grad-CAM thresholding.

    Args:
        original_image: Original image as numpy array (H, W) or (H, W, 3)
        cam: Grad-CAM heatmap normalized to [0, 1]
        threshold: Threshold value for contour extraction

    Returns:
        contour_image: Original image with red contour boundaries
        pixel_count: Number of pixels inside the contour
    """
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Threshold the Grad-CAM heatmap
    binary_mask = (cam_resized >= threshold).astype(np.uint8) * 255

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Convert original to 3-channel if grayscale
    if len(original_image.shape) == 2:
        contour_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        contour_image = original_image.copy()

    # Draw red contours
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)

    # Count pixels inside contours
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, contours, -1, 255, -1)  # Filled contour
    pixel_count = np.count_nonzero(contour_mask)

    return contour_image, pixel_count
