"""
Evaluation script for SAR Oil Spill Detection model.
Computes accuracy, precision, recall, F1-score, and confusion matrix.
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

import config
from model import load_model
from dataset_loader import get_dataloaders


def evaluate():
    """
    Evaluates the trained model on the test set.
    Prints classification metrics and saves confusion matrix.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  SAR Oil Spill Detection - Evaluation")
    print(f"{'='*60}")
    print(f"[Device] Using: {device}")

    # Load model
    model = load_model(config.MODEL_SAVE_PATH, device)

    # Load test data
    _, test_loader = get_dataloaders()

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='binary') * 100
    recall = recall_score(all_labels, all_preds, average='binary') * 100
    f1 = f1_score(all_labels, all_preds, average='binary') * 100

    print(f"\n{'─'*40}")
    print(f"  EVALUATION RESULTS")
    print(f"{'─'*40}")
    print(f"  Accuracy:   {accuracy:.2f}%")
    print(f"  Precision:  {precision:.2f}%")
    print(f"  Recall:     {recall:.2f}%")
    print(f"  F1-Score:   {f1:.2f}%")
    print(f"{'─'*40}\n")

    # Detailed classification report
    print("\n[Classification Report]")
    print(classification_report(all_labels, all_preds,
                                target_names=config.CLASS_NAMES))

    # Confusion matrix
    plot_confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_confusion_matrix(y_true, y_pred):
    """
    Plots and saves a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES,
                annot_kws={'size': 16},
                square=True, linewidths=1, linecolor='white')

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix — SAR Oil Spill Detection', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Confusion matrix saved to: {save_path}")


if __name__ == "__main__":
    evaluate()
