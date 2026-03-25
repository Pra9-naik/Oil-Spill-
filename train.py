"""
Training script for SAR Oil Spill Detection model.
Includes training loop, validation, best model saving, and loss/accuracy plotting.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from model import create_model
from dataset_loader import get_dataloaders


def train():
    """
    Main training function.
    Trains MobileNetV2 for binary classification on SAR images.
    Saves the best model based on validation accuracy.
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  SAR Oil Spill Detection - Training Pipeline")
    print(f"{'='*60}")
    print(f"[Device] Using: {device}")

    if device.type == 'cuda':
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    print(f"\n[Data] Loading dataset...")
    train_loader, test_loader = get_dataloaders()

    # Model
    print(f"\n[Model] Building MobileNetV2...")
    model = create_model(pretrained=True, freeze_features=True)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      patience=10, factor=0.5)

    # Training tracking
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\n[Training] Starting {config.NUM_EPOCHS} epochs...")
    print(f"[Training] LR: {config.LEARNING_RATE}, Batch Size: {config.BATCH_SIZE}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # ============ TRAINING PHASE ============
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{config.NUM_EPOCHS}",
                     ncols=100, leave=False)

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.1f}%'
            })

        train_loss = running_loss / total
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ============ VALIDATION PHASE ============
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # LR Scheduler step
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

        # Print epoch summary every 5 epochs or on improvement
        if epoch % 5 == 0 or val_acc >= best_val_acc:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
                  f"Best: {best_val_acc:.1f}% (Ep {best_epoch}) | "
                  f"Time: {elapsed:.0f}s")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Total Time: {total_time/60:.1f} minutes")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"{'='*60}\n")

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    return model


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Plots and saves training/validation loss and accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', linewidth=1.5, label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=1.5, label='Validation Loss')
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', linewidth=1.5, label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', linewidth=1.5, label='Validation Accuracy')
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Training curves saved to: {save_path}")


if __name__ == "__main__":
    train()
