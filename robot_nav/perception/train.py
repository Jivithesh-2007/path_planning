"""
Fast CPU training script for PyTorch TerrainCNN (<3 min runtime).
Generates synthetic data, trains network, prints confusion matrix, and saves model weights.
"""

import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from robot_nav.perception.dataset_gen import generate_synthetic_patch_dataset
from robot_nav.perception.cell_classifier import TerrainCNN


def train_terrain_classifier(
    save_path: Path | str = "robot_nav/perception/weights/cnn.pth",
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 0.003,
) -> TerrainCNN:
    """
    Generate synthetic data and train TerrainCNN on CPU.

    :param save_path: Path to save trained PyTorch state dict.
    :param epochs: Number of training epochs.
    :param batch_size: Mini-batch size.
    :param lr: Learning rate for Adam optimizer.
    :return: Trained TerrainCNN model.
    """
    start_time = time.time()
    print("Generating synthetic patch dataset...")
    X, y = generate_synthetic_patch_dataset(samples_per_class=400, patch_size=32)

    # 80/20 train/test split
    split_idx = int(0.8 * len(y))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = TerrainCNN(num_classes=3)
    print(f"TerrainCNN parameter count: {model.count_parameters():,}")
    assert model.count_parameters() < 100_000, "Model parameter count exceeds 100k constraint!"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting CPU training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        train_acc = correct / total
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/total:.4f}, Train Acc: {train_acc*100:.2f}%")

    # Evaluation on Test Set
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(batch_y.numpy())

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)

    test_acc = np.mean(all_preds_arr == all_targets_arr)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    # Confusion matrix
    conf_matrix = np.zeros((3, 3), dtype=np.int32)
    for t, p in zip(all_targets_arr, all_preds_arr):
        conf_matrix[t, p] += 1

    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print("           Free  Obstacle  Rough")
    classes = ["Free    ", "Obstacle", "Rough   "]
    for idx, row in enumerate(conf_matrix):
        print(f"{classes[idx]}  {row[0]:6d}    {row[1]:6d} {row[2]:6d}")

    # Save weights
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path_obj)
    print(f"\nSaved model weights to {save_path_obj}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds.")

    return model


if __name__ == "__main__":
    train_terrain_classifier()
