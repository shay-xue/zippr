"""
Train a neural decoder model from SciFi training data.

Takes HDF5 recordings with neural + label channels, trains a decoder model,
and exports to ONNX for on-device deployment.

Usage:
    python scripts/train_decoder.py --data data/recordings/easy_*.h5 --output models/decoder.onnx
    python scripts/train_decoder.py --data data/recordings/easy_*.h5 --model cnn --output models/decoder.onnx
"""

import argparse
import glob
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ─── Dataset ─────────────────────────────────────────────────────────────────

class NeuralDataset(Dataset):
    """Dataset of (neural_window, action_label) pairs."""

    def __init__(self, windows, labels):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


# ─── Models ──────────────────────────────────────────────────────────────────

class MLPDecoder(nn.Module):
    """Simple MLP decoder: flatten neural window → action class."""

    def __init__(self, n_channels, window_size, n_classes):
        super().__init__()
        input_dim = n_channels * window_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNNDecoder(nn.Module):
    """1D CNN decoder over temporal neural data."""

    def __init__(self, n_channels, window_size, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv(x))


class LinearDecoder(nn.Module):
    """Simplest possible linear decoder (baseline)."""

    def __init__(self, n_channels, window_size, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * window_size, n_classes),
        )

    def forward(self, x):
        return self.net(x)


MODEL_REGISTRY = {
    "mlp": MLPDecoder,
    "cnn": CNNDecoder,
    "linear": LinearDecoder,
}


# ─── Data Preprocessing ─────────────────────────────────────────────────────

def discretize_joystick(labels, n_directions=4):
    """
    Convert continuous joystick (X, Y) labels into discrete direction classes.

    For easy mode with 2 label channels (X, Y):
      - 4 directions: up, down, left, right + center = 5 classes
      - 8 directions: N, NE, E, SE, S, SW, W, NW + center = 9 classes

    Returns integer class labels.
    """
    x = labels[0]  # joystick X
    y = labels[1]  # joystick Y

    # Normalize to [-1, 1] if not already
    x_range = np.max(np.abs(x))
    y_range = np.max(np.abs(y))
    if x_range > 0:
        x = x / x_range
    if y_range > 0:
        y = y / y_range

    threshold = 0.3  # dead zone threshold

    if n_directions == 4:
        # 5 classes: center(0), up(1), right(2), down(3), left(4)
        classes = np.zeros(len(x), dtype=np.int64)
        magnitude = np.sqrt(x**2 + y**2)
        active = magnitude > threshold

        angles = np.arctan2(y, x)  # radians, [-pi, pi]

        # Map angles to 4 directions
        classes[active & (angles >= -np.pi/4) & (angles < np.pi/4)] = 2    # right
        classes[active & (angles >= np.pi/4) & (angles < 3*np.pi/4)] = 1   # up
        classes[active & ((angles >= 3*np.pi/4) | (angles < -3*np.pi/4))] = 4  # left
        classes[active & (angles >= -3*np.pi/4) & (angles < -np.pi/4)] = 3  # down

        class_names = ["center", "up", "right", "down", "left"]

    elif n_directions == 8:
        # 9 classes: center(0), N(1), NE(2), E(3), SE(4), S(5), SW(6), W(7), NW(8)
        classes = np.zeros(len(x), dtype=np.int64)
        magnitude = np.sqrt(x**2 + y**2)
        active = magnitude > threshold
        angles = np.arctan2(y, x)

        boundaries = np.linspace(-np.pi, np.pi, 9)
        for i in range(8):
            mask = active & (angles >= boundaries[i]) & (angles < boundaries[i+1])
            classes[mask] = [7, 6, 5, 4, 3, 2, 1, 8][i]  # map to direction

        class_names = ["center", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    return classes, class_names


def create_windows(neural, labels_discrete, window_ms=50, sample_rate=32000, stride_ms=25):
    """
    Segment continuous neural data into overlapping windows with labels.

    Args:
        neural: (n_channels, n_samples)
        labels_discrete: (n_samples,) integer class labels
        window_ms: window duration in milliseconds
        sample_rate: samples per second
        stride_ms: stride between windows in ms

    Returns:
        windows: (n_windows, n_channels, window_samples)
        window_labels: (n_windows,) majority label per window
    """
    window_samples = int(window_ms * sample_rate / 1000)
    stride_samples = int(stride_ms * sample_rate / 1000)

    n_channels, n_total = neural.shape
    n_windows = (n_total - window_samples) // stride_samples + 1

    windows = np.zeros((n_windows, n_channels, window_samples), dtype=np.float32)
    window_labels = np.zeros(n_windows, dtype=np.int64)

    for i in range(n_windows):
        start = i * stride_samples
        end = start + window_samples
        windows[i] = neural[:, start:end]

        # Majority vote for window label
        segment_labels = labels_discrete[start:end]
        window_labels[i] = np.bincount(segment_labels).argmax()

    return windows, window_labels


def load_and_preprocess(file_paths, n_directions=4, window_ms=50, stride_ms=25):
    """Load all recording files and create windowed dataset."""
    all_windows = []
    all_labels = []

    for fpath in file_paths:
        print(f"Loading {fpath}...")
        with h5py.File(fpath, "r") as f:
            neural = f["neural"][:]
            labels_raw = f["labels"][:]
            sample_rate = f.attrs["sample_rate"]
            mode = f.attrs["mode"]

        print(f"  {neural.shape[0]} neural ch, {labels_raw.shape[0]} label ch, "
              f"{neural.shape[1]} samples")

        # Normalize neural data per channel (z-score)
        mean = neural.mean(axis=1, keepdims=True)
        std = neural.std(axis=1, keepdims=True) + 1e-8
        neural = (neural - mean) / std

        # Discretize labels
        labels_disc, class_names = discretize_joystick(labels_raw, n_directions)

        # Create windows
        windows, wlabels = create_windows(
            neural, labels_disc, window_ms, sample_rate, stride_ms
        )
        all_windows.append(windows)
        all_labels.append(wlabels)

        print(f"  → {len(wlabels)} windows, class distribution: "
              f"{dict(zip(*np.unique(wlabels, return_counts=True)))}")

    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return windows, labels, class_names


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, n_epochs=50, lr=1e-3, device="cpu"):
    """Train the decoder model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            train_total += len(batch_y)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                val_total += len(batch_y)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss / val_total)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f} | "
                  f"Best: {best_val_acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_val_acc


def export_onnx(model, n_channels, window_size, output_path):
    """Export trained model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, n_channels, window_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["neural_input"],
        output_names=["action_logits"],
        dynamic_axes={
            "neural_input": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
        opset_version=13,
    )
    print(f"ONNX model exported to {output_path}")

    # Verify with ONNX Runtime
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    test_input = dummy_input.numpy()
    result = session.run(None, {"neural_input": test_input})
    print(f"ONNX verification: output shape = {result[0].shape}")


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train neural decoder")
    parser.add_argument("--data", nargs="+", required=True,
                        help="HDF5 recording file(s) or glob pattern")
    parser.add_argument("--output", "-o", default="models/decoder.onnx",
                        help="Output ONNX model path")
    parser.add_argument("--model", choices=["mlp", "cnn", "linear"], default="cnn",
                        help="Model architecture")
    parser.add_argument("--directions", type=int, default=4, choices=[4, 8],
                        help="Number of joystick directions")
    parser.add_argument("--window-ms", type=int, default=50,
                        help="Window size in milliseconds")
    parser.add_argument("--stride-ms", type=int, default=25,
                        help="Stride in milliseconds")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve file paths (handle globs)
    file_paths = []
    for pattern in args.data:
        file_paths.extend(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"No files found matching: {args.data}")
    print(f"Found {len(file_paths)} recording file(s)")

    # Load and preprocess
    n_classes = args.directions + 1  # directions + center
    windows, labels, class_names = load_and_preprocess(
        file_paths, args.directions, args.window_ms, args.stride_ms
    )
    print(f"\nTotal dataset: {len(labels)} windows, {n_classes} classes")
    print(f"Window shape: {windows.shape[1:]} (channels x samples)")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        windows, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_ds = NeuralDataset(X_train, y_train)
    val_ds = NeuralDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Build model
    n_channels = windows.shape[1]
    window_size = windows.shape[2]
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(n_channels, window_size, n_classes)
    print(f"\nModel: {args.model} | Params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on: {device}\n")
    model, best_acc = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)

    # Evaluate
    model.eval()
    model = model.cpu()
    with torch.no_grad():
        val_preds = []
        val_true = []
        for bx, by in val_loader:
            preds = model(bx).argmax(dim=1)
            val_preds.extend(preds.numpy())
            val_true.extend(by.numpy())

    print(f"\nFinal validation accuracy: {best_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(val_true, val_preds, target_names=class_names))

    # Export
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_onnx(model, n_channels, window_size, args.output)


if __name__ == "__main__":
    main()
