"""
Neural Decoder Training — Hard Mode (64 neural → 12 controller outputs)

Phase 1 findings:
- Raw voltage correlations ≈ 0 → useless as features
- Spike rate features show moderate correlation (max r=0.27)
- Linear decoder fails → need nonlinear model
- Approach: spike detection → temporal binning → MLP/CNN

Pipeline:
1. Load all training HDF5 files
2. Extract spike-rate features (threshold crossings in time bins)
3. Train MLP and 1D-CNN decoders
4. Evaluate on held-out data
5. Export best model to ONNX
"""

import h5py
import numpy as np
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Config ──────────────────────────────────────────────────────────────────

SAMPLE_RATE = 32000
N_NEURAL = 64
N_LABELS = 12
BIN_SIZE_MS = 10  # temporal bin size in ms
BIN_SIZE_SAMPLES = int(SAMPLE_RATE * BIN_SIZE_MS / 1000)  # 320 samples per bin
SPIKE_THRESHOLDS = [3.0, 4.0, 5.0]  # multiples of std for spike detection
WINDOW_BINS = 10  # number of bins per input window (100ms at 10ms bins)
STRIDE_BINS = 1   # stride for sliding window

# Joystick discretization
JOYSTICK_DEAD_ZONE = 5000
JOYSTICK_LABELS = ['center', 'up', 'down', 'left', 'right']

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ── Data Loading ────────────────────────────────────────────────────────────

def load_all_training_data(data_dir='data/recordings'):
    """Load and concatenate all training HDF5 files."""
    files = sorted(glob.glob(os.path.join(data_dir, 'train_*/broadband_data_*.h5')))
    print(f"Found {len(files)} training files")

    all_neural = []
    all_labels = []

    for f in files:
        trial = f.split('/')[-2]
        with h5py.File(f, 'r') as hf:
            raw = hf['acquisition/ElectricalSeries'][:]
            n_ch = len(hf['general/extracellular_ephys/electrodes/id'][:])
            n_samples = len(raw) // n_ch
            data = raw[:n_samples * n_ch].reshape(n_samples, n_ch)

            neural = data[:, :N_NEURAL].astype(np.float32)
            labels = data[:, N_NEURAL:N_NEURAL + N_LABELS].astype(np.float32)

            all_neural.append(neural)
            all_labels.append(labels)
            print(f"  {trial}: {n_samples} samples ({n_samples/SAMPLE_RATE:.1f}s)")

    neural = np.concatenate(all_neural, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"\nTotal: {neural.shape[0]} samples ({neural.shape[0]/SAMPLE_RATE:.1f}s)")
    return neural, labels


# ── Feature Extraction ──────────────────────────────────────────────────────

def extract_spike_features(neural, thresholds=SPIKE_THRESHOLDS):
    """Extract spike rate features using multiple thresholds."""
    n_samples, n_ch = neural.shape
    n_bins = n_samples // BIN_SIZE_SAMPLES

    # Compute per-channel statistics for thresholding
    ch_mean = neural.mean(axis=0)
    ch_std = neural.std(axis=0)

    features_list = []
    for thresh_mult in thresholds:
        thresh_pos = ch_mean + thresh_mult * ch_std
        thresh_neg = ch_mean - thresh_mult * ch_std

        # Detect threshold crossings
        above = neural > thresh_pos[None, :]
        below = neural < thresh_neg[None, :]
        crossings = (above | below).astype(np.float32)

        # Bin into time windows
        trimmed = crossings[:n_bins * BIN_SIZE_SAMPLES]
        binned = trimmed.reshape(n_bins, BIN_SIZE_SAMPLES, n_ch).sum(axis=1)
        features_list.append(binned)

    # Stack all threshold features: shape = (n_bins, n_ch * n_thresholds)
    features = np.concatenate(features_list, axis=1)
    print(f"Spike features: {features.shape} (bins × features)")
    return features, n_bins


def bin_labels(labels, n_bins):
    """Bin labels to match spike rate time resolution."""
    trimmed = labels[:n_bins * BIN_SIZE_SAMPLES]
    binned = trimmed.reshape(n_bins, BIN_SIZE_SAMPLES, N_LABELS).mean(axis=1)
    return binned


def create_windows(features, labels, window_size=WINDOW_BINS, stride=STRIDE_BINS):
    """Create sliding windows of features with corresponding labels."""
    n_bins, n_feat = features.shape
    n_windows = (n_bins - window_size) // stride + 1

    X = np.zeros((n_windows, window_size, n_feat), dtype=np.float32)
    Y = np.zeros((n_windows, N_LABELS), dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        X[i] = features[start:start + window_size]
        Y[i] = labels[start + window_size - 1]  # predict label at end of window

    print(f"Windows: {X.shape[0]} × {X.shape[1]} bins × {X.shape[2]} features")
    return X, Y


# ── Label Processing ────────────────────────────────────────────────────────

def process_labels(Y):
    """
    Process labels into regression targets (joysticks) and classification targets (buttons).

    Joystick channels (0-3): normalize to [-1, 1]
    Button channels (4-11): binarize to 0/1
    """
    Y_joystick = Y[:, :4] / 32767.0  # normalize to [-1, 1]
    Y_buttons = (Y[:, 4:] > 16000).astype(np.float32)  # binarize

    return Y_joystick, Y_buttons


# ── Models ──────────────────────────────────────────────────────────────────

class MLPDecoder(nn.Module):
    """Multi-layer perceptron decoder."""
    def __init__(self, input_size, n_joystick=4, n_buttons=8):
        super().__init__()
        self.flatten = nn.Flatten()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.joystick_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_joystick),
            nn.Tanh(),  # output in [-1, 1]
        )
        self.button_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_buttons),
        )

    def forward(self, x):
        x = self.flatten(x)
        shared = self.shared(x)
        joystick = self.joystick_head(shared)
        buttons = self.button_head(shared)
        return joystick, buttons


class CNNDecoder(nn.Module):
    """1D CNN decoder — temporal convolutions over binned spike features."""
    def __init__(self, n_features, n_bins, n_joystick=4, n_buttons=8):
        super().__init__()
        # Input: (batch, n_features, n_bins) — treat features as channels
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.shared = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.joystick_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_joystick),
            nn.Tanh(),
        )
        self.button_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_buttons),
        )

    def forward(self, x):
        # x: (batch, n_bins, n_features) → permute to (batch, n_features, n_bins)
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)
        shared = self.shared(x)
        joystick = self.joystick_head(shared)
        buttons = self.button_head(shared)
        return joystick, buttons


# ── Training ────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, model_name="model"):
    """Train a decoder model with dual loss (regression + classification)."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    joystick_loss_fn = nn.MSELoss()
    button_loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_joy_r2': [], 'val_btn_acc': []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, Y_joy_batch, Y_btn_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            Y_joy_batch = Y_joy_batch.to(DEVICE)
            Y_btn_batch = Y_btn_batch.to(DEVICE)

            joy_pred, btn_pred = model(X_batch)
            loss_joy = joystick_loss_fn(joy_pred, Y_joy_batch)
            loss_btn = button_loss_fn(btn_pred, Y_btn_batch)
            loss = loss_joy + loss_btn

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        all_joy_pred, all_joy_true = [], []
        all_btn_pred, all_btn_true = [], []

        with torch.no_grad():
            for X_batch, Y_joy_batch, Y_btn_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                Y_joy_batch = Y_joy_batch.to(DEVICE)
                Y_btn_batch = Y_btn_batch.to(DEVICE)

                joy_pred, btn_pred = model(X_batch)
                loss_joy = joystick_loss_fn(joy_pred, Y_joy_batch)
                loss_btn = button_loss_fn(btn_pred, Y_btn_batch)
                val_loss += (loss_joy + loss_btn).item()

                all_joy_pred.append(joy_pred.cpu().numpy())
                all_joy_true.append(Y_joy_batch.cpu().numpy())
                all_btn_pred.append((torch.sigmoid(btn_pred) > 0.5).cpu().numpy())
                all_btn_true.append(Y_btn_batch.cpu().numpy())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Metrics
        joy_pred_np = np.concatenate(all_joy_pred)
        joy_true_np = np.concatenate(all_joy_true)
        btn_pred_np = np.concatenate(all_btn_pred)
        btn_true_np = np.concatenate(all_btn_true)

        joy_r2 = r2_score(joy_true_np, joy_pred_np, multioutput='uniform_average')
        btn_acc = accuracy_score(btn_true_np.flatten(), btn_pred_np.flatten())

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_joy_r2'].append(joy_r2)
        history['val_btn_acc'].append(btn_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f} val={val_loss:.4f} "
                  f"joy_R²={joy_r2:.3f} btn_acc={btn_acc:.3f} lr={optimizer.param_groups[0]['lr']:.1e}")

    model.load_state_dict(best_state)
    return model, history


# ── Evaluation & Plotting ───────────────────────────────────────────────────

def evaluate_model(model, X_val, Y_joy_val, Y_btn_val, model_name, output_dir):
    """Detailed evaluation with per-channel metrics and plots."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_val).to(DEVICE)
        joy_pred, btn_pred = model(X_t)
        joy_pred = joy_pred.cpu().numpy()
        btn_pred = (torch.sigmoid(btn_pred) > 0.5).cpu().numpy().astype(float)

    # Per-channel R² for joysticks
    joy_names = ['L-Stick X', 'L-Stick Y', 'R-Stick X', 'R-Stick Y']
    btn_names = ['A', 'B', 'X', 'Y', 'LB', 'RB', 'LT', 'RT']

    print(f"\n{'='*50}")
    print(f"  {model_name} — Validation Results")
    print(f"{'='*50}")

    print("\nJoystick R² (per axis):")
    for i, name in enumerate(joy_names):
        r2 = r2_score(Y_joy_val[:, i], joy_pred[:, i])
        print(f"  {name}: R² = {r2:.4f}")

    print("\nButton Accuracy (per button):")
    for i, name in enumerate(btn_names):
        acc = accuracy_score(Y_btn_val[:, i], btn_pred[:, i])
        positives = Y_btn_val[:, i].sum()
        print(f"  {name}: acc = {acc:.4f} (n_pressed = {int(positives)})")

    # Plot predictions vs ground truth for joysticks
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    t = np.arange(len(Y_joy_val)) * BIN_SIZE_MS / 1000  # seconds
    for i, (ax, name) in enumerate(zip(axes, joy_names)):
        ax.plot(t, Y_joy_val[:, i], 'b-', alpha=0.5, label='True', linewidth=0.5)
        ax.plot(t, joy_pred[:, i], 'r-', alpha=0.5, label='Predicted', linewidth=0.5)
        r2 = r2_score(Y_joy_val[:, i], joy_pred[:, i])
        ax.set_ylabel(name)
        ax.set_title(f'{name} — R² = {r2:.4f}')
        ax.legend(loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'{model_name} — Joystick Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_joystick_predictions.png'), dpi=150)
    plt.close()

    # Plot button predictions
    fig, axes = plt.subplots(8, 1, figsize=(14, 12), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, btn_names)):
        ax.plot(t, Y_btn_val[:, i], 'b-', alpha=0.5, label='True', linewidth=0.5)
        ax.plot(t, btn_pred[:, i], 'r-', alpha=0.7, label='Predicted', linewidth=0.5)
        acc = accuracy_score(Y_btn_val[:, i], btn_pred[:, i])
        ax.set_ylabel(name)
        ax.set_title(f'{name} — Accuracy = {acc:.4f}')
        ax.set_ylim(-0.1, 1.1)
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'{model_name} — Button Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_button_predictions.png'), dpi=150)
    plt.close()

    return joy_pred, btn_pred


def plot_training_history(history, model_name, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].set_xlabel('Epoch')

    axes[1].plot(history['val_joy_r2'])
    axes[1].set_title('Joystick R²')
    axes[1].set_xlabel('Epoch')

    axes[2].plot(history['val_btn_acc'])
    axes[2].set_title('Button Accuracy')
    axes[2].set_xlabel('Epoch')

    plt.suptitle(f'{model_name} Training History', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_training_history.png'), dpi=150)
    plt.close()


# ── ONNX Export ─────────────────────────────────────────────────────────────

def export_onnx(model, input_shape, output_path):
    """Export model to ONNX format."""
    model.eval()
    model_cpu = model.cpu()
    dummy = torch.randn(1, *input_shape)

    # Wrapper to combine outputs into single tensor
    class ONNXWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            joy, btn = self.model(x)
            btn_sigmoid = torch.sigmoid(btn)
            return torch.cat([joy, btn_sigmoid], dim=1)

    wrapper = ONNXWrapper(model_cpu)
    torch.onnx.export(
        wrapper, dummy, output_path,
        input_names=['neural_features'],
        output_names=['controller_output'],
        dynamic_axes={'neural_features': {0: 'batch'}, 'controller_output': {0: 'batch'}},
        opset_version=13,
        dynamo=False,
    )
    print(f"ONNX model saved to {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    output_dir = 'analysis/decoder'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Load data
    print("=" * 60)
    print("  LOADING TRAINING DATA")
    print("=" * 60)
    neural, labels = load_all_training_data()

    # 2. Extract features
    print("\n" + "=" * 60)
    print("  EXTRACTING SPIKE FEATURES")
    print("=" * 60)
    features, n_bins = extract_spike_features(neural)
    binned_labels = bin_labels(labels, n_bins)

    # 3. Create windows
    X, Y = create_windows(features, binned_labels)
    Y_joy, Y_btn = process_labels(Y)

    print(f"\nJoystick targets: {Y_joy.shape} (range [{Y_joy.min():.2f}, {Y_joy.max():.2f}])")
    print(f"Button targets: {Y_btn.shape} (positive rate: {Y_btn.mean():.3f})")

    # 4. Train/val split (time-based, not random — avoid data leakage)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_joy_train, Y_joy_val = Y_joy[:split_idx], Y_joy[split_idx:]
    Y_btn_train, Y_btn_val = Y_btn[:split_idx], Y_btn[split_idx:]

    print(f"\nTrain: {X_train.shape[0]} windows")
    print(f"Val:   {X_val.shape[0]} windows")

    # Create data loaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(Y_joy_train),
        torch.FloatTensor(Y_btn_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(Y_joy_val),
        torch.FloatTensor(Y_btn_val),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    input_size = WINDOW_BINS * features.shape[1]  # flattened for MLP
    n_features = features.shape[1]

    # 5. Train MLP
    print("\n" + "=" * 60)
    print("  TRAINING MLP DECODER")
    print("=" * 60)
    mlp = MLPDecoder(input_size)
    mlp, mlp_history = train_model(mlp, train_loader, val_loader, epochs=150, lr=1e-3, model_name="MLP")
    plot_training_history(mlp_history, "MLP", output_dir)
    evaluate_model(mlp, X_val, Y_joy_val, Y_btn_val, "MLP", output_dir)

    # 6. Train CNN
    print("\n" + "=" * 60)
    print("  TRAINING CNN DECODER")
    print("=" * 60)
    cnn = CNNDecoder(n_features, WINDOW_BINS)
    cnn, cnn_history = train_model(cnn, train_loader, val_loader, epochs=150, lr=1e-3, model_name="CNN")
    plot_training_history(cnn_history, "CNN", output_dir)
    evaluate_model(cnn, X_val, Y_joy_val, Y_btn_val, "CNN", output_dir)

    # 7. Pick best model and export
    mlp_best_val = min(mlp_history['val_loss'])
    cnn_best_val = min(cnn_history['val_loss'])
    print(f"\n{'='*60}")
    print(f"  RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"MLP best val loss: {mlp_best_val:.4f}")
    print(f"CNN best val loss: {cnn_best_val:.4f}")

    if cnn_best_val < mlp_best_val:
        best_model = cnn
        best_name = "CNN"
        input_shape = (WINDOW_BINS, n_features)
    else:
        best_model = mlp
        best_name = "MLP"
        input_shape = (WINDOW_BINS, n_features)

    print(f"\nBest model: {best_name}")

    # Save PyTorch checkpoint first
    torch.save(best_model.state_dict(), os.path.join(output_dir, f'{best_name}_best.pt'))
    print(f"PyTorch checkpoint saved to {output_dir}/{best_name}_best.pt")

    export_onnx(best_model, input_shape, 'models/decoder.onnx')
    print(f"PyTorch checkpoint saved to {output_dir}/{best_name}_best.pt")

    print("\nDone! Check analysis/decoder/ for plots.")


if __name__ == '__main__':
    main()
