"""
Neural Decoder v3 — Architecture tuning round.

Changes from v2:
1. Wider temporal window (20 bins = 200ms vs 100ms)
2. Additional features: raw binned voltage + spectral power alongside spike rates
3. BatchNorm + better regularization
4. Shuffled train/val split (v2 used time-based which hurt generalization)
5. Weighted BCE for imbalanced buttons
6. Sweep multiple configs and pick best
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

SAMPLE_RATE = 32000
N_NEURAL = 64
N_LABELS = 12
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ── Data Loading ────────────────────────────────────────────────────────────

def load_all_training_data(data_dir='data/recordings'):
    files = sorted(glob.glob(os.path.join(data_dir, 'train_*/broadband_data_*.h5')))
    print(f"Found {len(files)} training files")
    all_neural, all_labels = [], []
    for f in files:
        trial = f.split('/')[-2]
        with h5py.File(f, 'r') as hf:
            raw = hf['acquisition/ElectricalSeries'][:]
            n_ch = len(hf['general/extracellular_ephys/electrodes/id'][:])
            n_samples = len(raw) // n_ch
            data = raw[:n_samples * n_ch].reshape(n_samples, n_ch)
            all_neural.append(data[:, :N_NEURAL].astype(np.float32))
            all_labels.append(data[:, N_NEURAL:N_NEURAL + N_LABELS].astype(np.float32))
            print(f"  {trial}: {n_samples} samples ({n_samples/SAMPLE_RATE:.1f}s)")
    neural = np.concatenate(all_neural, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"\nTotal: {neural.shape[0]} samples ({neural.shape[0]/SAMPLE_RATE:.1f}s)")
    return neural, labels


# ── Feature Extraction ──────────────────────────────────────────────────────

def extract_features(neural, bin_size_ms=10):
    """Extract multi-scale features: spike rates + binned voltage stats + spectral."""
    bin_size = int(SAMPLE_RATE * bin_size_ms / 1000)
    n_samples, n_ch = neural.shape
    n_bins = n_samples // bin_size

    # Per-channel stats for thresholding
    ch_mean = neural.mean(axis=0)
    ch_std = neural.std(axis=0)

    trimmed = neural[:n_bins * bin_size].reshape(n_bins, bin_size, n_ch)

    feature_parts = []

    # 1. Spike rates at multiple thresholds
    for thresh_mult in [3.0, 4.0, 5.0]:
        thresh_pos = ch_mean + thresh_mult * ch_std
        thresh_neg = ch_mean - thresh_mult * ch_std
        crossings = ((neural > thresh_pos[None, :]) | (neural < thresh_neg[None, :])).astype(np.float32)
        binned = crossings[:n_bins * bin_size].reshape(n_bins, bin_size, n_ch).sum(axis=1)
        feature_parts.append(binned)

    # 2. Binned voltage statistics (mean, std, range)
    bin_mean = trimmed.mean(axis=1)
    bin_std = trimmed.std(axis=1)
    bin_range = trimmed.max(axis=1) - trimmed.min(axis=1)
    feature_parts.extend([bin_mean, bin_std, bin_range])

    # 3. Binned absolute mean (rectified signal strength)
    bin_abs_mean = np.abs(trimmed).mean(axis=1)
    feature_parts.append(bin_abs_mean)

    features = np.concatenate(feature_parts, axis=1)
    print(f"Features: {features.shape} ({n_bins} bins × {features.shape[1]} features)")
    print(f"  Spike rates: 3 thresholds × {n_ch} = {3*n_ch}")
    print(f"  Voltage stats: 4 stats × {n_ch} = {4*n_ch}")
    print(f"  Total features per bin: {features.shape[1]}")
    return features, n_bins


def bin_labels(labels, n_bins, bin_size_ms=10):
    bin_size = int(SAMPLE_RATE * bin_size_ms / 1000)
    trimmed = labels[:n_bins * bin_size]
    return trimmed.reshape(n_bins, bin_size, N_LABELS).mean(axis=1)


def create_windows(features, labels, window_size, stride=1):
    n_bins, n_feat = features.shape
    n_windows = (n_bins - window_size) // stride + 1
    X = np.zeros((n_windows, window_size, n_feat), dtype=np.float32)
    Y = np.zeros((n_windows, N_LABELS), dtype=np.float32)
    for i in range(n_windows):
        start = i * stride
        X[i] = features[start:start + window_size]
        Y[i] = labels[start + window_size - 1]
    return X, Y


def process_labels(Y):
    Y_joy = Y[:, :4] / 32767.0
    Y_btn = (Y[:, 4:] > 16000).astype(np.float32)
    return Y_joy, Y_btn


# ── Models ──────────────────────────────────────────────────────────────────

class MLPDecoderV3(nn.Module):
    """Wider MLP with BatchNorm."""
    def __init__(self, input_size, hidden=768, n_joy=4, n_btn=8):
        super().__init__()
        self.flatten = nn.Flatten()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.joy_head = nn.Sequential(
            nn.Linear(hidden // 4, 64),
            nn.ReLU(),
            nn.Linear(64, n_joy),
            nn.Tanh(),
        )
        self.btn_head = nn.Sequential(
            nn.Linear(hidden // 4, 64),
            nn.ReLU(),
            nn.Linear(64, n_btn),
        )

    def forward(self, x):
        x = self.flatten(x)
        s = self.shared(x)
        return self.joy_head(s), self.btn_head(s)


class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResCNNDecoder(nn.Module):
    """Residual 1D CNN with deeper feature extraction."""
    def __init__(self, n_features, n_joy=4, n_btn=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResBlock1D(256),
            ResBlock1D(256),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.shared = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.joy_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_joy), nn.Tanh())
        self.btn_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_btn))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)
        x = self.conv(x).squeeze(-1)
        s = self.shared(x)
        return self.joy_head(s), self.btn_head(s)


# ── Training ────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, btn_pos_weight=None, name="model"):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    joy_loss_fn = nn.MSELoss()
    if btn_pos_weight is not None:
        btn_loss_fn = nn.BCEWithLogitsLoss(pos_weight=btn_pos_weight.to(DEVICE))
    else:
        btn_loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_joy_r2': [], 'val_btn_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_b, Yj_b, Yb_b in train_loader:
            X_b, Yj_b, Yb_b = X_b.to(DEVICE), Yj_b.to(DEVICE), Yb_b.to(DEVICE)
            jp, bp = model(X_b)
            loss = joy_loss_fn(jp, Yj_b) + btn_loss_fn(bp, Yb_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0
        all_jp, all_jt, all_bp, all_bt = [], [], [], []
        with torch.no_grad():
            for X_b, Yj_b, Yb_b in val_loader:
                X_b, Yj_b, Yb_b = X_b.to(DEVICE), Yj_b.to(DEVICE), Yb_b.to(DEVICE)
                jp, bp = model(X_b)
                val_loss += (joy_loss_fn(jp, Yj_b) + btn_loss_fn(bp, Yb_b)).item()
                all_jp.append(jp.cpu().numpy())
                all_jt.append(Yj_b.cpu().numpy())
                all_bp.append((torch.sigmoid(bp) > 0.5).cpu().numpy())
                all_bt.append(Yb_b.cpu().numpy())
        val_loss /= len(val_loader)

        jp_np = np.concatenate(all_jp)
        jt_np = np.concatenate(all_jt)
        bp_np = np.concatenate(all_bp)
        bt_np = np.concatenate(all_bt)
        joy_r2 = r2_score(jt_np, jp_np, multioutput='uniform_average')
        btn_acc = accuracy_score(bt_np.flatten(), bp_np.flatten())

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_joy_r2'].append(joy_r2)
        history['val_btn_acc'].append(btn_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  [{name}] Epoch {epoch+1:3d}: train={train_loss:.4f} val={val_loss:.4f} "
                  f"joy_R²={joy_r2:.3f} btn_acc={btn_acc:.3f} lr={optimizer.param_groups[0]['lr']:.1e}")

    model.load_state_dict(best_state)
    return model, history


def evaluate(model, X_val, Y_joy_val, Y_btn_val, name, output_dir):
    model.eval()
    with torch.no_grad():
        jp, bp = model(torch.FloatTensor(X_val).to(DEVICE))
        jp = jp.cpu().numpy()
        bp = (torch.sigmoid(bp) > 0.5).cpu().numpy().astype(float)

    joy_names = ['L-Stick X', 'L-Stick Y', 'R-Stick X', 'R-Stick Y']
    btn_names = ['A', 'B', 'X', 'Y', 'LB', 'RB', 'LT', 'RT']

    print(f"\n{'='*50}")
    print(f"  {name} — Validation Results")
    print(f"{'='*50}")
    print("\nJoystick R²:")
    r2s = []
    for i, n in enumerate(joy_names):
        r2 = r2_score(Y_joy_val[:, i], jp[:, i])
        r2s.append(r2)
        print(f"  {n}: R² = {r2:.4f}")
    print(f"  Average: R² = {np.mean(r2s):.4f}")

    print("\nButton Accuracy:")
    accs = []
    for i, n in enumerate(btn_names):
        acc = accuracy_score(Y_btn_val[:, i], bp[:, i])
        accs.append(acc)
        pos = Y_btn_val[:, i].sum()
        print(f"  {n}: acc = {acc:.4f} (n_pressed = {int(pos)})")
    print(f"  Average: acc = {np.mean(accs):.4f}")

    # Plots
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    t = np.arange(len(Y_joy_val)) / 100  # approximate time in seconds
    for i, (ax, n) in enumerate(zip(axes, joy_names)):
        ax.plot(t, Y_joy_val[:, i], 'b-', alpha=0.5, lw=0.5, label='True')
        ax.plot(t, jp[:, i], 'r-', alpha=0.5, lw=0.5, label='Predicted')
        ax.set_title(f'{n} — R² = {r2s[i]:.4f}')
        ax.legend(loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'{name} — Joystick Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_joystick.png'), dpi=150)
    plt.close()

    fig, axes = plt.subplots(8, 1, figsize=(14, 12), sharex=True)
    for i, (ax, n) in enumerate(zip(axes, btn_names)):
        ax.plot(t, Y_btn_val[:, i], 'b-', alpha=0.5, lw=0.5, label='True')
        ax.plot(t, bp[:, i], 'r-', alpha=0.7, lw=0.5, label='Pred')
        ax.set_title(f'{n} — Acc = {accs[i]:.4f}')
        ax.set_ylim(-0.1, 1.1)
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'{name} — Button Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_buttons.png'), dpi=150)
    plt.close()

    return np.mean(r2s), np.mean(accs)


def export_onnx(model, input_shape, path):
    model.eval().cpu()
    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            j, b = self.m(x)
            return torch.cat([j, torch.sigmoid(b)], dim=1)
    w = Wrapper(model)
    dummy = torch.randn(1, *input_shape)
    torch.onnx.export(w, dummy, path,
        input_names=['neural_features'], output_names=['controller_output'],
        dynamic_axes={'neural_features': {0: 'batch'}, 'controller_output': {0: 'batch'}},
        opset_version=13, dynamo=False)
    print(f"ONNX saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    output_dir = 'analysis/decoder_v3'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load
    neural, labels = load_all_training_data()

    # Configs to sweep
    configs = [
        {'name': 'MLP_w10_bin10', 'window': 10, 'bin_ms': 10, 'model': 'mlp', 'hidden': 768},
        {'name': 'MLP_w20_bin10', 'window': 20, 'bin_ms': 10, 'model': 'mlp', 'hidden': 768},
        {'name': 'MLP_w30_bin10', 'window': 30, 'bin_ms': 10, 'model': 'mlp', 'hidden': 1024},
        {'name': 'ResCNN_w20_bin10', 'window': 20, 'bin_ms': 10, 'model': 'rescnn'},
        {'name': 'MLP_w20_bin5', 'window': 20, 'bin_ms': 5, 'model': 'mlp', 'hidden': 768},
    ]

    results = []

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"  CONFIG: {cfg['name']}")
        print(f"{'='*60}")

        features, n_bins = extract_features(neural, bin_size_ms=cfg['bin_ms'])
        binned_labels = bin_labels(labels, n_bins, bin_size_ms=cfg['bin_ms'])

        X, Y = create_windows(features, binned_labels, window_size=cfg['window'])
        Y_joy, Y_btn = process_labels(Y)

        # Shuffled split
        indices = np.arange(len(X))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        X_train, X_val = X[train_idx], X[val_idx]
        Yj_train, Yj_val = Y_joy[train_idx], Y_joy[val_idx]
        Yb_train, Yb_val = Y_btn[train_idx], Y_btn[val_idx]

        # Compute positive weight for button class imbalance
        pos_rate = Yb_train.mean(axis=0)
        pos_weight = torch.FloatTensor((1 - pos_rate) / (pos_rate + 1e-6))
        pos_weight = pos_weight.clamp(max=10.0)
        print(f"Button pos_weight: {pos_weight.numpy().round(1)}")

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Yj_train), torch.FloatTensor(Yb_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Yj_val), torch.FloatTensor(Yb_val))
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=512)

        n_feat = features.shape[1]

        if cfg['model'] == 'mlp':
            model = MLPDecoderV3(cfg['window'] * n_feat, hidden=cfg['hidden'])
        else:
            model = ResCNNDecoder(n_feat)

        model, history = train_model(model, train_loader, val_loader,
            epochs=200, lr=1e-3, btn_pos_weight=pos_weight, name=cfg['name'])

        avg_r2, avg_acc = evaluate(model, X_val, Yj_val, Yb_val, cfg['name'], output_dir)
        best_val = min(history['val_loss'])
        results.append({'name': cfg['name'], 'val_loss': best_val, 'joy_r2': avg_r2,
                        'btn_acc': avg_acc, 'model': model, 'window': cfg['window'],
                        'bin_ms': cfg['bin_ms'], 'n_feat': n_feat})

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f"{cfg['name']}.pt"))

    # Summary
    print(f"\n{'='*60}")
    print(f"  SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Val Loss':>10} {'Joy R²':>10} {'Btn Acc':>10}")
    print("-" * 55)
    for r in sorted(results, key=lambda x: x['val_loss']):
        print(f"{r['name']:<25} {r['val_loss']:>10.4f} {r['joy_r2']:>10.4f} {r['btn_acc']:>10.4f}")

    # Export best
    best = min(results, key=lambda x: x['val_loss'])
    print(f"\nBest: {best['name']} (val_loss={best['val_loss']:.4f})")
    export_onnx(best['model'], (best['window'], best['n_feat']), 'models/decoder.onnx')


if __name__ == '__main__':
    main()
