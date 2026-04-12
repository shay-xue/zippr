#!/usr/bin/env python3
"""
compare_models.py

Trains and evaluates 5 neural decoding algorithms on labelled neural time-series
data and generates plots to find the optimal speed-accuracy balance for BCI
real-time deployment.

Data contract
-------------
Input  X : (n_samples, n_timesteps, 64)  – raw time-series for the first 64 channels
Label  y : (n_samples,)                  – integer in [0, 11], which of the 12
                                           output channels lit up (argmax of one-hot)

Algorithms
----------
1. LDA   – Linear Discriminant Analysis      (sklearn)  – flat input
2. RF    – Random Forest                      (sklearn)  – flat input
3. XGB   – XGBoost                            (xgboost)  – flat input
4. MLP   – Multi-Layer Perceptron             (PyTorch)  – flat input
5. LSTM  – Long Short-Term Memory             (PyTorch)  – sequential input

"Flat input" means X is reshaped from (n_samples, n_timesteps, 64) →
(n_samples, n_timesteps * 64) with NO other preprocessing.

Supported data file formats (pass via --data)
---------------------------------------------
.h5 / .hdf5
    Layout A – separate arrays:  keys 'X' (n,t,64) and 'y' (n,12) or (n,)
    Layout B – combined array:   key  'data' (n,t,76) → X=[:,:,:64], y=[:,:,64:]
.npz        keys 'X' and 'y'  (same shape conventions as above)
.npy        single array (n,t,76) → X=[:,:,:64], y=[:,:,64:]

If --data is omitted the script falls back to synthetic data.

Usage
-----
    # real data
    python scripts/compare_models.py --data data/recordings/session1.h5 --out-dir models/attempt\ 2

    # synthetic data
    python scripts/compare_models.py --out-dir models/attempt\ 2

    # tune hyper-parameters
    python scripts/compare_models.py --data session.h5 --epochs 80 --batch-size 128
"""

import argparse
import os
import time
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

# ─── Fixed architecture constants ────────────────────────────────────────────
N_INPUT_CHANNELS = 64   # first 64 channels are neural features
N_LABEL_CHANNELS = 12   # last 12 channels are the one-hot label
N_CLASSES        = 12   # one output class per label channel

PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

# ─── Defaults for synthetic fallback ─────────────────────────────────────────
SIM_DEFAULTS = dict(n_samples=4000, n_timesteps=20, seed=42)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load neural recording data from file.

    Returns
    -------
    X : float32 (n_samples, n_timesteps, 64)   – raw neural time series
    y : int64   (n_samples,)                    – class label 0-11
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            keys = list(f.keys())
            if "X" in keys and "y" in keys:
                X_raw = np.array(f["X"], dtype=np.float32)
                y_raw = np.array(f["y"])
            elif "data" in keys:
                data = np.array(f["data"], dtype=np.float32)
                X_raw, y_raw = _split_combined(data)
            else:
                raise KeyError(
                    f"HDF5 file must contain ('X','y') or 'data'. Found: {keys}"
                )

    elif ext == ".npz":
        data = np.load(p)
        if "X" in data and "y" in data:
            X_raw = data["X"].astype(np.float32)
            y_raw = data["y"]
        elif "arr_0" in data:
            X_raw, y_raw = _split_combined(data["arr_0"].astype(np.float32))
        else:
            raise KeyError(f"NPZ must contain 'X'+'y' or 'arr_0'. Found: {list(data.keys())}")

    elif ext == ".npy":
        arr = np.load(p).astype(np.float32)
        X_raw, y_raw = _split_combined(arr)

    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .h5, .npz, or .npy")

    X, y = _normalise_shapes(X_raw, y_raw)
    print(f"Loaded  X={X.shape}  y={y.shape}  classes={np.unique(y).tolist()}")
    return X, y


def _split_combined(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a (n, t, 76) combined array into X (n, t, 64) and y_raw (n, 12).
    y_raw is the mean of the label channels across time (majority vote per trial).
    """
    if data.ndim == 3 and data.shape[-1] >= 76:
        X_raw  = data[:, :, :N_INPUT_CHANNELS]
        y_ch   = data[:, :, N_INPUT_CHANNELS:N_INPUT_CHANNELS + N_LABEL_CHANNELS]
        y_raw  = y_ch.mean(axis=1)   # average over time → (n, 12)
        return X_raw, y_raw
    raise ValueError(
        f"Combined array must be (n, t, ≥76). Got shape {data.shape}"
    )


def _normalise_shapes(
    X_raw: np.ndarray, y_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce X to (n, t, 64) float32 and y to (n,) int64 class indices.

    y_raw can be:
      (n, 12) one-hot or soft activations → argmax → int label
      (n,)    already integer labels
    """
    # ── X ──────────────────────────────────────────────────────────────────
    if X_raw.ndim == 2:
        # (n, 64): no time dimension – add a dummy time axis of length 1
        X = X_raw[:, np.newaxis, :]
    elif X_raw.ndim == 3:
        X = X_raw
    else:
        raise ValueError(f"X must be 2-D or 3-D, got shape {X_raw.shape}")

    if X.shape[-1] != N_INPUT_CHANNELS:
        raise ValueError(
            f"Expected {N_INPUT_CHANNELS} input channels, got {X.shape[-1]}"
        )

    # ── y ──────────────────────────────────────────────────────────────────
    if y_raw.ndim == 2:
        y = y_raw.argmax(axis=1).astype(np.int64)
    elif y_raw.ndim == 1:
        y = y_raw.astype(np.int64)
    else:
        raise ValueError(f"y must be 1-D or 2-D, got shape {y_raw.shape}")

    if y.max() >= N_CLASSES:
        raise ValueError(
            f"y contains class index {y.max()} but N_CLASSES={N_CLASSES}"
        )

    return X.astype(np.float32), y


# ─── Synthetic Fallback ───────────────────────────────────────────────────────

def simulate_neural_data(n_samples: int, n_timesteps: int, seed: int = 42):
    """
    Generate mock neural spike-rate data with class-specific tuning curves.

    Returns X (n, t, 64) float32 and y (n,) int64.
    """
    rng = np.random.default_rng(seed)
    class_means = rng.uniform(0.1, 1.0, size=(N_CLASSES, N_INPUT_CHANNELS))
    y = rng.integers(0, N_CLASSES, size=n_samples).astype(np.int64)

    X = np.zeros((n_samples, n_timesteps, N_INPUT_CHANNELS), dtype=np.float32)
    for i, c in enumerate(y):
        for t in range(n_timesteps):
            scale = 8.0 + t * 0.1
            X[i, t] = (
                rng.poisson(class_means[c] * scale) / scale
                + rng.normal(0, 0.05, N_INPUT_CHANNELS)
            ).astype(np.float32)

    return X, y


# ─── Feature Extraction (flatten only — no preprocessing) ────────────────────

def flatten(X: np.ndarray) -> np.ndarray:
    """(n, t, 64) → (n, t*64) with no normalisation or filtering."""
    n = X.shape[0]
    return X.reshape(n, -1).astype(np.float32)


# ─── PyTorch Models ───────────────────────────────────────────────────────────

class MLPDecoder(nn.Module):
    """Fully-connected decoder. Input: flattened time-series (n_timesteps * 64)."""

    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMDecoder(nn.Module):
    """Stacked LSTM decoder. Input: (batch, n_timesteps, 64) raw time-series."""

    def __init__(self, hidden_size: int = 128, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            N_INPUT_CHANNELS, hidden_size, n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)         # (batch, t, hidden)
        return self.head(out[:, -1])  # last timestep → logits


# ─── Training & Evaluation ────────────────────────────────────────────────────

def train_torch(model, X_tr, y_tr, epochs, batch_size, device):
    """Return per-epoch loss list and total training seconds."""
    model.to(device)
    opt       = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(
        torch.from_numpy(X_tr).to(device),
        torch.from_numpy(y_tr).to(device),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    losses = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        scheduler.step()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{epochs}  loss={losses[-1]:.4f}")

    return losses, time.perf_counter() - t0


def eval_torch(model, X_te, y_te, device, n_timing=300):
    """Return acc, confusion matrix, probabilities, single-sample latency (ms)."""
    model.eval()
    X_t = torch.from_numpy(X_te).to(device)

    with torch.no_grad():
        logits = model(X_t)
        preds  = logits.argmax(dim=1).cpu().numpy()
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_te, preds)
    cm  = confusion_matrix(y_te, preds, labels=list(range(N_CLASSES)))

    single = torch.from_numpy(X_te[:1]).to(device)
    with torch.no_grad():
        for _ in range(20):          # warm up
            model(single)
    times = []
    with torch.no_grad():
        for _ in range(n_timing):
            t0 = time.perf_counter()
            model(single)
            times.append(time.perf_counter() - t0)
    lat_ms = float(np.median(times)) * 1000

    return acc, cm, probs, lat_ms


def eval_sklearn(clf, X_te, y_te, n_timing=300):
    """Return acc, confusion matrix, probabilities, single-sample latency (ms)."""
    preds  = clf.predict(X_te)
    probs  = clf.predict_proba(X_te) if hasattr(clf, "predict_proba") else None
    acc    = accuracy_score(y_te, preds)
    cm     = confusion_matrix(y_te, preds, labels=list(range(N_CLASSES)))

    for _ in range(20):
        clf.predict(X_te[:1])
    times = []
    for _ in range(n_timing):
        t0 = time.perf_counter()
        clf.predict(X_te[:1])
        times.append(time.perf_counter() - t0)
    lat_ms = float(np.median(times)) * 1000

    return acc, cm, probs, lat_ms


# ─── ITR ──────────────────────────────────────────────────────────────────────

def itr_bits_per_min(acc: float, lat_ms: float) -> float:
    """Nykopp ITR (bits/min) for N_CLASSES-class decoder."""
    if acc <= 0 or acc >= 1:
        return 0.0
    p = np.clip(acc, 1e-9, 1 - 1e-9)
    b = (
        np.log2(N_CLASSES)
        + p * np.log2(p)
        + (1 - p) * np.log2(np.clip((1 - p) / (N_CLASSES - 1), 1e-12, None))
    )
    return max(b, 0.0) * (60_000.0 / lat_ms)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(cfg: dict):
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or simulate data ─────────────────────────────────────────────────
    if cfg["data"]:
        X_seq, y = load_data(cfg["data"])
    else:
        print("No --data provided → using synthetic data")
        X_seq, y = simulate_neural_data(
            cfg["n_samples"], cfg["n_timesteps"], cfg["seed"]
        )
        print(f"Simulated  X={X_seq.shape}  y={y.shape}")

    n_timesteps = X_seq.shape[1]
    flat_features = n_timesteps * N_INPUT_CHANNELS
    print(f"n_timesteps={n_timesteps}  flat_features={flat_features}  n_classes={N_CLASSES}")

    # Flat version for LDA / RF / XGB / MLP — raw reshape, no preprocessing
    X_flat = flatten(X_seq)   # (n, t*64)

    (Xf_tr, Xf_te,
     Xs_tr, Xs_te,
     y_tr,  y_te) = train_test_split(
        X_flat, X_seq, y,
        test_size=0.2, random_state=cfg["seed"], stratify=y,
    )

    results = {}

    # ── 1. LDA ────────────────────────────────────────────────────────────────
    print("\n[1/5] LDA")
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis()
    lda.fit(Xf_tr, y_tr)
    tt = time.perf_counter() - t0
    acc, cm, probs, lat = eval_sklearn(lda, Xf_te, y_te)
    results["LDA"] = dict(acc=acc, latency_ms=lat, train_time=tt,
                          cm=cm, probs=probs, losses=None)
    print(f"  acc={acc:.3f}  lat={lat:.4f}ms  train={tt:.2f}s")

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    print("\n[2/5] Random Forest")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=cfg["seed"]
    )
    rf.fit(Xf_tr, y_tr)
    tt = time.perf_counter() - t0
    acc, cm, probs, lat = eval_sklearn(rf, Xf_te, y_te)
    results["Random Forest"] = dict(acc=acc, latency_ms=lat, train_time=tt,
                                    cm=cm, probs=probs, losses=None)
    print(f"  acc={acc:.3f}  lat={lat:.4f}ms  train={tt:.2f}s")

    # ── 3. XGBoost ────────────────────────────────────────────────────────────
    print("\n[3/5] XGBoost")
    t0 = time.perf_counter()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=cfg["seed"],
        verbosity=0,
    )
    xgb_clf.fit(Xf_tr, y_tr)
    tt = time.perf_counter() - t0
    acc, cm, probs, lat = eval_sklearn(xgb_clf, Xf_te, y_te)
    results["XGBoost"] = dict(acc=acc, latency_ms=lat, train_time=tt,
                               cm=cm, probs=probs, losses=None)
    print(f"  acc={acc:.3f}  lat={lat:.4f}ms  train={tt:.2f}s")

    # ── 4. MLP ────────────────────────────────────────────────────────────────
    print("\n[4/5] MLP")
    mlp = MLPDecoder(flat_features)
    losses_mlp, tt = train_torch(
        mlp, Xf_tr, y_tr, cfg["epochs"], cfg["batch_size"], device
    )
    acc, cm, probs, lat = eval_torch(mlp, Xf_te, y_te, device)
    results["MLP"] = dict(acc=acc, latency_ms=lat, train_time=tt,
                          cm=cm, probs=probs, losses=losses_mlp)
    print(f"  acc={acc:.3f}  lat={lat:.4f}ms  train={tt:.2f}s")

    # ── 5. LSTM ───────────────────────────────────────────────────────────────
    print("\n[5/5] LSTM")
    lstm = LSTMDecoder(hidden_size=128, n_layers=2)
    losses_lstm, tt = train_torch(
        lstm, Xs_tr, y_tr, cfg["epochs"], cfg["batch_size"], device
    )
    acc, cm, probs, lat = eval_torch(lstm, Xs_te, y_te, device)
    results["LSTM"] = dict(acc=acc, latency_ms=lat, train_time=tt,
                           cm=cm, probs=probs, losses=losses_lstm)
    print(f"  acc={acc:.3f}  lat={lat:.4f}ms  train={tt:.2f}s")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_comparison(results, out_dir)
    _plot_dynamics(results, y_te, out_dir)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "─" * 62)
    print(f"{'Model':<16} {'Acc':>7} {'Lat(ms)':>10} {'Train(s)':>10} {'ITR':>8}")
    print("─" * 62)
    for name in results:
        r = results[name]
        itr = itr_bits_per_min(r["acc"], r["latency_ms"])
        print(f"{name:<16} {r['acc']:>7.3f} {r['latency_ms']:>10.4f} "
              f"{r['train_time']:>10.2f} {itr:>8.1f}")
    print("─" * 62)

    best = max(results, key=lambda n: itr_bits_per_min(
        results[n]["acc"], results[n]["latency_ms"]
    ))
    print(f"\n★  Best ITR model: {best}")


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _plot_comparison(results: dict, out_dir: Path):
    """Figure 1: speed/accuracy scatter, bar charts, confusion matrices."""
    names  = list(results.keys())
    accs   = [results[n]["acc"]        for n in names]
    lats   = [results[n]["latency_ms"] for n in names]
    trains = [results[n]["train_time"] for n in names]

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle("Neural Decoder — Model Comparison  (12-class, 64-channel input)",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    # ── Speed vs Accuracy scatter ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    for i, name in enumerate(names):
        ax1.scatter(lats[i], accs[i], s=220, color=PALETTE[i], zorder=5,
                    label=name, edgecolors="white", linewidths=1.5)
        ax1.annotate(name, (lats[i], accs[i]),
                     textcoords="offset points", xytext=(10, 4), fontsize=10)

    # Pareto front
    sorted_pts = sorted(zip(lats, accs), key=lambda x: x[0])
    px, py, best_acc = [], [], -1
    for lat, acc in sorted_pts:
        if acc > best_acc:
            px.append(lat); py.append(acc); best_acc = acc
    if len(px) > 1:
        ax1.step(px, py, where="post", color="gray", linestyle="--",
                 alpha=0.55, label="Pareto front", linewidth=1.5)

    mid_lat = float(np.median(lats))
    mid_acc = float(np.median(accs))
    ax1.axhline(mid_acc, color="gray", alpha=0.18, linewidth=0.9)
    ax1.axvline(mid_lat, color="gray", alpha=0.18, linewidth=0.9)
    ax1.fill_betweenx([mid_acc, 1.05], 0, mid_lat, alpha=0.06, color="green")
    ax1.text(0.02, (mid_acc + 1.05) / 2, "OPTIMAL\nZONE",
             fontsize=9, color="green", alpha=0.7, va="center",
             transform=ax1.get_yaxis_transform())
    ax1.set_xlabel("Single-sample inference latency (ms)", fontsize=11)
    ax1.set_ylabel("Test accuracy", fontsize=11)
    ax1.set_title("Speed vs Accuracy Trade-off  (upper-left = ideal)", fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim(0, 1.08)
    ax1.grid(True, alpha=0.3)

    # ── Accuracy bar chart ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    bars = ax2.bar(names, accs, color=PALETTE, edgecolor="white", linewidth=1.2)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Test Accuracy", fontsize=12)
    ax2.tick_params(axis="x", rotation=30)
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    # ── Inference latency bar chart ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bars2 = ax3.bar(names, lats, color=PALETTE, edgecolor="white", linewidth=1.2)
    ax3.set_ylabel("Latency (ms)", fontsize=11)
    ax3.set_title("Single-Sample Inference Latency\n(lower = faster decoding)", fontsize=11)
    ax3.tick_params(axis="x", rotation=30)
    for bar, lat in zip(bars2, lats):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f"{lat:.4f}", ha="center", va="bottom", fontsize=9)
    ax3.grid(True, axis="y", alpha=0.3)

    # ── Training time bar chart ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    bars3 = ax4.bar(names, trains, color=PALETTE, edgecolor="white", linewidth=1.2)
    ax4.set_ylabel("Seconds", fontsize=11)
    ax4.set_title("Training Time", fontsize=12)
    ax4.tick_params(axis="x", rotation=30)
    for bar, t in zip(bars3, trains):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f"{t:.1f}s", ha="center", va="bottom", fontsize=9)
    ax4.grid(True, axis="y", alpha=0.3)

    # ── Composite score  acc / log(1 + lat_ms) ───────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    raw_scores  = [acc / np.log1p(lat) for acc, lat in zip(accs, lats)]
    scores_norm = np.array(raw_scores) / max(raw_scores)
    bars4 = ax5.bar(names, scores_norm, color=PALETTE, edgecolor="white", linewidth=1.2)
    best_i = int(np.argmax(scores_norm))
    bars4[best_i].set_edgecolor("gold"); bars4[best_i].set_linewidth(3)
    ax5.set_ylim(0, 1.18)
    ax5.set_ylabel("Score (normalised)", fontsize=11)
    ax5.set_title("Composite Score\nacc / log(1 + latency_ms)", fontsize=11)
    ax5.tick_params(axis="x", rotation=30)
    for bar, s in zip(bars4, scores_norm):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{s:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax5.text(0.5, 1.10, f"★  Best: {names[best_i]}",
             ha="center", va="center", transform=ax5.transAxes,
             fontsize=10, color="goldenrod", fontweight="bold")
    ax5.grid(True, axis="y", alpha=0.3)

    # ── Confusion matrices (top 3 by accuracy) ───────────────────────────────
    top3 = sorted(range(len(accs)), key=lambda i: accs[i], reverse=True)[:3]
    for col, mi in enumerate(top3):
        ax = fig.add_subplot(gs[2, col])
        cm = results[names[mi]]["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(
            f"Confusion Matrix\n{names[mi]}  (acc={accs[mi]:.3f})", fontsize=10
        )
        ax.set_xlabel("Predicted channel", fontsize=9)
        ax.set_ylabel("True channel", fontsize=9)
        ticks = range(N_CLASSES)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for r in range(N_CLASSES):
            for c in range(N_CLASSES):
                v = cm_norm[r, c]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.5 else "black")

    out = out_dir / "model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close(fig)


def _plot_dynamics(results: dict, y_te: np.ndarray, out_dir: Path):
    """Figure 2: training loss, macro ROC, ITR bar chart."""
    names = list(results.keys())
    accs  = [results[n]["acc"]        for n in names]
    lats  = [results[n]["latency_ms"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        "Training Dynamics & BCI Performance  (64-ch → 12-class decoder)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ── Learning curves (MLP & LSTM) ─────────────────────────────────────────
    ax = axes[0]
    for name, color in zip(["MLP", "LSTM"], PALETTE[3:]):
        losses = results[name]["losses"]
        if losses:
            ax.plot(losses, label=name, color=color, linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-entropy loss", fontsize=11)
    ax.set_title("Training Loss Curves\n(MLP & LSTM)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Macro-average ROC ─────────────────────────────────────────────────────
    ax = axes[1]
    y_bin = label_binarize(y_te, classes=list(range(N_CLASSES)))
    for i, name in enumerate(names):
        probs = results[name]["probs"]
        if probs is None:
            continue
        fpr_dict, tpr_dict, auc_vals = {}, {}, []
        for c in range(N_CLASSES):
            fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
            fpr_dict[c] = fpr; tpr_dict[c] = tpr
            auc_vals.append(auc(fpr, tpr))
        macro_auc = float(np.mean(auc_vals))
        all_fpr   = np.unique(np.concatenate(list(fpr_dict.values())))
        mean_tpr  = np.mean(
            [np.interp(all_fpr, fpr_dict[c], tpr_dict[c]) for c in range(N_CLASSES)],
            axis=0,
        )
        ax.plot(all_fpr, mean_tpr, color=PALETTE[i], lw=2,
                label=f"{name}  (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title("Macro-Average ROC Curves", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Information Transfer Rate ─────────────────────────────────────────────
    ax = axes[2]
    itrs  = [itr_bits_per_min(acc, lat) for acc, lat in zip(accs, lats)]
    bars  = ax.bar(names, itrs, color=PALETTE, edgecolor="white", linewidth=1.2)
    best_i = int(np.argmax(itrs))
    bars[best_i].set_edgecolor("gold"); bars[best_i].set_linewidth(3)
    ax.set_ylabel("ITR (bits / min)", fontsize=11)
    ax.set_title("Information Transfer Rate\nbalances accuracy AND speed", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    for bar, itr in zip(bars, itrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{itr:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(0.5, 0.97, f"★  Best ITR: {names[best_i]}",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=10, color="goldenrod", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "model_comparison_dynamics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare 5 neural decoders: 64-channel time-series → 12-class one-hot"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data file (.h5, .npz, .npy). Omit to use synthetic data.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models/attempt 2",
        help="Directory to save plots and results (default: 'models/attempt 2')",
    )
    parser.add_argument("--n-samples",   type=int, default=SIM_DEFAULTS["n_samples"],
                        help="Synthetic data only: number of trials")
    parser.add_argument("--n-timesteps", type=int, default=SIM_DEFAULTS["n_timesteps"],
                        help="Synthetic data only: time bins per trial")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--seed",        type=int, default=SIM_DEFAULTS["seed"])
    args = parser.parse_args()

    main({
        "data":        args.data,
        "out_dir":     args.out_dir,
        "n_samples":   args.n_samples,
        "n_timesteps": args.n_timesteps,
        "batch_size":  args.batch_size,
        "epochs":      args.epochs,
        "seed":        args.seed,
    })
