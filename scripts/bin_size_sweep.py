"""
Bin Size Sweep — find optimal spike-count integration window for MLP decoder.

Loads preprocessed binned features from data/preprocessed/bins_{W}ms.npz.
Run scripts/preprocess_bins.py first to generate those files.

For each bin size W in {20, 30, 40, 50, 60, 70, 80, 100} ms:
  1. Load canonical spike-count features from disk (precomputed, consistent).
  2. 80/20 temporal split (no shuffle — preserve causality).
  3. MLP: 64 → 128 → 64 → 12, ReLU, Adam, MSE, 100 epochs.
  4. Report mean R² and RMSE across all 12 outputs on val set.

Output: analysis/bin_size_sweep.png  +  stdout summary table.
"""

import numpy as np
import os
import gc
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_NEURAL     = 64
N_LABELS     = 12
PREPROCESSED = 'data/preprocessed'
BIN_SIZES_MS = [20, 30, 40, 50, 60, 70, 80, 100]
EPOCHS       = 100
BATCH_SIZE   = 256
DEVICE       = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def log(msg):
    print(msg, flush=True)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_bins(bin_ms):
    path = os.path.join(PREPROCESSED, f'bins_{bin_ms}ms.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run: python scripts/preprocess_bins.py")
    d = np.load(path)
    X = d['X']   # (n_bins, 64)  spike counts
    Y = d['Y']   # (n_bins, 12)  normalised labels
    feat_mean = d['feat_mean']
    feat_std  = d['feat_std']
    meta = {k: d[k].item() for k in ('bin_ms', 'spike_thresh', 'n_files',
                                      'total_seconds', 'sample_rate')}
    log(f"  Loaded bins_{bin_ms}ms.npz  —  {X.shape[0]:,} bins  "
        f"(thresh={meta['spike_thresh']}σ, {meta['n_files']} files, "
        f"{meta['total_seconds']:.0f}s data)")
    return X, Y, feat_mean, feat_std


# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim=64, out_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_eval(X, Y, bin_ms, feat_mean, feat_std):
    """80/20 temporal split, train MLP, return (mean_r2, mean_rmse)."""
    split = int(len(X) * 0.8)
    X_tr, X_val = X[:split], X[split:]
    Y_tr, Y_val = Y[:split], Y[split:]

    # Use precomputed normalisation stats (consistent with preprocessing)
    X_tr_n  = (X_tr  - feat_mean) / feat_std
    X_val_n = (X_val - feat_mean) / feat_std

    tr_ds  = TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val_n), torch.from_numpy(Y_val))
    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_r2 = -999.0
    best_state = None

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                preds.append(model(xb.to(DEVICE)).cpu().numpy())
                trues.append(yb.numpy())
        p = np.concatenate(preds)
        t = np.concatenate(trues)
        r2 = r2_score(t, p, multioutput='uniform_average')
        if r2 > best_r2:
            best_r2 = r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Final metrics with best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
            trues.append(yb.numpy())
    p = np.concatenate(preds)
    t = np.concatenate(trues)

    r2_per   = [r2_score(t[:, i], p[:, i]) for i in range(N_LABELS)]
    rmse_per = [np.sqrt(np.mean((t[:, i] - p[:, i])**2)) for i in range(N_LABELS)]

    mean_r2   = float(np.mean(r2_per))
    mean_rmse = float(np.mean(rmse_per))

    # Log per-output breakdown
    output_names = ['LStX','LStY','RStX','RStY','A','B','X','Y','LB','RB','LT','RT']
    detail = '  '.join(f'{n}={r:.3f}' for n, r in zip(output_names, r2_per))
    log(f"    per-output R²: {detail}")

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return mean_r2, mean_rmse, r2_per, rmse_per


# ── Main ──────────────────────────────────────────────────────────────────────

def find_elbow(values):
    """Return index of the elbow/plateau using max-curvature heuristic."""
    v = np.array(values, dtype=float)
    # Normalise to [0,1]
    v_norm = (v - v.min()) / (np.ptp(v) + 1e-12)
    x_norm = np.linspace(0, 1, len(v))
    # Distance from each point to line connecting first and last point
    line_vec  = np.array([x_norm[-1] - x_norm[0], v_norm[-1] - v_norm[0]])
    line_len  = np.linalg.norm(line_vec)
    line_unit = line_vec / (line_len + 1e-12)
    dists = []
    for xi, yi in zip(x_norm, v_norm):
        pt = np.array([xi - x_norm[0], yi - v_norm[0]])
        proj = np.dot(pt, line_unit) * line_unit
        dist = np.linalg.norm(pt - proj)
        dists.append(dist)
    return int(np.argmax(dists))


def main():
    t0 = time.time()
    log(f"Device: {DEVICE}")

    results = []   # (bin_ms, mean_r2, mean_rmse, r2_per, rmse_per)

    header = f"\n{'Bin(ms)':>8}  {'Bins':>7}  {'MeanR²':>8}  {'MeanRMSE':>10}  {'Time':>6}"
    log(header)
    log('-' * len(header))

    for bin_ms in BIN_SIZES_MS:
        t_bin = time.time()

        X, Y, feat_mean, feat_std = load_bins(bin_ms)

        mean_r2, mean_rmse, r2_per, rmse_per = train_and_eval(X, Y, bin_ms, feat_mean, feat_std)
        elapsed = time.time() - t_bin

        log(f"{bin_ms:>8}  {len(X):>7,}  {mean_r2:>+8.4f}  {mean_rmse:>10.4f}  {elapsed:>5.1f}s")
        results.append((bin_ms, mean_r2, mean_rmse, r2_per, rmse_per))

        del X, Y
        gc.collect()

    # ── Summary table ────────────────────────────────────────────────────────
    output_names = ['LStX','LStY','RStX','RStY','A','B','X','Y','LB','RB','LT','RT']
    best_r2_idx   = int(np.argmax([r[1] for r in results]))
    best_rmse_idx = int(np.argmin([r[2] for r in results]))

    log("\n" + "=" * 72)
    log("  BIN SIZE SWEEP — FULL RESULTS")
    log("=" * 72)
    log(f"{'Bin':>6}  {'Bins':>7}  {'MeanR²':>8}  {'MeanRMSE':>10}")
    log("-" * 36)
    for i, (bms, r2, rmse, _, _) in enumerate(results):
        n_bins = int(SAMPLE_RATE / (bms / 1000) * (len(neural) / SAMPLE_RATE))  # approx
        marker = " ← best R²" if i == best_r2_idx else (" ← best RMSE" if i == best_rmse_idx else "")
        log(f"{bms:>6}  {n_bins:>7,}  {r2:>+8.4f}  {rmse:>10.4f}{marker}")

    log("\nPer-output R² at best bin size "
        f"({results[best_r2_idx][0]}ms):")
    best_r2_per = results[best_r2_idx][3]
    for name, r2 in zip(output_names, best_r2_per):
        bar = '█' * max(0, int((r2 + 0.1) * 20))
        log(f"  {name:>5}: {r2:+.4f}  {bar}")
    log("=" * 72)
    log(f"Total time: {time.time()-t0:.1f}s")

    # ── Plot ─────────────────────────────────────────────────────────────────
    bin_sizes   = [r[0] for r in results]
    mean_r2s    = [r[1] for r in results]
    mean_rmses  = [r[2] for r in results]

    elbow_r2   = find_elbow(mean_r2s)
    elbow_rmse = find_elbow([-x for x in mean_rmses])  # invert for RMSE (lower is better)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # R² subplot
    ax1.plot(bin_sizes, mean_r2s, 'o-', color='steelblue', lw=2, ms=7)
    ax1.axvline(bin_sizes[elbow_r2], color='tomato', ls='--', lw=1.5,
                label=f'Elbow: {bin_sizes[elbow_r2]}ms')
    ax1.scatter([bin_sizes[best_r2_idx]], [mean_r2s[best_r2_idx]],
                s=120, zorder=5, color='gold', edgecolors='k', lw=1.2,
                label=f'Best: {bin_sizes[best_r2_idx]}ms ({mean_r2s[best_r2_idx]:+.3f})')
    for x, y in zip(bin_sizes, mean_r2s):
        ax1.annotate(f'{y:+.3f}', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax1.set_xlabel('Bin size (ms)')
    ax1.set_ylabel('Mean R²  (12 outputs)')
    ax1.set_title('Mean R² vs Bin Size')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # RMSE subplot
    ax2.plot(bin_sizes, mean_rmses, 's-', color='darkorange', lw=2, ms=7)
    ax2.axvline(bin_sizes[elbow_rmse], color='tomato', ls='--', lw=1.5,
                label=f'Elbow: {bin_sizes[elbow_rmse]}ms')
    ax2.scatter([bin_sizes[best_rmse_idx]], [mean_rmses[best_rmse_idx]],
                s=120, zorder=5, color='gold', edgecolors='k', lw=1.2,
                label=f'Best: {bin_sizes[best_rmse_idx]}ms ({mean_rmses[best_rmse_idx]:.3f})')
    for x, y in zip(bin_sizes, mean_rmses):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax2.set_xlabel('Bin size (ms)')
    ax2.set_ylabel('Mean RMSE  (12 outputs)')
    ax2.set_title('Mean RMSE vs Bin Size')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f'Bin Size Sweep — Spike Count MLP Decoder\n'
        f'64 neural channels → 128 → 64 → 12 outputs  |  '
        f'spike thresh = {SPIKE_THRESH}σ  |  non-overlapping bins',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig('analysis/bin_size_sweep.png', dpi=150)
    plt.close()
    log("Saved: analysis/bin_size_sweep.png")


if __name__ == '__main__':
    main()
