"""
Bin Size Sweep (RNN) — same sweep as bin_size_sweep.py but with a GRU decoder.

Loads preprocessed binned features from data/preprocessed/bins_{W}ms.npz.
Run scripts/preprocess_bins.py first to generate those files.

The RNN naturally accumulates temporal context across bins, so we feed it
non-overlapping sequences of SEQ_LEN bins.  The hidden state at the last
timestep predicts the 12 controller outputs.

Architecture:  GRU(input=64, hidden=128, layers=2, dropout=0.1) → Linear(128, 12)
Sweep:         W = 20, 30, 40, 50, 60, 70, 80, 100 ms  (non-overlapping bins)
Sequence:      SEQ_LEN = 15 bins (same temporal context as v4's 15-bin window)
Split:         80/20 temporal (no shuffle — no leakage)
Training:      100 epochs, Adam, MSE loss
Output:        analysis/bin_size_sweep_rnn.png  +  stdout summary table
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
SEQ_LEN      = 15
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
    X = d['X']
    Y = d['Y']
    feat_mean = d['feat_mean']
    feat_std  = d['feat_std']
    meta = {k: d[k].item() for k in ('bin_ms', 'spike_thresh', 'n_files',
                                      'total_seconds', 'sample_rate')}
    log(f"  Loaded bins_{bin_ms}ms.npz  —  {X.shape[0]:,} bins  "
        f"(thresh={meta['spike_thresh']}σ, {meta['n_files']} files, "
        f"{meta['total_seconds']:.0f}s data)")
    return X, Y, feat_mean, feat_std


def make_sequences(X, Y, seq_len):
    """
    Pack (n_bins, 64) and (n_bins, 12) into non-overlapping sequences.
    Returns X_seq (n_seq, seq_len, 64), Y_seq (n_seq, 12) — label at last bin.
    """
    n_seq = len(X) // seq_len
    X = X[:n_seq * seq_len]
    Y = Y[:n_seq * seq_len]
    X_seq = X.reshape(n_seq, seq_len, N_NEURAL)
    Y_seq = Y.reshape(n_seq, seq_len, N_LABELS)[:, -1, :]  # predict last bin's label
    return X_seq, Y_seq


# ── Model ─────────────────────────────────────────────────────────────────────

class GRUDecoder(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2,
                 dropout=0.1, output_size=12):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, 64)
        out, _ = self.gru(x)          # (batch, seq_len, hidden)
        return self.head(out[:, -1])  # (batch, 12)  — last timestep


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_eval(X, Y, bin_ms, feat_mean, feat_std):
    X_seq, Y_seq = make_sequences(X, Y, SEQ_LEN)

    split = int(len(X_seq) * 0.8)
    X_tr, X_val = X_seq[:split], X_seq[split:]
    Y_tr, Y_val = Y_seq[:split], Y_seq[split:]

    # Use precomputed normalisation stats (consistent with preprocessing)
    X_tr_n  = (X_tr  - feat_mean) / feat_std
    X_val_n = (X_val - feat_mean) / feat_std

    tr_dl  = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(Y_tr)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_val_n), torch.from_numpy(Y_val)),
        batch_size=BATCH_SIZE, shuffle=False)

    model   = GRUDecoder().to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_r2    = -999.0
    best_state = None

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

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
            best_r2    = r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

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

    output_names = ['LStX','LStY','RStX','RStY','A','B','X','Y','LB','RB','LT','RT']
    detail = '  '.join(f'{n}={r:.3f}' for n, r in zip(output_names, r2_per))
    log(f"    per-output R²: {detail}")

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return mean_r2, mean_rmse, r2_per, rmse_per


# ── Elbow detection ───────────────────────────────────────────────────────────

def find_elbow(values):
    v = np.array(values, dtype=float)
    v_norm = (v - v.min()) / (np.ptp(v) + 1e-12)
    x_norm = np.linspace(0, 1, len(v))
    line_vec  = np.array([x_norm[-1] - x_norm[0], v_norm[-1] - v_norm[0]])
    line_unit = line_vec / (np.linalg.norm(line_vec) + 1e-12)
    dists = []
    for xi, yi in zip(x_norm, v_norm):
        pt   = np.array([xi - x_norm[0], yi - v_norm[0]])
        proj = np.dot(pt, line_unit) * line_unit
        dists.append(np.linalg.norm(pt - proj))
    return int(np.argmax(dists))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log(f"Device: {DEVICE}")
    log(f"GRU: input=64, hidden=128, layers=2  |  seq_len={SEQ_LEN} bins")

    results = []

    header = (f"\n{'Bin(ms)':>8}  {'Seqs':>7}  {'CtxWin':>8}  "
              f"{'MeanR²':>8}  {'MeanRMSE':>10}  {'Time':>6}")
    log(header)
    log('-' * len(header))

    for bin_ms in BIN_SIZES_MS:
        ctx_ms = bin_ms * SEQ_LEN
        t_bin  = time.time()

        X, Y, feat_mean, feat_std = load_bins(bin_ms)
        n_seq = len(X) // SEQ_LEN

        mean_r2, mean_rmse, r2_per, rmse_per = train_and_eval(X, Y, bin_ms, feat_mean, feat_std)
        elapsed = time.time() - t_bin

        log(f"{bin_ms:>8}  {n_seq:>7,}  {ctx_ms:>7}ms  "
            f"{mean_r2:>+8.4f}  {mean_rmse:>10.4f}  {elapsed:>5.1f}s")
        results.append((bin_ms, mean_r2, mean_rmse, r2_per, rmse_per, ctx_ms))

        del X, Y
        gc.collect()

    # ── Summary table ─────────────────────────────────────────────────────────
    output_names  = ['LStX','LStY','RStX','RStY','A','B','X','Y','LB','RB','LT','RT']
    best_r2_idx   = int(np.argmax([r[1] for r in results]))
    best_rmse_idx = int(np.argmin([r[2] for r in results]))

    log("\n" + "=" * 72)
    log("  BIN SIZE SWEEP (RNN/GRU) — FULL RESULTS")
    log(f"  seq_len={SEQ_LEN}  |  spike_thresh={SPIKE_THRESH}σ  |  "
        f"GRU hidden=128 layers=2")
    log("=" * 72)
    log(f"{'Bin':>6}  {'Seqs':>7}  {'CtxWin':>8}  {'MeanR²':>8}  {'MeanRMSE':>10}")
    log("-" * 46)
    for i, (bms, r2, rmse, _, _, ctx) in enumerate(results):
        n_seq = int(len(neural) / (SAMPLE_RATE * bms / 1000)) // SEQ_LEN
        marker = " ← best R²" if i == best_r2_idx else (
                 " ← best RMSE" if i == best_rmse_idx else "")
        log(f"{bms:>6}  {n_seq:>7,}  {ctx:>7}ms  {r2:>+8.4f}  {rmse:>10.4f}{marker}")

    log(f"\nPer-output R² at best bin size ({results[best_r2_idx][0]}ms, "
        f"ctx={results[best_r2_idx][5]}ms):")
    for name, r2 in zip(output_names, results[best_r2_idx][3]):
        bar = '█' * max(0, int((r2 + 0.1) * 20))
        log(f"  {name:>5}: {r2:+.4f}  {bar}")
    log("=" * 72)
    log(f"Total time: {time.time()-t0:.1f}s")

    # ── Plot ──────────────────────────────────────────────────────────────────
    bin_sizes  = [r[0] for r in results]
    mean_r2s   = [r[1] for r in results]
    mean_rmses = [r[2] for r in results]
    ctx_wins   = [r[5] for r in results]

    elbow_r2   = find_elbow(mean_r2s)
    elbow_rmse = find_elbow([-x for x in mean_rmses])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(bin_sizes, mean_r2s, 'o-', color='steelblue', lw=2, ms=7)
    ax1.axvline(bin_sizes[elbow_r2], color='tomato', ls='--', lw=1.5,
                label=f'Elbow: {bin_sizes[elbow_r2]}ms bin ({ctx_wins[elbow_r2]}ms ctx)')
    ax1.scatter([bin_sizes[best_r2_idx]], [mean_r2s[best_r2_idx]],
                s=120, zorder=5, color='gold', edgecolors='k', lw=1.2,
                label=f'Best: {bin_sizes[best_r2_idx]}ms (R²={mean_r2s[best_r2_idx]:+.3f})')
    for x, y in zip(bin_sizes, mean_r2s):
        ax1.annotate(f'{y:+.3f}', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax1.set_xlabel('Bin size (ms)'); ax1.set_ylabel('Mean R²  (12 outputs)')
    ax1.set_title('Mean R² vs Bin Size  [GRU]')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.plot(bin_sizes, mean_rmses, 's-', color='darkorange', lw=2, ms=7)
    ax2.axvline(bin_sizes[elbow_rmse], color='tomato', ls='--', lw=1.5,
                label=f'Elbow: {bin_sizes[elbow_rmse]}ms bin ({ctx_wins[elbow_rmse]}ms ctx)')
    ax2.scatter([bin_sizes[best_rmse_idx]], [mean_rmses[best_rmse_idx]],
                s=120, zorder=5, color='gold', edgecolors='k', lw=1.2,
                label=f'Best: {bin_sizes[best_rmse_idx]}ms (RMSE={mean_rmses[best_rmse_idx]:.3f})')
    for x, y in zip(bin_sizes, mean_rmses):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax2.set_xlabel('Bin size (ms)'); ax2.set_ylabel('Mean RMSE  (12 outputs)')
    ax2.set_title('Mean RMSE vs Bin Size  [GRU]')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f'Bin Size Sweep — GRU Decoder  (seq_len={SEQ_LEN} bins)\n'
        f'64 neural channels  |  GRU hidden=128 layers=2  |  '
        f'spike thresh={SPIKE_THRESH}σ  |  non-overlapping bins',
        fontsize=11)
    plt.tight_layout()
    plt.savefig('analysis/bin_size_sweep_rnn.png', dpi=150)
    plt.close()
    log("Saved: analysis/bin_size_sweep_rnn.png")


if __name__ == '__main__':
    main()
