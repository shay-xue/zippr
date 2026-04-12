"""
Bin Size Sweep — v7 GRU with full preprocessing pipeline.

Same sweep as bin_size_sweep_rnn.py but using v7's preprocessing:
  - Bandpass filter 200–5000 Hz (2nd-order Butterworth)
  - Spike counts at 3σ/4σ/5σ → 192 features/bin (vs 64 in the plain sweep)
  - Stratified file-based split (same as v7)
  - GRU(input=192, hidden=192, layers=2) — identical to v7

For each bin size W in {20, 30, 40, 50, 60, 70, 80, 100} ms:
  - SEQ_LEN = 15 bins (context window = W × 15 ms)
  - 80/20 temporal split within each file, sequences don't cross file boundaries
  - 50 epochs, AdamW lr=5e-4, MSE loss, early stop patience=15

Output: analysis/decoder_rnn_shay/bin_sweep/
  bin_sweep.png     — mean R² and RMSE vs bin size
  results.txt       — full summary table
"""

import h5py
import numpy as np
import glob
import os
import gc
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, sosfilt
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAMPLE_RATE  = 32_000
N_NEURAL     = 64
N_LABELS     = 12
N_FEATURES   = 192          # 64ch × 3 thresholds
SEQ_LEN      = 15           # bins per sequence
BIN_SIZES_MS = [20, 30, 40, 50, 60, 70, 80, 100]
EPOCHS       = 50
PATIENCE     = 15
BATCH_SIZE   = 128
LR           = 5e-4
RECORDINGS   = 'data/recordings'
OUT_DIR      = 'analysis/decoder_rnn_shay/bin_sweep'
DEVICE       = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

JOY_NAMES  = ['LStX', 'LStY', 'RStX', 'RStY']
TRIG_NAMES = ['LT', 'RT']


def log(msg):
    print(msg, flush=True)


# ── Preprocessing (v7 pipeline) ───────────────────────────────────────────────

def bandpass_filter(neural):
    sos = butter(2, [200, 5000], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    return sosfilt(sos, neural, axis=0).astype(np.float32)


def extract_features(neural_filtered, bin_samples):
    """Spike counts at 3σ/4σ/5σ → (n_bins, 192)."""
    ch_std = neural_filtered.std(0) + 1e-8
    n_bins = len(neural_filtered) // bin_samples
    neural_tr = neural_filtered[:n_bins * bin_samples]
    parts = []
    for sigma in [3.0, 4.0, 5.0]:
        thresh = sigma * ch_std
        spikes = ((neural_tr > thresh) | (neural_tr < -thresh)).astype(np.float32)
        parts.append(spikes.reshape(n_bins, bin_samples, N_NEURAL).sum(1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def load_file(path, bin_samples):
    with h5py.File(path, 'r') as hf:
        raw = hf['acquisition/ElectricalSeries'][:]
    n = len(raw) // 76
    d = raw[:n * 76].reshape(n, 76).astype(np.float32)
    neural = d[:, :N_NEURAL]
    labels = d[:, N_NEURAL:N_NEURAL + N_LABELS]
    del raw, d
    neural_filt = bandpass_filter(neural)
    del neural
    features   = extract_features(neural_filt, bin_samples)
    del neural_filt
    n_bins     = len(features)
    label_bins = labels[:n_bins * bin_samples].reshape(
        n_bins, bin_samples, N_LABELS).mean(1).astype(np.float32)
    gc.collect()
    return features, label_bins


def get_stratified_split(files):
    categories = {
        'rest':       ['rest'],
        'lstick':     ['lstick', 'lstickx', 'lsticky', 'lstick_diag',
                       'lstick_buttons', 'lstick_triggers'],
        'rstick':     ['rstick', 'rstickx', 'rsticky', 'rstick_diag',
                       'rstick_bumpers', 'rstick_bumptrig'],
        'buttons':    ['buttons', 'ab_alt', 'xy_alt'],
        'bumpers':    ['bumpers', 'lb_spam', 'rb_spam'],
        'triggers':   ['lt_spam', 'rt_spam', 'lt_analog', 'rt_analog'],
        'freestyle':  ['freestyle', 'slow_all', 'robot_sim'],
        'bothsticks': ['bothsticks'],
    }

    def get_category(path):
        stem = os.path.basename(os.path.dirname(path)).replace('.h5', '').split('_', 2)[-1]
        for cat, keywords in categories.items():
            if any(stem.startswith(kw) or kw in stem for kw in keywords):
                return cat
        return 'other'

    by_cat = {}
    for f in files:
        by_cat.setdefault(get_category(f), []).append(f)

    val_files, train_files = [], []
    for cat, flist in sorted(by_cat.items()):
        vi = len(flist) // 2
        val_files.append(flist[vi])
        train_files.extend(f for i, f in enumerate(flist) if i != vi)
    return train_files, val_files


# ── Dataset ───────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, feats_list, labels_list, seq_len, stride, feat_mean, feat_std):
        self.seq_len = seq_len
        fm = feat_mean.reshape(1, -1)
        fs = feat_std.reshape(1, -1)
        self.feats  = [(f - fm) / fs for f in feats_list]
        # Labels: joy (0-3 /32767 → [-1,1]), trig (10-11 /32767 → [0,1]), gate (4-9)
        processed = []
        for lbl in labels_list:
            joy  = np.clip(lbl[:, :4]   / 32767.0, -1.0, 1.0)
            trig = np.clip(lbl[:, 10:12] / 32767.0,  0.0, 1.0)
            btn  = (lbl[:, 4:10] > 16000).astype(np.float32)
            gate = (btn.sum(1) == 0).astype(np.float32).reshape(-1, 1)
            processed.append(np.concatenate([joy, trig, gate], axis=1))
        self.labels = processed

        self.index_map = []
        for fi, feat in enumerate(self.feats):
            n = len(feat)
            if n < seq_len:
                continue
            for s in range(0, n - seq_len + 1, stride):
                self.index_map.append((fi, s))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fi, s = self.index_map[idx]
        x = self.feats[fi][s:s + self.seq_len]
        y = self.labels[fi][s + self.seq_len - 1]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


# ── Model ─────────────────────────────────────────────────────────────────────

class GRUDecoder(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_size=192,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = (nn.Linear(input_size, hidden_size)
                           if input_size != hidden_size else nn.Identity())
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.joy_head  = nn.Sequential(nn.Linear(hidden_size, 64), nn.GELU(),
                                       nn.Linear(64, 4),  nn.Tanh())
        self.trig_head = nn.Sequential(nn.Linear(hidden_size, 32), nn.GELU(),
                                       nn.Linear(32, 2),  nn.Sigmoid())
        self.gate_head = nn.Sequential(nn.Linear(hidden_size, 32), nn.GELU(),
                                       nn.Linear(32, 1),  nn.Sigmoid())

    def forward(self, x):
        x = self.input_proj(x)
        _, h_n = self.gru(x)
        z = h_n[-1]
        return self.joy_head(z), self.trig_head(z), self.gate_head(z)


# ── Train & eval one bin size ──────────────────────────────────────────────────

def train_and_eval(feats_tr, labels_tr, feats_val, labels_val, bin_ms):
    all_tr    = np.concatenate(feats_tr, axis=0)
    feat_mean = all_tr.mean(0).astype(np.float32)
    feat_std  = (all_tr.std(0) + 1e-8).astype(np.float32)
    del all_tr; gc.collect()

    tr_ds  = SequenceDataset(feats_tr,  labels_tr,  SEQ_LEN, stride=1,
                             feat_mean=feat_mean, feat_std=feat_std)
    val_ds = SequenceDataset(feats_val, labels_val, SEQ_LEN, stride=SEQ_LEN,
                             feat_mean=feat_mean, feat_std=feat_std)

    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=256,        shuffle=False)

    model   = GRUDecoder().to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    mse_fn  = nn.MSELoss()
    bce_fn  = nn.BCELoss()

    best_r2    = -999.0
    best_state = None
    no_imp     = 0

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(DEVICE) + 0.05 * torch.randn_like(xb.to(DEVICE))
            joy_b  = yb[:, :4].to(DEVICE)
            trig_b = yb[:, 4:6].to(DEVICE)
            gate_b = yb[:, 6:7].to(DEVICE)
            joy_p, trig_p, gate_p = model(xb)
            loss = mse_fn(joy_p, joy_b) + mse_fn(trig_p, trig_b) + \
                   0.5 * bce_fn(gate_p, gate_b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        jp_list, jt_list, tp_list, tt_list = [], [], [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                joy_p, trig_p, _ = model(xb.to(DEVICE))
                jp_list.append(joy_p.cpu().numpy())
                jt_list.append(yb[:, :4].numpy())
                tp_list.append(trig_p.cpu().numpy())
                tt_list.append(yb[:, 4:6].numpy())

        jp = np.concatenate(jp_list); jt = np.concatenate(jt_list)
        tp = np.concatenate(tp_list); tt = np.concatenate(tt_list)
        joy_r2 = r2_score(jt, jp, multioutput='uniform_average')

        if joy_r2 > best_r2:
            best_r2    = joy_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break

    # Final eval with best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    jp_list, jt_list, tp_list, tt_list = [], [], [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            joy_p, trig_p, _ = model(xb.to(DEVICE))
            jp_list.append(joy_p.cpu().numpy())
            jt_list.append(yb[:, :4].numpy())
            tp_list.append(trig_p.cpu().numpy())
            tt_list.append(yb[:, 4:6].numpy())

    jp = np.concatenate(jp_list); jt = np.concatenate(jt_list)
    tp = np.concatenate(tp_list); tt = np.concatenate(tt_list)

    joy_r2_per   = [r2_score(jt[:, i], jp[:, i]) for i in range(4)]
    trig_r2_per  = [r2_score(tt[:, i], tp[:, i]) for i in range(2)]
    joy_rmse_per = [float(np.sqrt(np.mean((jt[:, i] - jp[:, i])**2))) for i in range(4)]
    trig_rmse_per= [float(np.sqrt(np.mean((tt[:, i] - tp[:, i])**2))) for i in range(2)]

    mean_joy_r2   = float(np.mean(joy_r2_per))
    mean_trig_r2  = float(np.mean(trig_r2_per))
    mean_r2       = float(np.mean(joy_r2_per + trig_r2_per))
    mean_rmse     = float(np.mean(joy_rmse_per + trig_rmse_per))

    del model; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    per_joy  = '  '.join(f'{n}={r:.3f}' for n, r in zip(JOY_NAMES,  joy_r2_per))
    per_trig = '  '.join(f'{n}={r:.3f}' for n, r in zip(TRIG_NAMES, trig_r2_per))
    log(f"    joy:  {per_joy}")
    log(f"    trig: {per_trig}")

    return mean_r2, mean_rmse, mean_joy_r2, mean_trig_r2, joy_r2_per, trig_r2_per


# ── Elbow detection ───────────────────────────────────────────────────────────

def find_elbow(values):
    v      = np.array(values, dtype=float)
    v_norm = (v - v.min()) / (np.ptp(v) + 1e-12)
    x_norm = np.linspace(0, 1, len(v))
    lv     = np.array([x_norm[-1] - x_norm[0], v_norm[-1] - v_norm[0]])
    lu     = lv / (np.linalg.norm(lv) + 1e-12)
    dists  = []
    for xi, yi in zip(x_norm, v_norm):
        pt   = np.array([xi - x_norm[0], yi - v_norm[0]])
        proj = np.dot(pt, lu) * lu
        dists.append(np.linalg.norm(pt - proj))
    return int(np.argmax(dists))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    log(f"Device: {DEVICE}")
    log(f"Pipeline: bandpass 200-5000Hz → spike counts 3/4/5σ → 192 features/bin")
    log(f"GRU: input=192, hidden=192, layers=2  |  seq_len={SEQ_LEN} bins\n")

    files = sorted(glob.glob(f'{RECORDINGS}/train_*/broadband_data_*.h5'))
    train_files, val_files = get_stratified_split(files)
    log(f"Train: {len(train_files)} files  |  Val: {len(val_files)} files\n")

    results = []
    header = (f"\n{'Bin(ms)':>8}  {'CtxWin':>8}  {'MeanR²':>8}  "
              f"{'JoyR²':>8}  {'TrigR²':>8}  {'RMSE':>8}  {'Time':>6}")
    log(header)
    log('-' * len(header))

    for bin_ms in BIN_SIZES_MS:
        bin_samples = int(SAMPLE_RATE * bin_ms / 1000)
        ctx_ms      = bin_ms * SEQ_LEN
        t_bin       = time.time()

        # Load features for this bin size
        feats_tr, labels_tr   = [], []
        feats_val, labels_val = [], []

        for f in train_files:
            feat, lbl = load_file(f, bin_samples)
            feats_tr.append(feat); labels_tr.append(lbl)

        for f in val_files:
            feat, lbl = load_file(f, bin_samples)
            feats_val.append(feat); labels_val.append(lbl)

        mean_r2, mean_rmse, joy_r2, trig_r2, joy_r2_per, trig_r2_per = \
            train_and_eval(feats_tr, labels_tr, feats_val, labels_val, bin_ms)

        elapsed = time.time() - t_bin
        log(f"{bin_ms:>8}  {ctx_ms:>7}ms  {mean_r2:>+8.4f}  "
            f"{joy_r2:>+8.4f}  {trig_r2:>+8.4f}  {mean_rmse:>8.4f}  {elapsed:>5.1f}s")
        results.append((bin_ms, ctx_ms, mean_r2, mean_rmse, joy_r2, trig_r2,
                         joy_r2_per, trig_r2_per))

        del feats_tr, labels_tr, feats_val, labels_val
        gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────────
    best_r2_idx   = int(np.argmax([r[2] for r in results]))
    best_rmse_idx = int(np.argmin([r[3] for r in results]))

    summary_lines = []
    summary_lines.append('=' * 70)
    summary_lines.append('  BIN SWEEP (v7 GRU) — FULL RESULTS')
    summary_lines.append(f'  Pipeline: bandpass + 3/4/5σ spike counts (192 feat)  '
                         f'|  seq_len={SEQ_LEN}')
    summary_lines.append('=' * 70)
    summary_lines.append(f"{'Bin':>6}  {'CtxWin':>8}  {'MeanR²':>8}  "
                         f"{'JoyR²':>8}  {'TrigR²':>8}  {'RMSE':>8}")
    summary_lines.append('-' * 56)
    for i, (bms, ctx, r2, rmse, joy, trig, _, _) in enumerate(results):
        marker = ' ← best R²'   if i == best_r2_idx   else \
                 ' ← best RMSE' if i == best_rmse_idx  else ''
        summary_lines.append(f"{bms:>6}  {ctx:>7}ms  {r2:>+8.4f}  "
                              f"{joy:>+8.4f}  {trig:>+8.4f}  {rmse:>8.4f}{marker}")

    best = results[best_r2_idx]
    summary_lines.append(f"\nPer-axis R² at best bin size ({best[0]}ms, ctx={best[1]}ms):")
    for name, r2 in zip(JOY_NAMES, best[6]):
        bar = '█' * max(0, int((r2 + 0.2) * 15))
        summary_lines.append(f"  {name:<8} {r2:+.4f}  {bar}")
    for name, r2 in zip(TRIG_NAMES, best[7]):
        bar = '█' * max(0, int((r2 + 0.2) * 15))
        summary_lines.append(f"  {name:<8} {r2:+.4f}  {bar}")
    summary_lines.append(f"\nTotal time: {time.time()-t0:.1f}s")
    summary_lines.append('=' * 70)

    report = '\n'.join(summary_lines)
    log('\n' + report)
    with open(os.path.join(OUT_DIR, 'results.txt'), 'w') as f:
        f.write(report + '\n')

    # ── Plot ──────────────────────────────────────────────────────────────────
    bin_sizes  = [r[0] for r in results]
    mean_r2s   = [r[2] for r in results]
    mean_rmses = [r[3] for r in results]
    joy_r2s    = [r[4] for r in results]
    trig_r2s   = [r[5] for r in results]
    ctx_wins   = [r[1] for r in results]

    elbow_r2   = find_elbow(mean_r2s)
    elbow_rmse = find_elbow([-x for x in mean_rmses])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(bin_sizes, mean_r2s,  'o-',  color='steelblue',   lw=2, ms=7, label='Mean R²')
    ax1.plot(bin_sizes, joy_r2s,   's--', color='royalblue',   lw=1.5, ms=5, alpha=0.7, label='Joy R²')
    ax1.plot(bin_sizes, trig_r2s,  '^--', color='darkorange',  lw=1.5, ms=5, alpha=0.7, label='Trig R²')
    ax1.axvline(bin_sizes[elbow_r2], color='tomato', ls=':', lw=1.5,
                label=f'Elbow: {bin_sizes[elbow_r2]}ms')
    ax1.scatter([bin_sizes[best_r2_idx]], [mean_r2s[best_r2_idx]],
                s=130, zorder=5, color='gold', edgecolors='k', lw=1.2,
                label=f'Best: {bin_sizes[best_r2_idx]}ms ({mean_r2s[best_r2_idx]:+.3f})')
    for x, y in zip(bin_sizes, mean_r2s):
        ax1.annotate(f'{y:+.3f}', (x, y), textcoords='offset points',
                     xytext=(0, 9), ha='center', fontsize=8)
    ax1.axhline(0, color='k', lw=0.6)
    ax1.set_xlabel('Bin size (ms)'); ax1.set_ylabel('R²')
    ax1.set_title('R² vs Bin Size  [v7 GRU]')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.plot(bin_sizes, mean_rmses, 's-', color='darkorange', lw=2, ms=7)
    ax2.axvline(bin_sizes[elbow_rmse], color='tomato', ls=':', lw=1.5,
                label=f'Elbow: {bin_sizes[elbow_rmse]}ms')
    ax2.scatter([bin_sizes[best_rmse_idx]], [mean_rmses[best_rmse_idx]],
                s=130, zorder=5, color='gold', edgecolors='k', lw=1.2,
                label=f'Best: {bin_sizes[best_rmse_idx]}ms ({mean_rmses[best_rmse_idx]:.3f})')
    for x, y in zip(bin_sizes, mean_rmses):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                     xytext=(0, 9), ha='center', fontsize=8)
    ax2.set_xlabel('Bin size (ms)'); ax2.set_ylabel('Mean RMSE')
    ax2.set_title('RMSE vs Bin Size  [v7 GRU]')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f'Bin Size Sweep — v7 GRU  (seq_len={SEQ_LEN} bins)\n'
        f'Bandpass 200–5000 Hz  |  Spike counts 3/4/5σ  |  192 features/bin  |  '
        f'Stratified split',
        fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'bin_sweep.png'), dpi=150)
    plt.close()
    log(f"\nSaved: {OUT_DIR}/bin_sweep.png")
    log(f"Saved: {OUT_DIR}/results.txt")


if __name__ == '__main__':
    main()
