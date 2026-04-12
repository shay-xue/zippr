"""
Post-hoc analysis of decoder_rnn_shay results.

Loads model_meta.json and feat_norm.npz, reloads val data,
runs inference with the saved checkpoint, and reports per-channel
R² and RMSE across all 12 controller outputs.
"""

import h5py
import numpy as np
import glob
import os
import json
import gc
import torch
import torch.nn as nn
from scipy.signal import butter, sosfilt
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAMPLE_RATE = 32_000
N_NEURAL    = 64
N_LABELS    = 12
BIN_MS      = 10
BIN_SAMPLES = int(SAMPLE_RATE * BIN_MS / 1000)
N_FEATURES  = 192
OUT_DIR     = 'analysis/decoder_rnn_shay'
RECORDINGS  = 'data/recordings'
DEVICE      = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

LABEL_NAMES = ['LStX', 'LStY', 'RStX', 'RStY', 'A', 'B', 'X', 'Y', 'LB', 'RB', 'LT', 'RT']
LABEL_TYPES = ['joy', 'joy', 'joy', 'joy',
               'btn', 'btn', 'btn', 'btn', 'btn', 'btn',
               'trig', 'trig']
TYPE_COLORS = {'joy': 'steelblue', 'btn': 'tomato', 'trig': 'darkorange'}


def log(msg):
    print(msg, flush=True)


# ── Reuse preprocessing from v7 ───────────────────────────────────────────────

def bandpass_filter(neural):
    sos = butter(2, [200, 5000], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    return sosfilt(sos, neural, axis=0).astype(np.float32)


def extract_features_track_a(neural_filtered):
    ch_std  = neural_filtered.std(0) + 1e-8
    n_bins  = len(neural_filtered) // BIN_SAMPLES
    neural_tr = neural_filtered[:n_bins * BIN_SAMPLES]
    parts = []
    for sigma in [3.0, 4.0, 5.0]:
        thresh = sigma * ch_std
        spikes = ((neural_tr > thresh) | (neural_tr < -thresh)).astype(np.float32)
        parts.append(spikes.reshape(n_bins, BIN_SAMPLES, N_NEURAL).sum(1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def load_file(path):
    with h5py.File(path, 'r') as hf:
        raw = hf['acquisition/ElectricalSeries'][:]
    n = len(raw) // 76
    d = raw[:n * 76].reshape(n, 76).astype(np.float32)
    neural, labels = d[:, :N_NEURAL], d[:, N_NEURAL:N_NEURAL + N_LABELS]
    del raw, d
    neural_filt = bandpass_filter(neural)
    features    = extract_features_track_a(neural_filt)
    n_bins      = len(features)
    label_bins  = labels[:n_bins * BIN_SAMPLES].reshape(
        n_bins, BIN_SAMPLES, N_LABELS).mean(1)
    del neural, neural_filt
    gc.collect()
    return features, label_bins.astype(np.float32)


# ── Model (must match train_decoder_v7.py) ────────────────────────────────────

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


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, feats_norm, seq_len=30):
    """Slide non-overlapping windows over a single file's features."""
    model.eval()
    preds_joy, preds_trig, preds_gate = [], [], []
    n_bins = len(feats_norm)
    n_seq  = n_bins // seq_len

    x = torch.from_numpy(
        feats_norm[:n_seq * seq_len].reshape(n_seq, seq_len, N_FEATURES)
    ).to(DEVICE)

    with torch.no_grad():
        for i in range(0, n_seq, 64):
            j, t, g = model(x[i:i+64])
            preds_joy.append(j.cpu().numpy())
            preds_trig.append(t.cpu().numpy())
            preds_gate.append(g.cpu().numpy())

    return (np.concatenate(preds_joy),
            np.concatenate(preds_trig),
            np.concatenate(preds_gate),
            n_seq)


def analyze_label_distributions(all_files, val_stems):
    """
    Load raw labels from every file, compute per-axis statistics, and plot
    distributions split by train vs val set.  Saves label_distribution.png.
    """
    log("\n── Label Distribution Analysis ─────────────────────────────────────────")

    train_labels, val_labels = [], []
    file_stats = []   # (stem, split, axis_range, pct_active)

    for f in all_files:
        stem = os.path.basename(os.path.dirname(f)).replace('.h5', '')
        split = 'val' if stem in val_stems else 'train'
        try:
            with h5py.File(f, 'r') as hf:
                raw = hf['acquisition/ElectricalSeries'][:]
            n = len(raw) // 76
            d   = raw[:n * 76].reshape(n, 76).astype(np.float32)
            lbl = d[:, N_NEURAL:N_NEURAL + N_LABELS] / 32767.0
            axes_of_interest = lbl[:, [0, 1, 2, 3, 10, 11]]
            axis_range  = axes_of_interest.max(0) - axes_of_interest.min(0)
            pct_active  = (np.abs(axes_of_interest) > 0.05).mean(0)
            file_stats.append((stem, split, axis_range, pct_active))
            if split == 'val':
                val_labels.append(lbl)
            else:
                train_labels.append(lbl)
            del raw, d, lbl
        except Exception as e:
            log(f"  WARN: could not load {f}: {e}")

    train_all = np.concatenate(train_labels, axis=0) if train_labels else np.zeros((0, N_LABELS))
    val_all   = np.concatenate(val_labels,   axis=0) if val_labels   else np.zeros((0, N_LABELS))

    axis_names = ['LStX', 'LStY', 'RStX', 'RStY', 'LT', 'RT']
    axis_cols  = [0, 1, 2, 3, 10, 11]

    # ── Print per-axis stats ──────────────────────────────────────────────────
    log(f"\n{'Axis':<6}  {'Split':<6}  {'Mean':>7}  {'Std':>7}  {'Range':>7}  {'%Active':>8}")
    log("-" * 50)
    for name, col in zip(axis_names, axis_cols):
        for split_name, arr in [('train', train_all), ('val', val_all)]:
            if len(arr) == 0:
                continue
            v = arr[:, col]
            pct = float((np.abs(v) > 0.05).mean() * 100)
            log(f"{name:<6}  {split_name:<6}  {v.mean():>+7.3f}  {v.std():>7.3f}  "
                f"{(v.max()-v.min()):>7.3f}  {pct:>7.1f}%")

    # ── Flag LStY specifically ────────────────────────────────────────────────
    if len(train_all) > 0 and len(val_all) > 0:
        for split_name, arr in [('TRAIN', train_all), ('VAL', val_all)]:
            lsty = arr[:, 1]
            lstx = arr[:, 0]
            log(f"\n{split_name} LStY: mean={lsty.mean():+.4f}  std={lsty.std():.4f}  "
                f"range=[{lsty.min():.3f}, {lsty.max():.3f}]  "
                f"%active={100*(np.abs(lsty)>0.05).mean():.1f}%")
            log(f"{split_name} LStX: mean={lstx.mean():+.4f}  std={lstx.std():.4f}  "
                f"range=[{lstx.min():.3f}, {lstx.max():.3f}]  "
                f"%active={100*(np.abs(lstx)>0.05).mean():.1f}%")
        r_train = train_all[:, 0].std() / (train_all[:, 1].std() + 1e-8)
        r_val   = val_all[:, 0].std()   / (val_all[:, 1].std()   + 1e-8)
        log(f"\nLStX/LStY std ratio — train: {r_train:.2f}x  |  val: {r_val:.2f}x  "
            f"(1.0 = symmetric; >2 = LStY severely underrepresented)")

    # ── Plot 1: histograms for all 6 axes, train vs val ───────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    bins_hist = np.linspace(-1.05, 1.05, 60)
    for ax, name, col in zip(axes.flat, axis_names, axis_cols):
        tr = train_all[:, col] if len(train_all) else np.array([])
        vl = val_all[:,   col] if len(val_all)   else np.array([])
        if len(tr):
            ax.hist(tr, bins=bins_hist, alpha=0.55, color='steelblue',
                    density=True, label=f'Train (n={len(tr):,})')
        if len(vl):
            ax.hist(vl, bins=bins_hist, alpha=0.55, color='tomato',
                    density=True, label=f'Val   (n={len(vl):,})')
        ax.set_title(f'{name}   std(tr)={tr.std():.3f}  std(val)={vl.std():.3f}',
                     fontsize=9)
        ax.set_xlabel('Normalised value'); ax.set_ylabel('Density')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle('Label Distributions — Train vs Val\n'
                 '(LStY asymmetry vs LStX points to data coverage gap)',
                 fontsize=11)
    plt.tight_layout()
    dist_path = os.path.join(OUT_DIR, 'label_distribution.png')
    plt.savefig(dist_path, dpi=150)
    plt.close()
    log(f"\nSaved: {dist_path}")

    # ── Plot 2: per-file % active for LStX vs LStY ───────────────────────────
    from matplotlib.patches import Patch
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    for ax, col_idx, title in [(ax1, 0, 'LStX'), (ax2, 1, 'LStY')]:
        train_pcts = [(s[0], s[3][col_idx] * 100) for s in file_stats if s[1] == 'train']
        val_pcts   = [(s[0], s[3][col_idx] * 100) for s in file_stats if s[1] == 'val']
        all_pcts   = train_pcts + val_pcts
        names_     = [x[0] for x in all_pcts]
        vals_      = [x[1] for x in all_pcts]
        colors_    = ['steelblue'] * len(train_pcts) + ['tomato'] * len(val_pcts)
        ax.bar(range(len(names_)), vals_, color=colors_, alpha=0.8)
        ax.set_xticks(range(len(names_)))
        ax.set_xticklabels([n.replace('train_', '') for n in names_],
                           rotation=60, ha='right', fontsize=7)
        ax.set_ylabel('% time steps with |value| > 0.05')
        ax.set_title(f'{title} — Activity per File')
        mean_val = float(np.mean(vals_))
        ax.axhline(mean_val, color='k', ls='--', lw=1)
        ax.legend(handles=[Patch(color='steelblue', label='Train'),
                            Patch(color='tomato',    label='Val'),
                            plt.Line2D([0],[0], color='k', ls='--',
                                       label=f'Mean {mean_val:.1f}%')],
                  fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Per-File Joystick Activity: LStX vs LStY\n'
                 'Low LStY activity in val files → artificially low val R²',
                 fontsize=11)
    plt.tight_layout()
    activity_path = os.path.join(OUT_DIR, 'lsty_activity_per_file.png')
    plt.savefig(activity_path, dpi=150)
    plt.close()
    log(f"Saved: {activity_path}")

    # ── Plot 3: scatter LStX vs LStY (train and val) ─────────────────────────
    fig, plot_axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, arr, title, color in [
            (plot_axes[0], train_all, 'Train', 'steelblue'),
            (plot_axes[1], val_all,   'Val',   'tomato')]:
        if len(arr) == 0:
            continue
        idx = np.random.choice(len(arr), min(5000, len(arr)), replace=False)
        ax.scatter(arr[idx, 0], arr[idx, 1], alpha=0.15, s=2, color=color)
        ax.set_xlabel('LStX'); ax.set_ylabel('LStY')
        ax.set_title(f'{title}\nstd(X)={arr[:,0].std():.3f}  std(Y)={arr[:,1].std():.3f}')
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.grid(alpha=0.2); ax.set_aspect('equal')
    plt.suptitle('LStX vs LStY Scatter — Are movements isotropic?', fontsize=11)
    plt.tight_layout()
    scatter_path = os.path.join(OUT_DIR, 'lstxy_scatter.png')
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    log(f"Saved: {scatter_path}")

    log("── Distribution analysis complete ──────────────────────────────────────\n")


def main():
    # ── Load meta ─────────────────────────────────────────────────────────────
    meta_path = os.path.join(OUT_DIR, 'model_meta.json')
    ckpt_path = os.path.join(OUT_DIR, 'best_model.pt')
    norm_path = os.path.join(OUT_DIR, 'feat_norm.npz')

    for p in [meta_path, ckpt_path, norm_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p} — run train_decoder_v7.py first")

    with open(meta_path) as f:
        meta = json.load(f)
    log(f"Model meta: joy_r²={meta['joy_r2_avg']:+.4f}  "
        f"trig_r²={meta['trig_r2_avg']:+.4f}  gate_acc={meta['gate_acc']:.4f}")

    norm = np.load(norm_path)
    feat_mean = norm['feat_mean'].reshape(1, -1).astype(np.float32)
    feat_std  = norm['feat_std'].reshape(1, -1).astype(np.float32)

    # ── Load model ────────────────────────────────────────────────────────────
    model = GRUDecoder().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    log(f"Loaded checkpoint: {ckpt_path}")

    # ── Val files (same split as training) ────────────────────────────────────
    all_files = sorted(glob.glob(f'{RECORDINGS}/train_*/broadband_data_*.h5'))
    val_stems = {'train_004_bothsticks', 'train_011_lb_spam', 'train_015_ab_alt',
                 'train_021_slow_all',   'train_023_lstickx_sweep', 'train_001_rest',
                 'train_020_rstick_bumptrig', 'train_029_lt_analog'}
    val_files = [f for f in all_files
                 if os.path.basename(os.path.dirname(f)).replace('.h5', '') in val_stems]
    log(f"Val files: {len(val_files)}")

    # ── Label distribution analysis (all files) ───────────────────────────────
    analyze_label_distributions(all_files, val_stems)

    # ── Run inference on all val files ────────────────────────────────────────
    all_joy_pred, all_joy_true = [], []
    all_trig_pred, all_trig_true = [], []

    for f in val_files:
        feats, labels = load_file(f)
        feats_norm = (feats - feat_mean) / feat_std
        joy_p, trig_p, gate_p, n_seq = run_inference(model, feats_norm, seq_len=30)

        # Ground truth aligned to non-overlapping sequences (last bin of each)
        gt_idx   = np.arange(29, n_seq * 30, 30)
        joy_true  = np.clip(labels[gt_idx, :4]   / 32767.0, -1.0, 1.0)
        trig_true = np.clip(labels[gt_idx, 10:12] / 32767.0,  0.0, 1.0)

        all_joy_pred.append(joy_p);   all_joy_true.append(joy_true)
        all_trig_pred.append(trig_p); all_trig_true.append(trig_true)
        del feats, labels, feats_norm; gc.collect()

    jp = np.concatenate(all_joy_pred);  jt = np.concatenate(all_joy_true)
    tp = np.concatenate(all_trig_pred); tt = np.concatenate(all_trig_true)

    # ── Per-channel R² and RMSE ───────────────────────────────────────────────
    joy_names  = ['LStX', 'LStY', 'RStX', 'RStY']
    trig_names = ['LT',   'RT']

    r2_all   = {}
    rmse_all = {}

    for i, name in enumerate(joy_names):
        r2_all[name]   = r2_score(jt[:, i], jp[:, i])
        rmse_all[name] = float(np.sqrt(np.mean((jt[:, i] - jp[:, i])**2)))

    for i, name in enumerate(trig_names):
        r2_all[name]   = r2_score(tt[:, i], tp[:, i])
        rmse_all[name] = float(np.sqrt(np.mean((tt[:, i] - tp[:, i])**2)))

    # Buttons — model doesn't predict them directly; report as N/A
    for name in ['A', 'B', 'X', 'Y', 'LB', 'RB']:
        r2_all[name]   = None
        rmse_all[name] = None

    # ── Print report ──────────────────────────────────────────────────────────
    log(f"\n{'='*58}")
    log(f"  DECODER RNN (v7) — PER-CHANNEL R² & RMSE ANALYSIS")
    log(f"{'='*58}")
    log(f"{'Channel':<8}  {'Type':<6}  {'R²':>8}  {'RMSE':>8}  {'Bar'}")
    log(f"{'-'*58}")

    predicted_r2s   = []
    predicted_rmses = []

    for name in LABEL_NAMES:
        ltype = LABEL_TYPES[LABEL_NAMES.index(name)]
        r2    = r2_all[name]
        rmse  = rmse_all[name]
        if r2 is None:
            log(f"{name:<8}  {ltype:<6}  {'N/A':>8}  {'N/A':>8}  (not predicted by model)")
        else:
            bar = '█' * max(0, int((r2 + 0.2) * 15))
            log(f"{name:<8}  {ltype:<6}  {r2:>+8.4f}  {rmse:>8.4f}  {bar}")
            predicted_r2s.append(r2)
            predicted_rmses.append(rmse)

    log(f"{'-'*58}")
    joy_r2s   = [r2_all[n] for n in joy_names]
    trig_r2s  = [r2_all[n] for n in trig_names]
    joy_rmses  = [rmse_all[n] for n in joy_names]
    trig_rmses = [rmse_all[n] for n in trig_names]
    log(f"{'Joy avg':<8}  {'joy':<6}  {np.mean(joy_r2s):>+8.4f}  "
        f"{np.mean(joy_rmses):>8.4f}")
    log(f"{'Trig avg':<8}  {'trig':<6}  {np.mean(trig_r2s):>+8.4f}  "
        f"{np.mean(trig_rmses):>8.4f}")
    log(f"{'Overall':<8}  {'all':<6}  {np.mean(predicted_r2s):>+8.4f}  "
        f"{np.mean(predicted_rmses):>8.4f}")
    log(f"{'='*58}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    pred_names  = joy_names + trig_names
    pred_r2s    = joy_r2s + trig_r2s
    pred_rmses  = joy_rmses + trig_rmses
    colors      = [TYPE_COLORS[LABEL_TYPES[LABEL_NAMES.index(n)]] for n in pred_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # R² per channel
    bars = ax1.bar(pred_names, pred_r2s, color=colors, alpha=0.85)
    ax1.axhline(0, color='k', lw=0.8)
    ax1.axhline(np.mean(joy_r2s), color='steelblue', ls='--', lw=1.2,
                label=f'Joy avg: {np.mean(joy_r2s):+.3f}')
    ax1.axhline(np.mean(trig_r2s), color='darkorange', ls='--', lw=1.2,
                label=f'Trig avg: {np.mean(trig_r2s):+.3f}')
    for bar, r2 in zip(bars, pred_r2s):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.01 if r2 >= 0 else -0.04),
                 f'{r2:+.3f}', ha='center', va='bottom', fontsize=9)
    ax1.set_ylabel('R²'); ax1.set_title('Per-Channel R²')
    ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)

    # RMSE per channel
    bars2 = ax2.bar(pred_names, pred_rmses, color=colors, alpha=0.85)
    ax2.axhline(np.mean(joy_rmses), color='steelblue', ls='--', lw=1.2,
                label=f'Joy avg: {np.mean(joy_rmses):.3f}')
    ax2.axhline(np.mean(trig_rmses), color='darkorange', ls='--', lw=1.2,
                label=f'Trig avg: {np.mean(trig_rmses):.3f}')
    for bar, rmse in zip(bars2, pred_rmses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{rmse:.3f}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('RMSE'); ax2.set_title('Per-Channel RMSE')
    ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Decoder RNN (v7) — Per-Channel Accuracy Analysis\n'
                 'GRU · 300ms context · Track A features · Stratified val split',
                 fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'per_channel_analysis.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    log(f"\nSaved: {out_path}")

    # Prediction traces for each predicted channel
    n_plot  = min(200, len(jp))
    t_axis  = np.arange(n_plot) * 30 * BIN_MS / 1000

    fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True)
    for i, (ax, name) in enumerate(zip(axes[:4], joy_names)):
        ax.plot(t_axis, jt[:n_plot, i], 'b-', alpha=0.6, lw=0.8, label='True')
        ax.plot(t_axis, jp[:n_plot, i], 'r-', alpha=0.6, lw=0.8, label='Pred')
        ax.set_title(f'{name}  R²={r2_all[name]:+.4f}  RMSE={rmse_all[name]:.4f}')
        ax.legend(loc='upper right', fontsize=7); ax.set_ylabel('Norm.')
    for i, (ax, name) in enumerate(zip(axes[4:], trig_names)):
        ax.plot(t_axis, tt[:n_plot, i], 'b-', alpha=0.6, lw=0.8, label='True')
        ax.plot(t_axis, tp[:n_plot, i], 'r-', alpha=0.6, lw=0.8, label='Pred')
        ax.set_title(f'{name}  R²={r2_all[name]:+.4f}  RMSE={rmse_all[name]:.4f}')
        ax.legend(loc='upper right', fontsize=7); ax.set_ylabel('Norm.')
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('Decoder RNN (v7) — Prediction Traces (val set)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'prediction_traces.png'), dpi=150)
    plt.close()
    log(f"Saved: {os.path.join(OUT_DIR, 'prediction_traces.png')}")
    log("Analysis complete.")


if __name__ == '__main__':
    main()
