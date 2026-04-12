"""
Decoder PCA — PCA + linear readout baseline.

Pipeline:
  1. Load all hard-mode training HDF5s (train_001 … train_032)
  2. Extract per-bin spike-rate + voltage features  (448 dims, 10ms bins)
  3. File-based train/val split (every 5th file → val, no leakage)
  4. Fit PCA on training features, project both splits
  5. Ridge regression  → 4 joystick axes + 2 triggers
  6. Logistic regression → 6 buttons (A, B, X, Y, LB, RB)
  7. Report R² and RMSE for regression; accuracy + F1 for buttons
  8. Save PCA components + model params + diagnostic plots

Outputs in analysis/decoder_pca/:
  pca_model.npz       — components, mean, explained variance
  ridge_joy.npz       — joystick Ridge coefficients
  ridge_trig.npz      — trigger Ridge coefficients
  logreg_btn.npz      — button LogReg coefficients
  results.txt         — R² / RMSE / accuracy report
  pca_variance.png    — cumulative explained variance
  joystick.png        — prediction vs truth for each axis
  triggers.png        — prediction vs truth for LT/RT
  buttons.png         — per-button accuracy bar chart
"""

import h5py
import numpy as np
import glob
import os
import gc
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

SAMPLE_RATE   = 32_000
N_NEURAL      = 64
N_LABELS      = 12
BIN_MS        = 10
BIN_SAMPLES   = int(SAMPLE_RATE * BIN_MS / 1000)   # 320 samples
OUT_DIR       = 'analysis/decoder_pca'
RECORDINGS    = 'data/recordings'

JOYSTICK_NAMES = ['L-Stick X', 'L-Stick Y', 'R-Stick X', 'R-Stick Y']
TRIGGER_NAMES  = ['LT (open)', 'RT (close)']
BUTTON_NAMES   = ['A', 'B', 'X', 'Y', 'LB', 'RB']


def log(msg: str) -> None:
    print(msg, flush=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_h5(path: str):
    """Return (neural [N,64], labels [N,12]) float32 from one HDF5 file."""
    with h5py.File(path, 'r') as hf:
        raw = hf['acquisition/ElectricalSeries'][:]
    n = len(raw) // 76
    d = raw[:n * 76].reshape(n, 76)
    neural = d[:, :N_NEURAL].astype(np.float32)
    labels = d[:, N_NEURAL:N_NEURAL + N_LABELS].astype(np.float32)
    del raw, d
    return neural, labels


def extract_bin_features(neural: np.ndarray) -> np.ndarray:
    """
    Convert raw (N_samples, 64) → (N_bins, 448) feature matrix.

    Per 10ms bin, per channel:
      spike rates at 3σ, 4σ, 5σ above AND below mean (6)
      mean voltage (1)   std (1)   peak-to-peak (1)   |mean| (1)
    = 10 features × 64 channels = 640 dims

    Actually using: rates3, rates4, rates5 (3×64=192) + mean, std, pp, absmean (4×64=256)
    = 7 × 64 = 448
    """
    ch_mean = neural.mean(0)          # (64,)
    ch_std  = neural.std(0) + 1e-8   # (64,)

    n_bins = len(neural) // BIN_SAMPLES
    neural = neural[:n_bins * BIN_SAMPLES]
    binned = neural.reshape(n_bins, BIN_SAMPLES, N_NEURAL)  # (B, 320, 64)

    parts = []
    for thresh in [3.0, 4.0, 5.0]:
        above = neural > (ch_mean + thresh * ch_std)
        below = neural < (ch_mean - thresh * ch_std)
        spikes = (above | below).astype(np.float32)
        parts.append(spikes.reshape(n_bins, BIN_SAMPLES, N_NEURAL).sum(1))  # (B,64)

    parts.append(binned.mean(1))                      # mean voltage  (B,64)
    parts.append(binned.std(1))                       # std           (B,64)
    parts.append(binned.max(1) - binned.min(1))       # peak-to-peak  (B,64)
    parts.append(np.abs(binned).mean(1))              # |mean|        (B,64)

    return np.concatenate(parts, axis=1).astype(np.float32)  # (B, 448)


def bin_labels(labels: np.ndarray) -> np.ndarray:
    """Average labels within each bin → (N_bins, 12)."""
    n_bins = len(labels) // BIN_SAMPLES
    return labels[:n_bins * BIN_SAMPLES].reshape(n_bins, BIN_SAMPLES, N_LABELS).mean(1)


def process_labels(Y: np.ndarray):
    """
    Y: (N_bins, 12)
    Returns:
      joy   (N,4)  — axes 0-3, normalised to [-1,1]
      trig  (N,2)  — channels 10-11, normalised to [0,1]
      btn   (N,6)  — channels 4-9, binary
    """
    joy  = Y[:, :4]    / 32767.0
    trig = np.clip(Y[:, 10:12] / 32767.0, 0.0, 1.0)
    btn  = (Y[:, 4:10] > 16000).astype(np.float32)
    return joy, trig, btn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    # ── 1. Discover files ──────────────────────────────────────────────────
    files = sorted(glob.glob(os.path.join(RECORDINGS, 'train_*/broadband_data_*.h5')))
    if not files:
        # Fallback: flat .h5 alongside subdirs
        files = sorted(glob.glob(os.path.join(RECORDINGS, 'train_*.h5')))
    log(f"Found {len(files)} training files")

    if len(files) == 0:
        raise FileNotFoundError(
            f"No training HDF5 files found under {RECORDINGS}/train_*/broadband_data_*.h5"
        )

    # File-based split: every 5th file → val
    val_idx   = set(range(2, len(files), 5))
    train_files = [f for i, f in enumerate(files) if i not in val_idx]
    val_files   = [f for i, f in enumerate(files) if i in val_idx]
    log(f"Train: {len(train_files)} files  |  Val: {len(val_files)} files")

    # ── 2. Extract features ────────────────────────────────────────────────
    def load_set(flist, tag):
        feats_list, labs_list = [], []
        for i, f in enumerate(flist):
            neural, labels = load_h5(f)
            feats_list.append(extract_bin_features(neural))
            labs_list.append(bin_labels(labels))
            del neural, labels
            if (i + 1) % 8 == 0 or (i + 1) == len(flist):
                log(f"  [{tag}] {i+1}/{len(flist)} loaded")
        gc.collect()
        return (np.concatenate(feats_list, axis=0),
                np.concatenate(labs_list,  axis=0))

    log("\nLoading training set…")
    X_tr, Y_tr = load_set(train_files, 'TRAIN')
    log(f"  Train features: {X_tr.shape}  labels: {Y_tr.shape}")

    log("\nLoading validation set…")
    X_val, Y_val = load_set(val_files, 'VAL')
    log(f"  Val   features: {X_val.shape}  labels: {Y_val.shape}")

    # ── 3. Feature normalisation ───────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    del X_tr, X_val
    gc.collect()

    # ── 4. PCA ────────────────────────────────────────────────────────────
    log("\nFitting PCA (95% variance)…")
    pca = PCA(n_components=0.95, svd_solver='full')
    Z_tr  = pca.fit_transform(X_tr_s).astype(np.float32)
    Z_val = pca.transform(X_val_s).astype(np.float32)
    log(f"  PCA kept {pca.n_components_} components "
        f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

    np.savez(os.path.join(OUT_DIR, 'pca_model.npz'),
             components=pca.components_.astype(np.float32),
             mean=pca.mean_.astype(np.float32),
             explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
             scaler_mean=scaler.mean_.astype(np.float32),
             scaler_std=scaler.scale_.astype(np.float32))

    # Cumulative variance plot
    fig, ax = plt.subplots(figsize=(8, 4))
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, lw=1.5)
    ax.axhline(95, color='r', ls='--', label='95%')
    ax.axhline(99, color='orange', ls='--', label='99%')
    ax.set_xlabel('# PCA components'); ax.set_ylabel('Cumulative variance (%)')
    ax.set_title('PCA explained variance'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pca_variance.png'), dpi=150)
    plt.close()

    # ── 5. Process labels ─────────────────────────────────────────────────
    joy_tr,  trig_tr,  btn_tr  = process_labels(Y_tr)
    joy_val, trig_val, btn_val = process_labels(Y_val)
    del Y_tr, Y_val

    # ── 6. Fit Ridge regression — joystick ────────────────────────────────
    log("\nFitting Ridge → joystick (4 axes)…")
    ridge_joy = Ridge(alpha=1.0)
    ridge_joy.fit(Z_tr, joy_tr)
    joy_pred = ridge_joy.predict(Z_val).astype(np.float32)
    joy_pred = np.clip(joy_pred, -1.0, 1.0)

    np.savez(os.path.join(OUT_DIR, 'ridge_joy.npz'),
             coef=ridge_joy.coef_.astype(np.float32),
             intercept=ridge_joy.intercept_.astype(np.float32))

    # ── 7. Fit Ridge regression — triggers ───────────────────────────────
    log("Fitting Ridge → triggers (LT, RT)…")
    ridge_trig = Ridge(alpha=1.0)
    ridge_trig.fit(Z_tr, trig_tr)
    trig_pred = np.clip(ridge_trig.predict(Z_val).astype(np.float32), 0.0, 1.0)

    np.savez(os.path.join(OUT_DIR, 'ridge_trig.npz'),
             coef=ridge_trig.coef_.astype(np.float32),
             intercept=ridge_trig.intercept_.astype(np.float32))

    # ── 8. Fit Logistic regression — buttons ─────────────────────────────
    log("Fitting Logistic regression → buttons (A, B, X, Y, LB, RB)…")
    logreg_coefs = []
    logreg_intercepts = []
    btn_pred = np.zeros_like(btn_val)

    for i, name in enumerate(BUTTON_NAMES):
        y_i = btn_tr[:, i]
        pos_rate = y_i.mean()
        w = {0: 1.0, 1: max((1 - pos_rate) / (pos_rate + 1e-6), 1.0)}
        clf = LogisticRegression(C=1.0, max_iter=500, class_weight=w, solver='lbfgs')
        clf.fit(Z_tr, y_i)
        btn_pred[:, i] = clf.predict(Z_val)
        logreg_coefs.append(clf.coef_[0])
        logreg_intercepts.append(clf.intercept_[0])
        log(f"  {name}: pos_rate={pos_rate:.3f}")

    np.savez(os.path.join(OUT_DIR, 'logreg_btn.npz'),
             coef=np.array(logreg_coefs, dtype=np.float32),
             intercept=np.array(logreg_intercepts, dtype=np.float32),
             button_names=BUTTON_NAMES)

    # ── 9. Metrics ────────────────────────────────────────────────────────
    lines = []
    lines.append('=' * 60)
    lines.append('  PCA DECODER — RESULTS')
    lines.append(f'  PCA components: {pca.n_components_} ({pca.explained_variance_ratio_.sum()*100:.1f}% var)')
    lines.append(f'  Train bins: {len(Z_tr):,}  |  Val bins: {len(Z_val):,}')
    lines.append('=' * 60)

    lines.append('\nJOYSTICK (Ridge regression):')
    joy_r2_per = [r2_score(joy_val[:, i], joy_pred[:, i]) for i in range(4)]
    joy_rmse_per = [np.sqrt(np.mean((joy_val[:, i] - joy_pred[:, i])**2)) for i in range(4)]
    for i, name in enumerate(JOYSTICK_NAMES):
        lines.append(f'  {name:<16}  R²={joy_r2_per[i]:+.4f}  RMSE={joy_rmse_per[i]:.4f}')
    lines.append(f'  {"AVERAGE":<16}  R²={np.mean(joy_r2_per):+.4f}  RMSE={np.mean(joy_rmse_per):.4f}')

    lines.append('\nTRIGGERS (Ridge regression):')
    trig_r2_per = [r2_score(trig_val[:, i], trig_pred[:, i]) for i in range(2)]
    trig_rmse_per = [np.sqrt(np.mean((trig_val[:, i] - trig_pred[:, i])**2)) for i in range(2)]
    for i, name in enumerate(TRIGGER_NAMES):
        lines.append(f'  {name:<16}  R²={trig_r2_per[i]:+.4f}  RMSE={trig_rmse_per[i]:.4f}')
    lines.append(f'  {"AVERAGE":<16}  R²={np.mean(trig_r2_per):+.4f}  RMSE={np.mean(trig_rmse_per):.4f}')

    lines.append('\nBUTTONS (Logistic regression):')
    btn_acc_per = [accuracy_score(btn_val[:, i], btn_pred[:, i]) for i in range(6)]
    btn_f1_per  = [f1_score(btn_val[:, i], btn_pred[:, i], zero_division=0) for i in range(6)]
    for i, name in enumerate(BUTTON_NAMES):
        lines.append(f'  {name:<6}  acc={btn_acc_per[i]:.4f}  F1={btn_f1_per[i]:.4f}  '
                     f'(pos={btn_val[:,i].mean():.3f})')
    lines.append(f'  {"AVG":<6}  acc={np.mean(btn_acc_per):.4f}  F1={np.mean(btn_f1_per):.4f}')

    elapsed = time.time() - t_start
    lines.append(f'\nTotal time: {elapsed:.1f}s')
    lines.append('=' * 60)

    report = '\n'.join(lines)
    log('\n' + report)
    with open(os.path.join(OUT_DIR, 'results.txt'), 'w') as f:
        f.write(report + '\n')

    # ── 10. Plots ─────────────────────────────────────────────────────────
    t_axis = np.arange(len(joy_val)) * (BIN_MS / 1000)

    # Joystick
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, JOYSTICK_NAMES)):
        ax.plot(t_axis, joy_val[:, i],  'b-', alpha=0.5, lw=0.5, label='True')
        ax.plot(t_axis, joy_pred[:, i], 'r-', alpha=0.5, lw=0.5, label='Pred')
        ax.set_title(f'{name}  R²={joy_r2_per[i]:+.4f}  RMSE={joy_rmse_per[i]:.4f}')
        ax.set_ylabel('Normalised'); ax.legend(loc='upper right', fontsize=7)
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('PCA Decoder — Joystick', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'joystick.png'), dpi=150)
    plt.close()

    # Triggers
    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, TRIGGER_NAMES)):
        ax.plot(t_axis, trig_val[:, i],  'b-', alpha=0.5, lw=0.5, label='True')
        ax.plot(t_axis, trig_pred[:, i], 'r-', alpha=0.5, lw=0.5, label='Pred')
        ax.set_title(f'{name}  R²={trig_r2_per[i]:+.4f}  RMSE={trig_rmse_per[i]:.4f}')
        ax.legend(loc='upper right', fontsize=7)
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('PCA Decoder — Triggers', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'triggers.png'), dpi=150)
    plt.close()

    # Buttons bar chart
    x = np.arange(len(BUTTON_NAMES))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, btn_acc_per, color='steelblue', alpha=0.8, label='Accuracy')
    ax.bar(x, btn_f1_per, color='tomato', alpha=0.6, width=0.4, label='F1')
    ax.set_xticks(x); ax.set_xticklabels(BUTTON_NAMES)
    ax.set_ylim(0, 1); ax.set_ylabel('Score'); ax.set_title('PCA Decoder — Button Accuracy & F1')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'buttons.png'), dpi=150)
    plt.close()

    log(f"\nOutputs saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
