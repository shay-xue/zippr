"""
Decoder v7 — GRU on binned spike-count sequences.

Architecture: GRU(192, hidden=192, layers=2) with separate heads for
joystick regression, trigger regression, and button gate classification.

Preprocessing (Track A):
  - Bandpass filter 200–5000 Hz (2nd-order Butterworth)
  - Bin into 10ms windows (320 samples)
  - Spike counts at 3σ/4σ/5σ thresholds → 192 features/bin (64ch × 3)

Sequence: 30 bins = 300ms context, stride=1 for training density
Split: stratified file-based (one file per category held out for val)

Outputs → analysis/decoder_rnn_shay/
"""

import h5py
import numpy as np
import glob
import os
import gc
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, sosfilt
from sklearn.metrics import r2_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 32_000
N_NEURAL     = 64
N_LABELS     = 12
BIN_MS       = 10
BIN_SAMPLES  = int(SAMPLE_RATE * BIN_MS / 1000)   # 320
N_FEATURES   = 192   # 64ch × 3 thresholds

SEQ_LEN      = 30    # bins (300ms)
HIDDEN_SIZE  = 192
N_LAYERS     = 2
DROPOUT      = 0.3
BATCH_SIZE   = 128
LR           = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 300
PATIENCE     = 40
NOISE_STD    = 0.05

RECORDINGS   = 'data/recordings'
OUT_DIR      = 'analysis/decoder_rnn_shay'
MODELS_DIR   = 'models'

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
LABEL_NAMES  = ['LStX','LStY','RStX','RStY','A','B','X','Y','LB','RB','LT','RT']
JOY_NAMES    = ['LStX','LStY','RStX','RStY']
TRIG_NAMES   = ['LT','RT']
BTN_NAMES    = ['A','B','X','Y','LB','RB']


def log(msg):
    print(msg, flush=True)


# ── Step 1: Preprocessing ─────────────────────────────────────────────────────

def bandpass_filter(neural):
    """
    2nd-order Butterworth bandpass 200–5000 Hz.
    Input/output: (n_samples, 64) float32
    """
    sos = butter(2, [200, 5000], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    return sosfilt(sos, neural, axis=0).astype(np.float32)


def extract_features_track_a(neural_filtered):
    """
    Track A: spike counts at 3σ/4σ/5σ → 192 features/bin.

    Thresholds computed per-channel from filtered signal.
    Returns (n_bins, 192) float32.
    """
    ch_std = neural_filtered.std(0) + 1e-8   # (64,)
    n_bins = len(neural_filtered) // BIN_SAMPLES
    neural_tr = neural_filtered[:n_bins * BIN_SAMPLES]

    parts = []
    for sigma in [3.0, 4.0, 5.0]:
        thresh = sigma * ch_std
        above = neural_tr >  thresh
        below = neural_tr < -thresh
        spikes = (above | below).astype(np.float32)  # (N, 64)
        counts = spikes.reshape(n_bins, BIN_SAMPLES, N_NEURAL).sum(1)  # (n_bins, 64)
        parts.append(counts)
        del above, below, spikes

    return np.concatenate(parts, axis=1).astype(np.float32)   # (n_bins, 192)


def load_file(path):
    """
    Load one HDF5, bandpass filter, bin → (features, labels).
    Returns:
      features  (n_bins, 192)  Track A spike counts
      labels    (n_bins, 12)   raw controller values (un-normalised)
    """
    with h5py.File(path, 'r') as hf:
        raw = hf['acquisition/ElectricalSeries'][:]
    n = len(raw) // 76
    d = raw[:n * 76].reshape(n, 76).astype(np.float32)
    neural = d[:, :N_NEURAL]
    labels = d[:, N_NEURAL:N_NEURAL + N_LABELS]
    del raw, d

    neural_filt = bandpass_filter(neural)
    del neural

    features = extract_features_track_a(neural_filt)
    del neural_filt

    n_bins = len(features)
    label_bins = labels[:n_bins * BIN_SAMPLES].reshape(
        n_bins, BIN_SAMPLES, N_LABELS).mean(1)

    gc.collect()
    return features, label_bins.astype(np.float32)


def get_stratified_split(files):
    """
    File-based train/val split ensuring one file per recording category in val.

    Categories inferred from filename. This prevents the val set from being
    all-rest or all-freestyle — it must see examples of every input type.
    """
    categories = {
        'rest':       ['rest'],
        'lstick':     ['lstick', 'lstickx', 'lsticky', 'lstick_diag', 'lstick_buttons',
                       'lstick_triggers'],
        'rstick':     ['rstick', 'rstickx', 'rsticky', 'rstick_diag', 'rstick_bumpers',
                       'rstick_bumptrig'],
        'buttons':    ['buttons', 'ab_alt', 'xy_alt'],
        'bumpers':    ['bumpers', 'lb_spam', 'rb_spam'],
        'triggers':   ['lt_spam', 'rt_spam', 'lt_analog', 'rt_analog'],
        'freestyle':  ['freestyle', 'slow_all', 'robot_sim'],
        'bothsticks': ['bothsticks'],
    }

    def get_category(path):
        stem = os.path.basename(os.path.dirname(path)).split('_', 2)[-1]
        for cat, keywords in categories.items():
            if any(stem.startswith(kw) or kw in stem for kw in keywords):
                return cat
        return 'other'

    # Group files by category
    by_cat = {}
    for f in files:
        cat = get_category(f)
        by_cat.setdefault(cat, []).append(f)

    val_files = []
    train_files = []
    for cat, flist in sorted(by_cat.items()):
        # Take middle file of each category for val (avoids boundary effects)
        val_idx = len(flist) // 2
        val_files.append(flist[val_idx])
        train_files.extend(f for i, f in enumerate(flist) if i != val_idx)
        log(f"  {cat:12s}: {len(flist)} files  → val: {os.path.basename(os.path.dirname(flist[val_idx]))}")

    return train_files, val_files


def process_labels(label_bins):
    """
    Split (n_bins, 12) label array into task-specific targets.

    Returns:
      joy   (n_bins, 4)  — axes normalised to [-1, 1]
      trig  (n_bins, 2)  — LT/RT normalised to [0, 1]
      btn   (n_bins, 6)  — A/B/X/Y/LB/RB binary (0/1)
      gate  (n_bins, 1)  — 1 if no buttons pressed, 0 otherwise
    """
    joy  = np.clip(label_bins[:, :4]   / 32767.0, -1.0, 1.0).astype(np.float32)
    trig = np.clip(label_bins[:, 10:12] / 32767.0,  0.0, 1.0).astype(np.float32)
    btn  = (label_bins[:, 4:10] > 16000).astype(np.float32)
    gate = (btn.sum(1) == 0).astype(np.float32).reshape(-1, 1)
    return joy, trig, btn, gate


def extract_all(files, tag=''):
    """Load all files → lists of (features, joy, trig, btn, gate) per file."""
    feats_list  = []
    joy_list    = []
    trig_list   = []
    btn_list    = []
    gate_list   = []
    total_bins  = 0

    for i, f in enumerate(files):
        feat, lbl = load_file(f)
        joy, trig, btn, gate = process_labels(lbl)
        feats_list.append(feat)
        joy_list.append(joy)
        trig_list.append(trig)
        btn_list.append(btn)
        gate_list.append(gate)
        total_bins += len(feat)
        if (i + 1) % 8 == 0 or (i + 1) == len(files):
            log(f"  [{tag}] {i+1}/{len(files)} files  ({total_bins:,} bins)")

    return feats_list, joy_list, trig_list, btn_list, gate_list


# ── Step 2: Sequence Dataset ──────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Returns (seq, label) pairs. No sequences cross file boundaries.

    seq   : (SEQ_LEN, 192)  normalised spike-count features
    label : (7,)            [joy×4, trig×2, gate×1]
    """
    def __init__(self, feats_list, joy_list, trig_list, gate_list,
                 seq_len, stride, feat_mean, feat_std):
        self.seq_len   = seq_len
        self.feat_mean = feat_mean.reshape(1, -1).astype(np.float32)
        self.feat_std  = feat_std.reshape(1, -1).astype(np.float32)

        self.feats  = [(f - self.feat_mean) / self.feat_std for f in feats_list]
        self.labels = [np.concatenate([j, t, g], axis=1)
                       for j, t, g in zip(joy_list, trig_list, gate_list)]

        # Build flat index: (file_idx, start_bin)
        self.index_map = []
        for fi, feat in enumerate(self.feats):
            n_bins = len(feat)
            if n_bins < seq_len:
                continue
            starts = range(0, n_bins - seq_len + 1, stride)
            self.index_map.extend((fi, s) for s in starts)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fi, start = self.index_map[idx]
        x = self.feats[fi][start:start + self.seq_len]          # (T, 192)
        y = self.labels[fi][start + self.seq_len - 1]           # (7,)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


# ── Step 3: Model ─────────────────────────────────────────────────────────────

class GRUDecoder(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = (nn.Linear(input_size, hidden_size)
                           if input_size != hidden_size else nn.Identity())
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.joy_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.GELU(),
            nn.Linear(64, 4), nn.Tanh(),
        )
        self.trig_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.GELU(),
            nn.Linear(32, 2), nn.Sigmoid(),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input_proj(x)          # (B, T, H)
        _, h_n = self.gru(x)            # h_n: (n_layers, B, H)
        z = h_n[-1]                     # (B, H)
        return self.joy_head(z), self.trig_head(z), self.gate_head(z)


# ── ONNX export wrapper ───────────────────────────────────────────────────────

class GRUDecoderForExport(nn.Module):
    """
    Wraps GRUDecoder for on-device ONNX convention.
    Input:  (1, n_features, seq_len)  — channels-first as Synapse app expects
    Output: (1, 7)                    — [joy×4, trig×2, gate]
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.permute(0, 2, 1)                         # → (1, seq_len, n_features)
        joy, trig, gate = self.model(x)
        return torch.cat([joy, trig, gate], dim=1)      # (1, 7)


# ── Step 4: Training ──────────────────────────────────────────────────────────

def train_rnn(feats_tr, joy_tr, trig_tr, btn_tr, gate_tr,
              feats_val, joy_val, trig_val, btn_val, gate_val, epochs=EPOCHS):

    # Compute global normalisation from training bins only
    all_tr = np.concatenate(feats_tr, axis=0)
    feat_mean = all_tr.mean(0).astype(np.float32)
    feat_std  = (all_tr.std(0) + 1e-8).astype(np.float32)
    del all_tr; gc.collect()

    log(f"\n  Feature mean range: [{feat_mean.min():.2f}, {feat_mean.max():.2f}]")
    log(f"  Feature std  range: [{feat_std.min():.2f}, {feat_std.max():.2f}]")

    # Datasets
    tr_ds = SequenceDataset(feats_tr, joy_tr, trig_tr, gate_tr,
                            SEQ_LEN, stride=1, feat_mean=feat_mean, feat_std=feat_std)
    val_ds = SequenceDataset(feats_val, joy_val, trig_val, gate_val,
                             SEQ_LEN, stride=SEQ_LEN, feat_mean=feat_mean, feat_std=feat_std)

    log(f"  Train sequences: {len(tr_ds):,}  |  Val sequences: {len(val_ds):,}")

    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0,
                        drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=256,        shuffle=False, num_workers=0)

    model = GRUDecoder().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model params: {n_params:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)

    mse_fn  = nn.MSELoss()
    bce_fn  = nn.BCELoss()

    best_score = -999.0
    best_state = None
    no_imp     = 0
    history    = []

    for ep in range(epochs):
        t_ep = time.time()

        # ── Train ──
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb = xb.to(DEVICE)
            # Input noise augmentation
            xb = xb + NOISE_STD * torch.randn_like(xb)
            joy_b  = yb[:, :4].to(DEVICE)
            trig_b = yb[:, 4:6].to(DEVICE)
            gate_b = yb[:, 6:7].to(DEVICE)

            joy_p, trig_p, gate_p = model(xb)
            loss = mse_fn(joy_p, joy_b) + mse_fn(trig_p, trig_b) + \
                   0.5 * bce_fn(gate_p, gate_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()

        sched.step()
        tr_loss /= len(tr_dl)

        # ── Validate ──
        model.eval()
        joy_preds, joy_trues = [], []
        trig_preds, trig_trues = [], []
        gate_preds, gate_trues = [], []
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE)
                joy_b  = yb[:, :4].to(DEVICE)
                trig_b = yb[:, 4:6].to(DEVICE)
                gate_b = yb[:, 6:7].to(DEVICE)
                joy_p, trig_p, gate_p = model(xb)
                val_loss += (mse_fn(joy_p, joy_b) + mse_fn(trig_p, trig_b) +
                             0.5 * bce_fn(gate_p, gate_b)).item()
                joy_preds.append(joy_p.cpu().numpy())
                joy_trues.append(joy_b.cpu().numpy())
                trig_preds.append(trig_p.cpu().numpy())
                trig_trues.append(trig_b.cpu().numpy())
                gate_preds.append((gate_p > 0.5).cpu().numpy().astype(float))
                gate_trues.append(gate_b.cpu().numpy())

        val_loss /= len(val_dl)
        jp = np.concatenate(joy_preds);  jt = np.concatenate(joy_trues)
        tp = np.concatenate(trig_preds); tt = np.concatenate(trig_trues)
        gp = np.concatenate(gate_preds); gt = np.concatenate(gate_trues)

        joy_r2   = r2_score(jt, jp, multioutput='uniform_average')
        trig_r2  = r2_score(tt, tp, multioutput='uniform_average')
        gate_acc = accuracy_score(gt.flatten(), gp.flatten())
        score    = joy_r2  # optimise for joystick R²

        history.append({
            'tr_loss': tr_loss, 'val_loss': val_loss,
            'joy_r2': joy_r2, 'trig_r2': trig_r2, 'gate_acc': gate_acc,
        })

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 10 == 0:
            per_joy = [r2_score(jt[:, i], jp[:, i]) for i in range(4)]
            log(f"  Ep {ep+1:3d}  tr={tr_loss:.4f}  val={val_loss:.4f}  "
                f"joy_r²={joy_r2:+.4f}  best={best_score:+.4f}  "
                f"[{per_joy[0]:.3f},{per_joy[1]:.3f},{per_joy[2]:.3f},{per_joy[3]:.3f}]  "
                f"trig={trig_r2:+.4f}  gate={gate_acc:.3f}  "
                f"lr={opt.param_groups[0]['lr']:.1e}  pat={no_imp}  "
                f"({time.time()-t_ep:.1f}s)")

        if no_imp >= PATIENCE:
            log(f"  Early stop at epoch {ep+1} (patience={PATIENCE})")
            break

    # Re-evaluate with best checkpoint so returned predictions match best model
    model.load_state_dict(best_state)
    model.eval()
    joy_preds, joy_trues = [], []
    trig_preds, trig_trues = [], []
    gate_preds, gate_trues = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(DEVICE)
            joy_p, trig_p, gate_p = model(xb)
            joy_preds.append(joy_p.cpu().numpy())
            joy_trues.append(yb[:, :4].numpy())
            trig_preds.append(trig_p.cpu().numpy())
            trig_trues.append(yb[:, 4:6].numpy())
            gate_preds.append((gate_p > 0.5).cpu().numpy().astype(float))
            gate_trues.append(yb[:, 6:7].numpy())
    jp = np.concatenate(joy_preds);  jt = np.concatenate(joy_trues)
    tp = np.concatenate(trig_preds); tt = np.concatenate(trig_trues)
    gp = np.concatenate(gate_preds); gt = np.concatenate(gate_trues)

    return model, feat_mean, feat_std, history, jp, jt, tp, tt, gp, gt


# ── Step 5: ONNX Export ───────────────────────────────────────────────────────

def export_onnx(model, feat_mean, feat_std, out_path):
    """
    Export with device-expected input shape (1, n_features, seq_len).
    Internally transposes to (1, seq_len, n_features) for the GRU.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_model = GRUDecoderForExport(model.cpu().eval())
    dummy = torch.randn(1, N_FEATURES, SEQ_LEN)

    torch.onnx.export(
        export_model,
        dummy,
        out_path,
        input_names=['neural_features'],
        output_names=['arm_control'],
        dynamic_axes={
            'neural_features': {0: 'batch'},
            'arm_control':     {0: 'batch'},
        },
        opset_version=13,
        dynamo=False,
    )
    log(f"  ONNX saved: {out_path}")
    log(f"  Input:  (batch, {N_FEATURES}, {SEQ_LEN})  — channels × window")
    log(f"  Output: (batch, 7)  — [LStX, LStY, RStX, RStY, LT, RT, gate]")

    # Verify: compare PyTorch vs ONNX Runtime on 10 random inputs
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path)
        diffs = []
        for _ in range(10):
            x_np = np.random.randn(1, N_FEATURES, SEQ_LEN).astype(np.float32)
            with torch.no_grad():
                pt_out = export_model(torch.from_numpy(x_np)).numpy()
            ort_out = sess.run(None, {'neural_features': x_np})[0]
            diffs.append(np.abs(pt_out - ort_out).max())
        log(f"  ONNX verification: max |PyTorch − ORT| = {max(diffs):.2e}  ✓")
    except Exception as e:
        log(f"  ONNX verification skipped: {e}")


# ── XGBoost baseline ──────────────────────────────────────────────────────────

def run_xgboost_baseline(feats_tr, joy_tr, feats_val, joy_val):
    """Quick XGBoost baseline on flattened single-bin features."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        log("  XGBoost not installed — skipping baseline")
        return None

    X_tr  = np.concatenate(feats_tr,  axis=0)
    X_val = np.concatenate(feats_val, axis=0)
    Y_tr  = np.concatenate(joy_tr,    axis=0)
    Y_val = np.concatenate(joy_val,   axis=0)

    # Normalise
    m, s = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr  = (X_tr  - m) / s
    X_val = (X_val - m) / s

    log("  Fitting XGBoost (joystick only, no temporal context)...")
    r2s = []
    for i, name in enumerate(JOY_NAMES):
        xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8,
                           tree_method='hist', n_jobs=-1, verbosity=0)
        xgb.fit(X_tr, Y_tr[:, i])
        r2 = r2_score(Y_val[:, i], xgb.predict(X_val))
        r2s.append(r2)
        log(f"    {name}: R²={r2:.4f}")
    avg = float(np.mean(r2s))
    log(f"  XGBoost mean joystick R²: {avg:.4f}  (baseline to beat)")
    return avg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    t0 = time.time()

    log(f"Device: {DEVICE}")
    log(f"SEQ_LEN={SEQ_LEN} bins ({SEQ_LEN * BIN_MS}ms)  |  "
        f"HIDDEN={HIDDEN_SIZE}  |  LAYERS={N_LAYERS}  |  DROPOUT={DROPOUT}")
    log(f"LR={LR}  |  BATCH={BATCH_SIZE}  |  EPOCHS={EPOCHS}  |  PATIENCE={PATIENCE}\n")

    # ── Load files ────────────────────────────────────────────────────────────
    files = sorted(glob.glob(f'{RECORDINGS}/train_*/broadband_data_*.h5'))
    log(f"Found {len(files)} training files")

    log("\nStratified file split:")
    train_files, val_files = get_stratified_split(files)
    log(f"\nTrain: {len(train_files)} files  |  Val: {len(val_files)} files")

    # ── Extract features ──────────────────────────────────────────────────────
    log("\nExtracting training features...")
    feats_tr, joy_tr, trig_tr, btn_tr, gate_tr = extract_all(train_files, 'TRAIN')

    log("\nExtracting validation features...")
    feats_val, joy_val, trig_val, btn_val, gate_val = extract_all(val_files, 'VAL')

    tr_bins  = sum(len(f) for f in feats_tr)
    val_bins = sum(len(f) for f in feats_val)
    log(f"\nTotal: {tr_bins:,} train bins ({tr_bins*BIN_MS/1000:.1f}s)  |  "
        f"{val_bins:,} val bins ({val_bins*BIN_MS/1000:.1f}s)")

    # ── XGBoost baseline ──────────────────────────────────────────────────────
    log("\n── XGBoost Baseline ──────────────────────────────────────────")
    xgb_r2 = run_xgboost_baseline(feats_tr, joy_tr, feats_val, joy_val)

    # ── Train GRU ─────────────────────────────────────────────────────────────
    log("\n── GRU Training ──────────────────────────────────────────────")
    (model, feat_mean, feat_std, history,
     jp, jt, tp, tt, gp, gt) = train_rnn(
        feats_tr, joy_tr, trig_tr, btn_tr, gate_tr,
        feats_val, joy_val, trig_val, btn_val, gate_val,
    )

    # ── Final metrics ─────────────────────────────────────────────────────────
    joy_r2_per   = [r2_score(jt[:, i], jp[:, i]) for i in range(4)]
    joy_rmse_per = [np.sqrt(np.mean((jt[:, i] - jp[:, i])**2)) for i in range(4)]
    trig_r2_per  = [r2_score(tt[:, i], tp[:, i]) for i in range(2)]
    gate_acc     = accuracy_score(gt.flatten(), gp.flatten())

    log(f"\n{'='*60}")
    log(f"  GRU DECODER v7 — FINAL RESULTS")
    log(f"{'='*60}")
    log(f"\nJoystick R² / RMSE:")
    for name, r2, rmse in zip(JOY_NAMES, joy_r2_per, joy_rmse_per):
        log(f"  {name:<8}  R²={r2:+.4f}  RMSE={rmse:.4f}")
    log(f"  {'AVERAGE':<8}  R²={np.mean(joy_r2_per):+.4f}  RMSE={np.mean(joy_rmse_per):.4f}")
    log(f"\nTriggers R²:")
    for name, r2 in zip(TRIG_NAMES, trig_r2_per):
        log(f"  {name:<8}  R²={r2:+.4f}")
    log(f"  {'AVERAGE':<8}  R²={np.mean(trig_r2_per):+.4f}")
    log(f"\nGate accuracy: {gate_acc:.4f}")
    if xgb_r2 is not None:
        log(f"\nXGBoost baseline joystick R²: {xgb_r2:.4f}")
        log(f"GRU joystick R²:              {np.mean(joy_r2_per):+.4f}  "
            f"({'↑ better' if np.mean(joy_r2_per) > xgb_r2 else '↓ worse'} than baseline)")
    log(f"\nTotal time: {time.time()-t0:.1f}s")
    log(f"{'='*60}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    np.savez(os.path.join(OUT_DIR, 'feat_norm.npz'),
             feat_mean=feat_mean, feat_std=feat_std,
             channel_stds=np.zeros(N_NEURAL, dtype=np.float32))  # placeholder

    meta = {
        'seq_len':       SEQ_LEN,
        'bin_ms':        BIN_MS,
        'feature_type':  'track_a',
        'sigma_thresholds': [3.0, 4.0, 5.0],
        'n_features':    N_FEATURES,
        'hidden_size':   HIDDEN_SIZE,
        'n_layers':      N_LAYERS,
        'dropout':       DROPOUT,
        'sample_rate':   SAMPLE_RATE,
        'bandpass_hz':   [200, 5000],
        'joy_r2_avg':    float(np.mean(joy_r2_per)),
        'joy_r2_per':    [float(r) for r in joy_r2_per],
        'trig_r2_avg':   float(np.mean(trig_r2_per)),
        'gate_acc':      float(gate_acc),
        'xgb_baseline':  float(xgb_r2) if xgb_r2 is not None else None,
    }
    with open(os.path.join(OUT_DIR, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    log(f"\nSaved feat_norm.npz and model_meta.json to {OUT_DIR}/")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = os.path.join(OUT_DIR, 'best_model.pt')
    torch.save(model.state_dict(), ckpt_path)
    log(f"\nCheckpoint saved: {ckpt_path}")

    # ── ONNX Export ───────────────────────────────────────────────────────────
    log("\n── ONNX Export ───────────────────────────────────────────────")
    export_onnx(model, feat_mean, feat_std,
                os.path.join(MODELS_DIR, 'decoder.onnx'))

    # ── Plots ─────────────────────────────────────────────────────────────────
    eps = np.arange(1, len(history) + 1)

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(eps, [h['tr_loss']  for h in history], label='Train')
    axes[0].plot(eps, [h['val_loss'] for h in history], label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps, [h['joy_r2']  for h in history], label='Joy R²')
    axes[1].plot(eps, [h['trig_r2'] for h in history], label='Trig R²')
    axes[1].axhline(max(h['joy_r2'] for h in history), color='r', ls='--', lw=0.8)
    axes[1].set_title('Validation R²'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(eps, [h['gate_acc'] for h in history])
    axes[2].set_title('Gate Accuracy'); axes[2].set_ylabel('Acc')
    axes[2].grid(True, alpha=0.3)
    plt.suptitle('GRU Decoder v7 — Training Curves', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'training_curves.png'), dpi=150)
    plt.close()

    # Joystick predictions vs ground truth
    n_plot = min(300, len(jt))
    t_axis = np.arange(n_plot) * SEQ_LEN * BIN_MS / 1000
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, JOY_NAMES)):
        ax.plot(t_axis, jt[:n_plot, i], 'b-', alpha=0.6, lw=0.8, label='True')
        ax.plot(t_axis, jp[:n_plot, i], 'r-', alpha=0.6, lw=0.8, label='Pred')
        ax.set_title(f'{name}  R²={joy_r2_per[i]:+.4f}  RMSE={joy_rmse_per[i]:.4f}')
        ax.legend(loc='upper right', fontsize=7); ax.set_ylabel('Normalised')
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('GRU Decoder v7 — Joystick Predictions vs Ground Truth', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'joystick_predictions.png'), dpi=150)
    plt.close()

    # Per-axis R² bar chart
    all_names = JOY_NAMES + TRIG_NAMES
    all_r2    = joy_r2_per + trig_r2_per
    colors    = ['steelblue'] * 4 + ['darkorange'] * 2
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(all_names, all_r2, color=colors, alpha=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.axhline(np.mean(joy_r2_per), color='steelblue', ls='--', lw=1,
               label=f'Joy avg R²={np.mean(joy_r2_per):+.3f}')
    if xgb_r2 is not None:
        ax.axhline(xgb_r2, color='gray', ls=':', lw=1.5,
                   label=f'XGB baseline={xgb_r2:.3f}')
    for bar, r2 in zip(bars, all_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{r2:+.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('R²'); ax.set_title('GRU Decoder v7 — Per-Output R²')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'per_output_r2.png'), dpi=150)
    plt.close()

    log(f"\nAll plots saved to {OUT_DIR}/")
    log("Done.")


if __name__ == '__main__':
    main()
