"""
Decoder v6 — Full restart with proper evaluation.

Key changes from v1-v5:
  1. Stratified file-based train/val split (no data leakage, no distribution shift)
  2. Two feature tracks: spike-only (Track A) vs rich features (Track B)
  3. XGBoost baseline first to establish R² ceiling
  4. Multi-head MLP: joystick (4) + triggers (2) + gate (1)
  5. ONNX export for on-device deployment

Robot arm mapping:
  L-Stick X → arm X     | L-Stick Y → arm Y
  R-Stick X → clamp rot | R-Stick Y → arm Z depth
  LT → open clamp       | RT → close clamp
  A/B/X/Y/LB/RB → gate (suppress joystick when pressed)
"""

import h5py, numpy as np, glob, os, gc, sys, time, json
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, accuracy_score
from scipy.signal import butter, sosfiltfilt

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

SAMPLE_RATE = 32000
N_NEURAL = 64
BIN_MS = 10
BIN_SAMPLES = int(SAMPLE_RATE * BIN_MS / 1000)  # 320
WINDOW = 15  # bins of context = 150ms
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

label_names = ['LStickX', 'LStickY', 'RStickX', 'RStickY',
               'A', 'B', 'X', 'Y', 'LB', 'RB', 'LT', 'RT']

OUT_DIR = 'analysis/decoder_v6'
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


# ── File grouping for stratified split ────────────────────────────────────

FILE_CATEGORIES = {
    'rest':       ['001'],
    'single_stick': ['002', '003', '017', '018', '023', '024', '025', '026'],
    'both_sticks': ['004', '027', '028'],
    'buttons_triggers': ['005', '006', '011', '012', '013', '014', '015', '016', '029', '030'],
    'combined':   ['007', '008', '019', '020', '031'],
    'freestyle':  ['009', '010', '021', '022'],
    'robot_sim':  ['032'],
}


def get_stratified_split(files, seed=42):
    """Split files into train/val ensuring each category is represented in both."""
    rng = np.random.RandomState(seed)

    # Map file paths to their 3-digit IDs
    def get_id(f):
        basename = os.path.basename(os.path.dirname(f))  # train_001_rest.h5
        return basename.split('_')[1]  # '001'

    file_ids = {get_id(f): f for f in files}
    train_files, val_files = [], []

    for cat, ids in FILE_CATEGORIES.items():
        available = [file_ids[i] for i in ids if i in file_ids]
        if len(available) <= 1:
            # Single file category — put in train (we need it)
            train_files.extend(available)
        else:
            # Pick ~20% for validation (at least 1)
            n_val = max(1, len(available) // 5)
            rng.shuffle(available)
            val_files.extend(available[:n_val])
            train_files.extend(available[n_val:])

    log(f"Split: {len(train_files)} train, {len(val_files)} val")
    for cat, ids in FILE_CATEGORIES.items():
        n_train = sum(1 for f in train_files if get_id(f) in ids)
        n_val = sum(1 for f in val_files if get_id(f) in ids)
        log(f"  {cat:20s}: {n_train} train, {n_val} val")

    return sorted(train_files), sorted(val_files)


# ── Feature extraction ────────────────────────────────────────────────────

def bandpass_filter(data, low=200, high=5000, fs=SAMPLE_RATE, order=2):
    """Apply bandpass filter matching on-device pipeline."""
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data, axis=0).astype(np.float32)


def extract_features_track_a(neural_bins, channel_stds):
    """Track A: Spike counts at 3 thresholds. Shape: (n_bins, 192)"""
    n_bins, bs, n_ch = neural_bins.shape
    parts = []
    for thresh in [3.0, 4.0, 5.0]:
        threshold = thresh * channel_stds  # (64,)
        # Count threshold crossings per bin
        above = neural_bins > threshold[None, None, :]
        below = neural_bins < -threshold[None, None, :]
        spikes = (above | below).astype(np.float32).sum(1)  # (n_bins, 64)
        parts.append(spikes)
    return np.concatenate(parts, axis=1)  # (n_bins, 192)


def extract_features_track_b(neural_bins, channel_stds):
    """Track B: Rich features. Shape: (n_bins, 768)"""
    n_bins, bs, n_ch = neural_bins.shape

    # Spike counts (192)
    spike_feats = extract_features_track_a(neural_bins, channel_stds)

    # Voltage statistics (256)
    mean_v = neural_bins.mean(1)                    # (n_bins, 64)
    std_v = neural_bins.std(1)                      # (n_bins, 64)
    abs_mean = np.abs(neural_bins).mean(1)           # (n_bins, 64)
    ptp = neural_bins.max(1) - neural_bins.min(1)   # (n_bins, 64)

    # FFT band powers (320)
    fft_data = np.fft.rfft(neural_bins, axis=1)     # (n_bins, 161, 64)
    power = np.abs(fft_data) ** 2
    # 5 frequency bands spanning the 200-5000Hz bandpass range
    # At 32kHz with 320 samples: freq_resolution = 100Hz, max_freq = 16kHz
    # Bins: 0=DC, 1=100Hz, 2=200Hz, ... 50=5000Hz, ... 160=16kHz
    band_edges = [(2, 5), (5, 15), (15, 30), (30, 50), (50, 100)]
    fft_parts = [power[:, lo:hi, :].mean(1) for lo, hi in band_edges]
    fft_feats = np.concatenate(fft_parts, axis=1)   # (n_bins, 320)

    return np.concatenate([spike_feats, mean_v, std_v, abs_mean, ptp, fft_feats], axis=1)  # 768


def load_file(filepath):
    """Load a single HDF5 file, return neural (n_bins, bs, 64) and labels (n_bins, 12)."""
    with h5py.File(filepath, 'r') as hf:
        raw = hf['acquisition/ElectricalSeries'][:]
    n = len(raw) // 76
    d = raw[:n * 76].reshape(n, 76)
    neural = d[:, :64].astype(np.float32)
    labels = d[:, 64:76].astype(np.float32)
    del raw, d

    # Apply bandpass filter (matching on-device pipeline)
    neural = bandpass_filter(neural)

    nb = len(neural) // BIN_SAMPLES
    neural_bins = neural[:nb * BIN_SAMPLES].reshape(nb, BIN_SAMPLES, 64)
    label_bins = labels[:nb * BIN_SAMPLES].reshape(nb, BIN_SAMPLES, 12).mean(1)

    return neural_bins, label_bins


def extract_all(files, track='A'):
    """Extract features from multiple files. Returns per-file lists for proper windowing."""
    # First pass: compute global channel stds for threshold-based spike detection
    log("  Computing channel statistics...")
    all_stds = []
    for f in files:
        with h5py.File(f, 'r') as hf:
            raw = hf['acquisition/ElectricalSeries'][:]
        n = len(raw) // 76
        d = raw[:n * 76].reshape(n, 76)
        neural = d[:, :64].astype(np.float32)
        neural = bandpass_filter(neural)
        all_stds.append(neural.std(0))
        del raw, d, neural

    channel_stds = np.mean(all_stds, axis=0)  # global average std per channel
    log(f"  Channel std range: [{channel_stds.min():.1f}, {channel_stds.max():.1f}]")

    # Second pass: extract features
    all_feats, all_labels = [], []
    for i, f in enumerate(files):
        neural_bins, label_bins = load_file(f)
        if track == 'A':
            feats = extract_features_track_a(neural_bins, channel_stds)
        else:
            feats = extract_features_track_b(neural_bins, channel_stds)
        all_feats.append(feats)
        all_labels.append(label_bins)
        del neural_bins
        if (i + 1) % 8 == 0:
            log(f"  Extracted {i+1}/{len(files)} files")

    log(f"  Extracted {len(files)} files total")
    return all_feats, all_labels, channel_stds


def make_windows_per_file(feats_list, labels_list, w=WINDOW, stride=1):
    """Create windows per file, then concatenate. No cross-file windows."""
    all_X, all_Y = [], []
    for feats, labels in zip(feats_list, labels_list):
        if len(feats) < w:
            continue
        # Sliding window
        Xw = np.lib.stride_tricks.sliding_window_view(feats, w, axis=0)
        Xw = Xw[::stride]  # apply stride
        Xw = Xw.reshape(len(Xw), -1)  # flatten: (n_windows, w * n_features)
        Yw = labels[w - 1:][::stride]
        all_X.append(Xw)
        all_Y.append(Yw)
    X = np.concatenate(all_X)
    Y = np.concatenate(all_Y)
    return X, Y


def process_labels(Y):
    """Split labels into joystick, trigger, gate targets."""
    Y_joy = Y[:, :4] / 32767.0                        # joystick [-1, 1]
    Y_trig = Y[:, 10:12] / 32767.0                    # LT(idx10), RT(idx11) → [0, 1]
    Y_trig = np.clip(Y_trig, 0, 1)
    Y_btn = (Y[:, 4:10] > 16000).astype(np.float32)   # A,B,X,Y,LB,RB → binary
    Y_gate = (Y_btn.sum(1) == 0).astype(np.float32).reshape(-1, 1)  # 1=no buttons, 0=buttons pressed
    return Y_joy, Y_trig, Y_btn, Y_gate


# ── XGBoost Baseline ──────────────────────────────────────────────────────

def run_xgboost_baseline(Xtr, Ytr, Xval, Yval, track_name):
    """Run XGBoost to get R² ceiling."""
    try:
        from xgboost import XGBRegressor, XGBClassifier
    except ImportError:
        log("XGBoost not available, installing...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', '-q'])
        from xgboost import XGBRegressor, XGBClassifier

    Y_joy_tr, Y_trig_tr, Y_btn_tr, Y_gate_tr = process_labels(Ytr)
    Y_joy_val, Y_trig_val, Y_btn_val, Y_gate_val = process_labels(Yval)

    # Normalize features
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_n = (Xtr - m) / s
    Xval_n = (Xval - m) / s

    log(f"\n{'='*60}")
    log(f"XGBoost Baseline — {track_name}")
    log(f"{'='*60}")
    log(f"Train: {Xtr_n.shape}, Val: {Xval_n.shape}")

    results = {}

    # Joystick axes
    joy_names = ['LStickX', 'LStickY', 'RStickX', 'RStickY']
    joy_r2s = []
    for i, name in enumerate(joy_names):
        t0 = time.time()
        model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                             n_jobs=-1, verbosity=0)
        model.fit(Xtr_n, Y_joy_tr[:, i])
        pred = model.predict(Xval_n)
        r2 = r2_score(Y_joy_val[:, i], pred)
        joy_r2s.append(r2)
        log(f"  {name:12s}: R²={r2:.4f} ({time.time()-t0:.1f}s)")
    results['joy_r2'] = np.mean(joy_r2s)
    log(f"  Mean joystick R²: {results['joy_r2']:.4f}")

    # Triggers
    trig_names = ['LT', 'RT']
    trig_r2s = []
    for i, name in enumerate(trig_names):
        t0 = time.time()
        model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                             n_jobs=-1, verbosity=0)
        model.fit(Xtr_n, Y_trig_tr[:, i])
        pred = model.predict(Xval_n)
        r2 = r2_score(Y_trig_val[:, i], pred)
        trig_r2s.append(r2)
        log(f"  {name:12s}: R²={r2:.4f} ({time.time()-t0:.1f}s)")
    results['trig_r2'] = np.mean(trig_r2s)

    # Button gate (classification)
    t0 = time.time()
    gate_model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                                n_jobs=-1, verbosity=0)
    gate_model.fit(Xtr_n, Y_gate_tr.ravel())
    gate_pred = gate_model.predict(Xval_n)
    gate_acc = accuracy_score(Y_gate_val.ravel(), gate_pred)
    results['gate_acc'] = gate_acc
    log(f"  Gate accuracy: {gate_acc:.4f} ({time.time()-t0:.1f}s)")

    log(f"\n  SUMMARY: joy_R²={results['joy_r2']:.4f}, trig_R²={results['trig_r2']:.4f}, gate_acc={results['gate_acc']:.4f}")
    return results


# ── MLP Models ────────────────────────────────────────────────────────────

class MultiHeadDecoder(nn.Module):
    """Combined decoder with shared backbone + separate heads."""
    def __init__(self, input_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
        )
        self.joy_head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 4), nn.Tanh()  # [-1, 1]
        )
        self.trig_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 2), nn.Sigmoid()  # [0, 1]
        )
        self.gate_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid()  # gate probability
        )

    def forward(self, x):
        z = self.backbone(x)
        joy = self.joy_head(z)
        trig = self.trig_head(z)
        gate = self.gate_head(z)
        return joy, trig, gate


def train_mlp(Xtr, Ytr, Xval, Yval, track_name, epochs=200):
    """Train multi-head MLP with proper evaluation."""
    Y_joy_tr, Y_trig_tr, _, Y_gate_tr = process_labels(Ytr)
    Y_joy_val, Y_trig_val, _, Y_gate_val = process_labels(Yval)

    # Normalize features
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_n = ((Xtr - m) / s).astype(np.float32)
    Xval_n = ((Xval - m) / s).astype(np.float32)

    input_dim = Xtr_n.shape[1]
    model = MultiHeadDecoder(input_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"\n{'='*60}")
    log(f"MLP Training — {track_name}")
    log(f"{'='*60}")
    log(f"Input dim: {input_dim}, Parameters: {n_params:,}")

    # Prepare data
    train_ds = TensorDataset(
        torch.FloatTensor(Xtr_n),
        torch.FloatTensor(Y_joy_tr),
        torch.FloatTensor(Y_trig_tr),
        torch.FloatTensor(Y_gate_tr),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(Xval_n),
        torch.FloatTensor(Y_joy_val),
        torch.FloatTensor(Y_trig_val),
        torch.FloatTensor(Y_gate_val),
    )
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=256)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5, min_lr=1e-6)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()

    best_score = -999
    best_state = None
    no_imp = 0

    for ep in range(epochs):
        model.train()
        for xb, yj, yt, yg in train_dl:
            xb = xb.to(DEVICE)
            yj, yt, yg = yj.to(DEVICE), yt.to(DEVICE), yg.to(DEVICE)

            joy_pred, trig_pred, gate_pred = model(xb)
            loss = mse_fn(joy_pred, yj) + mse_fn(trig_pred, yt) + 0.5 * bce_fn(gate_pred, yg)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validation
        model.eval()
        all_joy_p, all_joy_t = [], []
        all_trig_p, all_trig_t = [], []
        all_gate_p, all_gate_t = [], []
        with torch.no_grad():
            for xb, yj, yt, yg in val_dl:
                xb = xb.to(DEVICE)
                jp, tp, gp = model(xb)
                all_joy_p.append(jp.cpu().numpy())
                all_joy_t.append(yj.numpy())
                all_trig_p.append(tp.cpu().numpy())
                all_trig_t.append(yt.numpy())
                all_gate_p.append(gp.cpu().numpy())
                all_gate_t.append(yg.numpy())

        jp = np.concatenate(all_joy_p)
        jt = np.concatenate(all_joy_t)
        tp = np.concatenate(all_trig_p)
        tt = np.concatenate(all_trig_t)
        gp = np.concatenate(all_gate_p)
        gt = np.concatenate(all_gate_t)

        joy_r2 = r2_score(jt, jp, multioutput='uniform_average')
        trig_r2 = r2_score(tt, tp, multioutput='uniform_average')
        gate_acc = accuracy_score(gt.ravel() > 0.5, gp.ravel() > 0.5)

        # Combined score for early stopping
        score = joy_r2 + 0.5 * trig_r2 + 0.3 * gate_acc
        val_loss = mse_fn(torch.FloatTensor(jp), torch.FloatTensor(jt)).item()
        sched.step(val_loss)

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {'joy_r2': joy_r2, 'trig_r2': trig_r2, 'gate_acc': gate_acc}
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 10 == 0:
            per_ax = [r2_score(jt[:, i], jp[:, i]) for i in range(4)]
            log(f"  Ep {ep+1:3d}: joy_R²={joy_r2:.4f} "
                f"[{per_ax[0]:.3f},{per_ax[1]:.3f},{per_ax[2]:.3f},{per_ax[3]:.3f}] "
                f"trig_R²={trig_r2:.4f} gate_acc={gate_acc:.3f} "
                f"lr={opt.param_groups[0]['lr']:.1e} pat={no_imp}")

        if no_imp >= 30:
            log(f"  Early stop at ep {ep+1}")
            break

    model.load_state_dict(best_state)
    log(f"\n  BEST: joy_R²={best_metrics['joy_r2']:.4f}, "
        f"trig_R²={best_metrics['trig_r2']:.4f}, gate_acc={best_metrics['gate_acc']:.4f}")

    return model, best_metrics, m, s


# ── ONNX Export ───────────────────────────────────────────────────────────

class DecoderForExport(nn.Module):
    """Wrapper that outputs a single tensor for ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        joy, trig, gate = self.model(x)
        return torch.cat([joy, trig, gate], dim=1)  # (batch, 7)


def export_onnx(model, input_dim, feat_mean, feat_std, channel_stds):
    """Export model to ONNX and save normalization parameters."""
    model = model.cpu().eval()
    export_model = DecoderForExport(model)
    export_model.eval()

    dummy = torch.randn(1, input_dim)
    out = export_model(dummy)
    log(f"Export test: input {dummy.shape} → output {out.shape}")

    onnx_path = 'models/decoder.onnx'
    os.makedirs('models', exist_ok=True)
    torch.onnx.export(
        export_model, dummy, onnx_path,
        input_names=['neural_features'],
        output_names=['controller_output'],
        dynamic_axes={'neural_features': {0: 'batch'}, 'controller_output': {0: 'batch'}},
        opset_version=13,
    )
    log(f"Saved ONNX model: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    # Save normalization params
    np.savez(f'{OUT_DIR}/feat_norm.npz',
             feat_mean=feat_mean, feat_std=feat_std, channel_stds=channel_stds)
    log(f"Saved normalization: {OUT_DIR}/feat_norm.npz")

    # Save metadata
    meta = {
        'input_dim': input_dim,
        'window': WINDOW,
        'bin_ms': BIN_MS,
        'sample_rate': SAMPLE_RATE,
        'output_names': ['joy_x', 'joy_y', 'rot', 'depth', 'lt', 'rt', 'gate'],
        'output_ranges': [[-1,1], [-1,1], [-1,1], [-1,1], [0,1], [0,1], [0,1]],
    }
    with open(f'{OUT_DIR}/model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    log(f"Saved metadata: {OUT_DIR}/model_meta.json")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    log(f"Decoder v6 — Device: {DEVICE}")
    log(f"Bin: {BIN_MS}ms ({BIN_SAMPLES} samples), Window: {WINDOW} bins ({WINDOW*BIN_MS}ms)")

    # Find all training files
    files = sorted(glob.glob('data/recordings/train_*.h5/broadband_data_*.h5'))
    log(f"\nFound {len(files)} training files")

    # Stratified split
    train_files, val_files = get_stratified_split(files)

    # ── Track A: Spike-count features ────────────────────────────────────
    log(f"\n{'='*60}")
    log("TRACK A: Spike-count features (192/bin)")
    log(f"{'='*60}")

    log("\nExtracting train features (Track A)...")
    train_feats_a, train_labels, channel_stds = extract_all(train_files, track='A')
    log("Extracting val features (Track A)...")
    val_feats_a, val_labels, _ = extract_all(val_files, track='A')

    # Create windows (stride=1 for train, stride=WINDOW for val to avoid overlap inflation)
    Xtr_a, Ytr = make_windows_per_file(train_feats_a, train_labels, stride=1)
    Xval_a, Yval = make_windows_per_file(val_feats_a, val_labels, stride=WINDOW)
    log(f"Track A windows — Train: {Xtr_a.shape}, Val: {Xval_a.shape}")

    # XGBoost baseline on Track A
    xgb_a = run_xgboost_baseline(Xtr_a, Ytr, Xval_a, Yval, "Track A (spike counts)")

    # ── Track B: Rich features ───────────────────────────────────────────
    log(f"\n{'='*60}")
    log("TRACK B: Rich features (768/bin)")
    log(f"{'='*60}")

    log("\nExtracting train features (Track B)...")
    train_feats_b, _, _ = extract_all(train_files, track='B')
    log("Extracting val features (Track B)...")
    val_feats_b, _, _ = extract_all(val_files, track='B')

    Xtr_b, _ = make_windows_per_file(train_feats_b, train_labels, stride=1)
    Xval_b, _ = make_windows_per_file(val_feats_b, val_labels, stride=WINDOW)
    log(f"Track B windows — Train: {Xtr_b.shape}, Val: {Xval_b.shape}")

    # XGBoost baseline on Track B
    xgb_b = run_xgboost_baseline(Xtr_b, Ytr, Xval_b, Yval, "Track B (rich features)")

    # ── Decide which track to use ────────────────────────────────────────
    log(f"\n{'='*60}")
    log("TRACK COMPARISON")
    log(f"{'='*60}")
    log(f"Track A joy_R²: {xgb_a['joy_r2']:.4f}")
    log(f"Track B joy_R²: {xgb_b['joy_r2']:.4f}")
    diff = xgb_b['joy_r2'] - xgb_a['joy_r2']
    log(f"Difference: {diff:+.4f}")

    if diff > 0.05:
        best_track = 'B'
        Xtr, Xval = Xtr_b, Xval_b
        log("→ Using Track B (rich features) — significantly better")
    else:
        best_track = 'A'
        Xtr, Xval = Xtr_a, Xval_a
        log("→ Using Track A (spike counts) — simpler deployment, similar performance")

    # Free unused track
    if best_track == 'A':
        del Xtr_b, Xval_b, train_feats_b, val_feats_b
    else:
        del Xtr_a, Xval_a, train_feats_a, val_feats_a
    gc.collect()

    # ── Train MLP ────────────────────────────────────────────────────────
    model, mlp_metrics, feat_mean, feat_std = train_mlp(
        Xtr, Ytr, Xval, Yval, f"Track {best_track}"
    )

    # ── Compare MLP vs XGBoost ───────────────────────────────────────────
    xgb_best = xgb_b if best_track == 'B' else xgb_a
    log(f"\n{'='*60}")
    log("FINAL COMPARISON")
    log(f"{'='*60}")
    log(f"XGBoost joy_R²: {xgb_best['joy_r2']:.4f}")
    log(f"MLP     joy_R²: {mlp_metrics['joy_r2']:.4f}")
    log(f"XGBoost trig_R²: {xgb_best['trig_r2']:.4f}")
    log(f"MLP     trig_R²: {mlp_metrics['trig_r2']:.4f}")
    log(f"XGBoost gate_acc: {xgb_best['gate_acc']:.4f}")
    log(f"MLP     gate_acc: {mlp_metrics['gate_acc']:.4f}")

    # ── Export ONNX ──────────────────────────────────────────────────────
    export_onnx(model, Xtr.shape[1], feat_mean, feat_std, channel_stds)

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        'track': best_track,
        'xgb_track_a': xgb_a,
        'xgb_track_b': xgb_b,
        'mlp': mlp_metrics,
        'window': WINDOW,
        'bin_ms': BIN_MS,
        'n_train': len(Xtr),
        'n_val': len(Xval),
        'input_dim': Xtr.shape[1],
        'total_time': time.time() - t_start,
    }
    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    log(f"\nTotal time: {time.time() - t_start:.0f}s")
    log(f"Results saved to {OUT_DIR}/")
    log("Done!")


if __name__ == '__main__':
    main()
