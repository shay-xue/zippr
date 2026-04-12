"""
Decoder v8 — GRU with anti-lag fixes.

Fixes over v7:
  1. Shorter sequence (15 bins / 150ms) — less temporal smoothing
  2. Transition-weighted loss — upweight samples where joystick is changing direction
  3. Temporal attention — learned weighted sum of all GRU timesteps (not just last h_n)

Robot arm mapping:
  L-Stick X → arm X     | L-Stick Y → arm Y
  R-Stick X → clamp rot | R-Stick Y → arm Z depth
  LT → open clamp       | RT → close clamp
  A/B/X/Y/LB/RB → gate (suppress joystick when pressed)
"""

import h5py, numpy as np, glob, os, time, json
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, accuracy_score
from scipy.signal import butter, sosfiltfilt

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

SAMPLE_RATE = 32000
N_NEURAL = 64
BIN_MS = 10
BIN_SAMPLES = int(SAMPLE_RATE * BIN_MS / 1000)  # 320
SEQ_LEN = 15       # 150ms context (Fix 1: shorter = less lag)
HIDDEN_SIZE = 192
N_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 128
LR = 5e-4
GRAD_CLIP = 1.0
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

label_names = ['LStickX', 'LStickY', 'RStickX', 'RStickY',
               'A', 'B', 'X', 'Y', 'LB', 'RB', 'LT', 'RT']

OUT_DIR = 'analysis/decoder_v8'
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

    def get_id(f):
        basename = os.path.basename(os.path.dirname(f))
        return basename.split('_')[1]

    file_ids = {get_id(f): f for f in files}
    train_files, val_files = [], []

    for cat, ids in FILE_CATEGORIES.items():
        available = [file_ids[i] for i in ids if i in file_ids]
        if len(available) <= 1:
            train_files.extend(available)
        else:
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
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data, axis=0).astype(np.float32)


def extract_features(neural_bins, channel_stds):
    """Spike counts at 3/4/5σ thresholds. Shape: (n_bins, 192)"""
    n_bins, bs, n_ch = neural_bins.shape
    parts = []
    for thresh in [3.0, 4.0, 5.0]:
        threshold = thresh * channel_stds
        above = neural_bins > threshold[None, None, :]
        below = neural_bins < -threshold[None, None, :]
        spikes = (above | below).astype(np.float32).sum(1)
        parts.append(spikes)
    return np.concatenate(parts, axis=1)


def load_file(filepath):
    """Load HDF5, return neural bins (n_bins, BIN_SAMPLES, 64) and labels (n_bins, 12)."""
    with h5py.File(filepath, 'r') as hf:
        raw = hf['acquisition/ElectricalSeries'][:]
    n = len(raw) // 76
    d = raw[:n * 76].reshape(n, 76)
    neural = d[:, :64].astype(np.float32)
    labels = d[:, 64:76].astype(np.float32)
    del raw, d

    neural = bandpass_filter(neural)

    nb = len(neural) // BIN_SAMPLES
    neural_bins = neural[:nb * BIN_SAMPLES].reshape(nb, BIN_SAMPLES, 64)
    label_bins = labels[:nb * BIN_SAMPLES].reshape(nb, BIN_SAMPLES, 12).mean(1)

    return neural_bins, label_bins


def extract_all(files, channel_stds=None):
    """Extract features from files. Returns per-file lists + channel_stds."""
    if channel_stds is None:
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
        channel_stds = np.mean(all_stds, axis=0)
        log(f"  Channel std range: [{channel_stds.min():.1f}, {channel_stds.max():.1f}]")

    all_feats, all_labels = [], []
    for i, f in enumerate(files):
        neural_bins, label_bins = load_file(f)
        feats = extract_features(neural_bins, channel_stds)
        all_feats.append(feats)
        all_labels.append(label_bins)
        del neural_bins
        if (i + 1) % 8 == 0:
            log(f"  Extracted {i+1}/{len(files)} files")

    log(f"  Extracted {len(files)} files total")
    return all_feats, all_labels, channel_stds


def process_labels(Y):
    """Split labels into joystick, trigger, gate targets."""
    Y_joy = Y[:, :4] / 32767.0
    Y_trig = Y[:, 10:12] / 32767.0
    Y_trig = np.clip(Y_trig, 0, 1)
    Y_btn = (Y[:, 4:10] > 16000).astype(np.float32)
    Y_gate = (Y_btn.sum(1) == 0).astype(np.float32).reshape(-1, 1)
    return Y_joy, Y_trig, Y_btn, Y_gate


def process_labels_with_prev(y_curr, y_prev, device):
    """Process labels, returning absolute targets + prev joy for transition weighting."""
    y_joy = y_curr[:, :4] / 32767.0
    joy_prev = y_prev[:, :4] / 32767.0

    y_trig = y_curr[:, 10:12] / 32767.0
    y_trig = torch.clamp(y_trig, 0, 1)
    y_btn = (y_curr[:, 4:10] > 16000).float()
    y_gate = (y_btn.sum(1) == 0).float().unsqueeze(1)

    return y_joy.to(device), y_trig.to(device), y_gate.to(device), joy_prev.to(device)


# ── Sequence Dataset ─────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Returns (sequence, current_label, prev_label) for transition weighting.
    No cross-file sequences. Starts at index 1 so prev_label is always valid."""
    def __init__(self, feats_list, labels_list, seq_len=SEQ_LEN, stride=1):
        self.feats = feats_list
        self.labels = labels_list
        self.seq_len = seq_len
        self.index_map = []
        for fi, feats in enumerate(feats_list):
            n_bins = len(feats)
            if n_bins < seq_len + 1:
                continue
            for start in range(1, n_bins - seq_len + 1, stride):
                self.index_map.append((fi, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fi, start = self.index_map[idx]
        x = self.feats[fi][start : start + self.seq_len]
        y_curr = self.labels[fi][start + self.seq_len - 1]
        y_prev = self.labels[fi][start + self.seq_len - 2]
        return torch.FloatTensor(x), torch.FloatTensor(y_curr), torch.FloatTensor(y_prev)


# ── GRU Model ────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Fix 4: Learned weighted sum over GRU timesteps.
    Learns which timesteps matter most — allows focusing on recent bins."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, gru_output):
        # gru_output: (batch, seq_len, hidden)
        scores = self.attn(gru_output)          # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = (gru_output * weights).sum(dim=1)  # (batch, hidden)
        return context, weights


class GRUDecoder(nn.Module):
    def __init__(self, input_dim, hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size) if input_dim != hidden_size else nn.Identity()
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.temporal_attn = TemporalAttention(hidden_size)  # Fix 4
        self.joy_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.GELU(),
            nn.Linear(64, 4), nn.Tanh()  # absolute position [-1, 1]
        )
        self.trig_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.GELU(),
            nn.Linear(32, 2), nn.Sigmoid()
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_proj(x)
        output, _ = self.gru(x)                     # output: (B, T, H)
        z, attn_weights = self.temporal_attn(output) # z: (B, H), Fix 4
        return self.joy_head(z), self.trig_head(z), self.gate_head(z), attn_weights


# ── Training ─────────────────────────────────────────────────────────────

TRANSITION_WEIGHT = 5.0   # Fix 3: multiplier for samples with large joystick changes
TRANSITION_THRESH = 0.05  # Fix 3: delta threshold (normalized) to count as transition

def compute_transition_weights(joy_target, joy_prev):
    """Fix 2: Upweight samples where joystick is changing direction."""
    delta = torch.abs(joy_target - joy_prev)     # (B, 4)
    max_delta = delta.max(dim=1).values           # (B,)
    is_transition = (max_delta > TRANSITION_THRESH).float()
    weights = 1.0 + is_transition * (TRANSITION_WEIGHT - 1.0)  # 1x normal, 5x transition
    return weights  # (B,)


def train_rnn(feats_tr, labels_tr, feats_val, labels_val, epochs=300):
    """Train GRU decoder with all anti-lag fixes."""

    # Normalize features
    all_train = np.concatenate(feats_tr)
    feat_mean = all_train.mean(0)
    feat_std = all_train.std(0) + 1e-8
    del all_train

    feats_tr_n = [(f - feat_mean) / feat_std for f in feats_tr]
    feats_val_n = [(f - feat_mean) / feat_std for f in feats_val]

    # Create datasets (now returns curr + prev labels)
    train_ds = SequenceDataset(feats_tr_n, labels_tr, seq_len=SEQ_LEN, stride=1)
    val_ds = SequenceDataset(feats_val_n, labels_val, seq_len=SEQ_LEN, stride=SEQ_LEN)
    log(f"Dataset sizes — Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=256, num_workers=0)

    input_dim = feats_tr[0].shape[1]
    model = GRUDecoder(input_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    log(f"\n{'='*60}")
    log(f"GRU Training (v8 — anti-lag fixes)")
    log(f"{'='*60}")
    log(f"Input dim: {input_dim}, Hidden: {HIDDEN_SIZE}, Layers: {N_LAYERS}")
    log(f"Seq len: {SEQ_LEN} ({SEQ_LEN * BIN_MS}ms), Parameters: {n_params:,}")
    log(f"Transition weight: {TRANSITION_WEIGHT}x")
    log(f"Device: {DEVICE}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2, eta_min=1e-6)
    bce_fn = nn.BCELoss()

    best_score = -999
    best_state = None
    best_metrics = {}
    no_imp = 0

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, y_curr, y_prev in train_dl:
            xb = xb.to(DEVICE)
            xb = xb + 0.05 * torch.randn_like(xb)

            y_joy, y_trig, y_gate, joy_prev = process_labels_with_prev(y_curr, y_prev, DEVICE)
            joy_p, trig_p, gate_p, _ = model(xb)

            # Fix 2: transition-weighted MSE for joystick
            tw = compute_transition_weights(y_joy, joy_prev).to(DEVICE)
            joy_mse = ((joy_p - y_joy) ** 2).mean(dim=1)
            weighted_joy_loss = (joy_mse * tw).mean()

            trig_loss = ((trig_p - y_trig) ** 2).mean()
            gate_loss = bce_fn(gate_p, y_gate)
            loss = weighted_joy_loss + trig_loss + 0.5 * gate_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            train_loss += loss.item()
            n_batches += 1

        sched.step()

        # Validation
        model.eval()
        all_joy_p, all_joy_t = [], []
        all_trig_p, all_trig_t = [], []
        all_gate_p, all_gate_t = [], []
        with torch.no_grad():
            for xb, y_curr, y_prev in val_dl:
                xb = xb.to(DEVICE)
                y_joy, y_trig, y_gate, _ = process_labels_with_prev(y_curr, y_prev, DEVICE)
                jp, tp, gp, _ = model(xb)

                all_joy_p.append(jp.cpu().numpy())
                all_joy_t.append(y_joy.cpu().numpy())
                all_trig_p.append(tp.cpu().numpy())
                all_trig_t.append(y_trig.cpu().numpy())
                all_gate_p.append(gp.cpu().numpy())
                all_gate_t.append(y_gate.cpu().numpy())

        jp = np.concatenate(all_joy_p)
        jt = np.concatenate(all_joy_t)
        tp = np.concatenate(all_trig_p)
        tt = np.concatenate(all_trig_t)
        gp = np.concatenate(all_gate_p)
        gt = np.concatenate(all_gate_t)

        joy_r2 = r2_score(jt, jp, multioutput='uniform_average')
        trig_r2 = r2_score(tt, tp, multioutput='uniform_average')
        gate_acc = accuracy_score(gt.ravel() > 0.5, gp.ravel() > 0.5)

        score = joy_r2 + 0.5 * trig_r2 + 0.3 * gate_acc

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                'joy_r2': float(joy_r2), 'trig_r2': float(trig_r2),
                'gate_acc': float(gate_acc), 'epoch': ep + 1,
                'joy_r2_per_axis': [float(r2_score(jt[:, i], jp[:, i])) for i in range(4)],
            }
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 10 == 0:
            per_ax = [r2_score(jt[:, i], jp[:, i]) for i in range(4)]
            log(f"  Ep {ep+1:3d}: loss={train_loss/n_batches:.4f} joy_R²={joy_r2:.4f} "
                f"[{per_ax[0]:.3f},{per_ax[1]:.3f},{per_ax[2]:.3f},{per_ax[3]:.3f}] "
                f"trig_R²={trig_r2:.4f} gate={gate_acc:.3f} "
                f"lr={opt.param_groups[0]['lr']:.1e} pat={no_imp}")

        if no_imp >= 40:
            log(f"  Early stop at ep {ep+1}")
            break

    model.load_state_dict(best_state)
    log(f"\n  BEST (ep {best_metrics['epoch']}): joy_R²={best_metrics['joy_r2']:.4f} "
        f"{best_metrics['joy_r2_per_axis']}, "
        f"trig_R²={best_metrics['trig_r2']:.4f}, gate_acc={best_metrics['gate_acc']:.4f}")

    return model, best_metrics, feat_mean, feat_std


# ── ONNX Export ──────────────────────────────────────────────────────────

class GRUDecoderForExport(nn.Module):
    """Wrapper: accepts device convention (1, n_features, seq_len), transposes for GRU."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: (batch, 192, 15) → (batch, 15, 192)
        x = x.permute(0, 2, 1)
        joy, trig, gate, _ = self.model(x)
        return torch.cat([joy, trig, gate], dim=1)  # (batch, 7)


def export_onnx(model, input_dim, feat_mean, feat_std, channel_stds):
    model = model.cpu().eval()
    export_model = GRUDecoderForExport(model)
    export_model.eval()

    # Device convention: (1, n_features, seq_len)
    dummy = torch.randn(1, input_dim, SEQ_LEN)
    out = export_model(dummy)
    log(f"Export test: input {dummy.shape} → output {out.shape}")

    onnx_path = 'models/decoder.onnx'
    os.makedirs('models', exist_ok=True)
    torch.onnx.export(
        export_model, dummy, onnx_path,
        input_names=['neural_features'],
        output_names=['controller_output'],
        dynamic_axes={'neural_features': {0: 'batch'}, 'controller_output': {0: 'batch'}},
        opset_version=14,
    )
    log(f"Saved ONNX model: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    # Verify ONNX matches PyTorch
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        for i in range(5):
            test_input = np.random.randn(1, input_dim, SEQ_LEN).astype(np.float32)
            torch_out = export_model(torch.FloatTensor(test_input)).detach().numpy()
            onnx_out = sess.run(None, {'neural_features': test_input})[0]
            max_diff = np.max(np.abs(torch_out - onnx_out))
            assert max_diff < 1e-4, f"ONNX mismatch: {max_diff}"
        log(f"ONNX verification passed (max diff < 1e-4)")
    except ImportError:
        log("onnxruntime not installed, skipping verification")

    # Save normalization params
    np.savez(f'{OUT_DIR}/feat_norm.npz',
             feat_mean=feat_mean, feat_std=feat_std, channel_stds=channel_stds)
    log(f"Saved normalization: {OUT_DIR}/feat_norm.npz")

    meta = {
        'model_type': 'gru_v8_antilag',
        'input_dim': input_dim,
        'input_shape': [1, input_dim, SEQ_LEN],
        'input_convention': '(batch, n_features, seq_len) — transposed internally',
        'output_shape': [1, 7],
        'seq_len': SEQ_LEN,
        'bin_ms': BIN_MS,
        'sample_rate': SAMPLE_RATE,
        'hidden_size': HIDDEN_SIZE,
        'n_layers': N_LAYERS,
        'feature_type': 'spike_counts_3thresh',
        'sigma_thresholds': [3.0, 4.0, 5.0],
        'fixes': ['shorter_seq_15', 'transition_weighted_loss', 'temporal_attention'],
        'output_names': ['joy_x', 'joy_y', 'rot', 'depth', 'lt', 'rt', 'gate'],
        'output_ranges': [[-1,1], [-1,1], [-1,1], [-1,1], [0,1], [0,1], [0,1]],
    }
    with open(f'{OUT_DIR}/model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    log(f"Saved metadata: {OUT_DIR}/model_meta.json")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    log(f"Decoder v8 (GRU + anti-lag) — Device: {DEVICE}")
    log(f"Bin: {BIN_MS}ms ({BIN_SAMPLES} samples), Seq: {SEQ_LEN} bins ({SEQ_LEN*BIN_MS}ms)")
    log(f"GRU: hidden={HIDDEN_SIZE}, layers={N_LAYERS}, dropout={DROPOUT}")
    log(f"Fixes: shorter_seq={SEQ_LEN}, transition_weight={TRANSITION_WEIGHT}x, temporal_attn")

    # Find training files
    files = sorted(glob.glob('data/recordings/train_*.h5/broadband_data_*.h5'))
    log(f"\nFound {len(files)} training files")

    if len(files) == 0:
        log("ERROR: No training files found. Check data/recordings/")
        return

    # Stratified split
    train_files, val_files = get_stratified_split(files)

    # Extract features
    log(f"\nExtracting train features...")
    train_feats, train_labels, channel_stds = extract_all(train_files)
    log(f"Extracting val features...")
    val_feats, val_labels, _ = extract_all(val_files, channel_stds=channel_stds)

    total_train_bins = sum(len(f) for f in train_feats)
    total_val_bins = sum(len(f) for f in val_feats)
    log(f"Total bins — Train: {total_train_bins}, Val: {total_val_bins}")

    # GRU training
    model, gru_metrics, feat_mean, feat_std = train_rnn(
        train_feats, train_labels, val_feats, val_labels
    )

    # Export ONNX
    export_onnx(model, train_feats[0].shape[1], feat_mean, feat_std, channel_stds)

    # Save results
    results = {
        'gru': gru_metrics,
        'config': {
            'seq_len': SEQ_LEN, 'hidden_size': HIDDEN_SIZE,
            'n_layers': N_LAYERS, 'dropout': DROPOUT,
            'lr': LR, 'batch_size': BATCH_SIZE,
            'bin_ms': BIN_MS, 'feature_type': 'spike_counts_3thresh',
            'transition_weight': TRANSITION_WEIGHT,
            'fixes': ['shorter_seq_15', 'transition_weighted_loss', 'temporal_attention'],
        },
        'data': {
            'n_train_files': len(train_files),
            'n_val_files': len(val_files),
            'n_train_bins': total_train_bins,
            'n_val_bins': total_val_bins,
        },
        'total_time': time.time() - t_start,
    }
    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    log(f"\nTotal time: {time.time() - t_start:.0f}s")
    log(f"Results saved to {OUT_DIR}/")
    log("Done!")


if __name__ == '__main__':
    main()
