"""
Decoder v5 — 1D CNN on raw waveforms.

Key differences from v4:
  - No hand-crafted features — 1D CNN learns from raw 32kHz waveforms
  - File-based train/test split (no data leakage from overlapping windows)
  - In-memory contiguous arrays with on-the-fly window slicing (fast + bounded memory)
"""

import h5py, numpy as np, glob, os, gc, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, accuracy_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

SAMPLE_RATE = 32000
N_NEURAL = 64
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def log(msg):
    print(msg, flush=True)


# ── Dataset: contiguous array + on-the-fly windowing ───────────────────────

class ContiguousWindowDataset(Dataset):
    """Stores raw data as contiguous arrays, creates windows on-the-fly.

    Memory: only stores raw samples (no window materialization).
    Speed: contiguous array slicing is fast (no disk I/O).
    """
    def __init__(self, neural, labels, window_samples, stride_samples, ch_mean, ch_std):
        """
        neural: (total_samples, 64) float32, contiguous
        labels: (total_samples, 12) float32, contiguous
        """
        self.neural = neural
        self.labels = labels
        self.window_samples = window_samples
        self.ch_mean = ch_mean.reshape(1, 64)   # (1, 64)
        self.ch_std = ch_std.reshape(1, 64)     # (1, 64)

        self.n_windows = (len(neural) - window_samples) // stride_samples + 1
        self.stride = stride_samples

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_samples

        # Slice window from contiguous array (fast, no copy until .T)
        x = self.neural[start:end]  # (window_samples, 64)
        # Normalize and transpose to (64, window_samples)
        x = ((x - self.ch_mean) / self.ch_std).T

        # Label: average over window
        y = self.labels[start:end].mean(0)  # (12,)

        return torch.from_numpy(x.copy().astype(np.float32)), torch.from_numpy(y.astype(np.float32))


def load_files_contiguous(file_list):
    """Load files into one contiguous array."""
    all_neural = []
    all_labels = []
    total = 0
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            raw = hf['acquisition/ElectricalSeries'][:]
        n = len(raw) // 76
        d = raw[:n * 76].reshape(n, 76)
        all_neural.append(d[:, :64].astype(np.float32))
        all_labels.append(d[:, 64:76].astype(np.float32))
        total += n
        del raw, d
    neural = np.concatenate(all_neural); del all_neural
    labels = np.concatenate(all_labels); del all_labels
    gc.collect()
    return neural, labels


def process_labels_array(Y):
    Y_joy = Y[:, :4] / 32767.0
    Y_trig = Y[:, 10:12] / 32767.0
    Y_trig = np.clip(Y_trig, 0, 1)
    Y_btn = (Y[:, 4:10] > 16000).astype(np.float32)
    return Y_joy, Y_trig, Y_btn


# ── Models ─────────────────────────────────────────────────────────────────

class Conv1DEncoder(nn.Module):
    """1D CNN: (batch, 64, 3200) → (batch, feat_dim)"""
    def __init__(self, in_channels=64, window_samples=3200):
        super().__init__()
        self.conv = nn.Sequential(
            # 3200 → 800
            nn.Conv1d(in_channels, 128, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(128), nn.GELU(),
            # 800 → 200
            nn.Conv1d(128, 128, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(128), nn.GELU(),
            # 200 → 50
            nn.Conv1d(128, 256, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(256), nn.GELU(),
            # 50 → 13
            nn.Conv1d(256, 256, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(256), nn.GELU(),
        )
        with torch.no_grad():
            out = self.conv(torch.zeros(1, in_channels, window_samples))
            self.feat_dim = out.shape[1] * out.shape[2]

    def forward(self, x):
        return self.conv(x).flatten(1)


class JoystickDecoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.feat_dim, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 4), nn.Tanh()
        )

    def forward(self, x):
        return self.head(self.encoder(x))


class TriggerDecoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.feat_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 2), nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(self.encoder(x))


class ButtonGate(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.shared = nn.Sequential(
            nn.Linear(encoder.feat_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.btn_head = nn.Linear(64, 6)
        self.gate_head = nn.Sequential(nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        f = self.shared(self.encoder(x))
        return self.btn_head(f), self.gate_head(f)


# ── Training ───────────────────────────────────────────────────────────────

def train_joystick(model, train_dl, val_dl, epochs=200):
    log("\nTraining Joystick Decoder (1D CNN)...")
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    best_r2 = -999; best_state = None; no_imp = 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        for xb, yb_full in train_dl:
            yb = yb_full[:, :4] / 32767.0
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb_full in val_dl:
                yb = yb_full[:, :4] / 32767.0
                preds.append(model(xb.to(DEVICE)).cpu().numpy())
                trues.append(yb.numpy())
        p, t = np.concatenate(preds), np.concatenate(trues)
        r2 = r2_score(t, p, multioutput='uniform_average')
        vl = loss_fn(torch.FloatTensor(p), torch.FloatTensor(t)).item()
        sched.step(vl)

        if r2 > best_r2:
            best_r2 = r2; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 5 == 0:
            per_ax = [r2_score(t[:, i], p[:, i]) for i in range(4)]
            log(f"  Ep {ep+1}: R²={r2:.4f} best={best_r2:.4f} "
                f"[{per_ax[0]:.3f},{per_ax[1]:.3f},{per_ax[2]:.3f},{per_ax[3]:.3f}] "
                f"lr={opt.param_groups[0]['lr']:.1e} pat={no_imp} ({time.time()-t0:.1f}s)")
        if no_imp >= 40:
            log(f"  Early stop at {ep+1}"); break

    model.load_state_dict(best_state)
    return model, best_r2


def train_triggers(model, train_dl, val_dl, epochs=200):
    log("\nTraining Trigger Decoder (1D CNN)...")
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    best_r2 = -999; best_state = None; no_imp = 0
    for ep in range(epochs):
        model.train()
        for xb, yb_full in train_dl:
            yb = yb_full[:, 10:12] / 32767.0
            yb = yb.clamp(0, 1)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb_full in val_dl:
                yb = yb_full[:, 10:12] / 32767.0
                yb = yb.clamp(0, 1)
                preds.append(model(xb.to(DEVICE)).cpu().numpy())
                trues.append(yb.numpy())
        p, t = np.concatenate(preds), np.concatenate(trues)
        r2 = r2_score(t, p, multioutput='uniform_average')
        vl = loss_fn(torch.FloatTensor(p), torch.FloatTensor(t)).item()
        sched.step(vl)

        if r2 > best_r2:
            best_r2 = r2; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 5 == 0:
            log(f"  Ep {ep+1}: R²={r2:.4f} best={best_r2:.4f} "
                f"[LT={r2_score(t[:,0],p[:,0]):.3f}, RT={r2_score(t[:,1],p[:,1]):.3f}] "
                f"lr={opt.param_groups[0]['lr']:.1e} pat={no_imp}")
        if no_imp >= 40:
            log(f"  Early stop at {ep+1}"); break

    model.load_state_dict(best_state)
    return model, best_r2


def train_gate(model, train_dl, val_dl, epochs=150):
    log("\nTraining Button Gate (1D CNN)...")
    model = model.to(DEVICE)

    # Compute pos weights
    btn_sums = np.zeros(6); btn_count = 0
    for _, yb_full in train_dl:
        yb_btn = (yb_full[:, 4:10] > 16000).float()
        btn_sums += yb_btn.sum(0).numpy()
        btn_count += len(yb_btn)
    btn_rate = btn_sums / btn_count
    pw = torch.FloatTensor(np.clip((1 - btn_rate) / (btn_rate + 1e-6), 0, 10)).to(DEVICE)
    log(f"  Button rates: {btn_rate.round(3)}, pw: {pw.cpu().numpy().round(1)}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5, min_lr=1e-6)
    btn_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    gate_fn = nn.BCELoss()

    best_acc = 0; best_state = None; no_imp = 0
    for ep in range(epochs):
        model.train()
        for xb, yb_full in train_dl:
            yb_btn = (yb_full[:, 4:10] > 16000).float()
            yb_gate = (yb_btn.sum(1) == 0).float().unsqueeze(1)
            xb, yb_btn, yb_gate = xb.to(DEVICE), yb_btn.to(DEVICE), yb_gate.to(DEVICE)
            btn_logits, gate = model(xb)
            loss = btn_fn(btn_logits, yb_btn) + gate_fn(gate, yb_gate)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        all_bp, all_bt, all_gp, all_gt = [], [], [], []
        vl = 0
        with torch.no_grad():
            for xb, yb_full in val_dl:
                yb_btn = (yb_full[:, 4:10] > 16000).float()
                yb_gate = (yb_btn.sum(1) == 0).float().unsqueeze(1)
                xb = xb.to(DEVICE)
                bl, g = model(xb)
                vl += (btn_fn(bl, yb_btn.to(DEVICE)) + gate_fn(g, yb_gate.to(DEVICE))).item()
                all_bp.append((torch.sigmoid(bl) > 0.5).cpu().numpy())
                all_bt.append(yb_btn.numpy())
                all_gp.append((g > 0.5).cpu().numpy())
                all_gt.append(yb_gate.numpy())
        vl /= max(len(val_dl), 1); sched.step(vl)
        bp, bt = np.concatenate(all_bp), np.concatenate(all_bt)
        gp, gt = np.concatenate(all_gp), np.concatenate(all_gt)
        btn_acc = accuracy_score(bt.flatten(), bp.flatten())
        gate_acc = accuracy_score(gt.flatten(), gp.flatten())
        combined = (btn_acc + gate_acc) / 2

        if combined > best_acc:
            best_acc = combined; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 5 == 0:
            log(f"  Ep {ep+1}: btn_acc={btn_acc:.4f} gate_acc={gate_acc:.4f} "
                f"lr={opt.param_groups[0]['lr']:.1e} pat={no_imp}")
        if no_imp >= 40:
            log(f"  Early stop at {ep+1}"); break

    model.load_state_dict(best_state)
    return model


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('analysis/decoder_v5', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    WINDOW_MS = 100
    STRIDE_MS = 50
    BATCH_SIZE = 32

    window_samples = int(SAMPLE_RATE * WINDOW_MS / 1000)
    stride_samples = int(SAMPLE_RATE * STRIDE_MS / 1000)

    log(f"Device: {DEVICE}")
    log(f"Window: {WINDOW_MS}ms ({window_samples} samples), Stride: {STRIDE_MS}ms")

    files = sorted(glob.glob('data/recordings/train_*.h5/broadband_data_*.h5'))
    log(f"Found {len(files)} files")

    # File-based split: every 5th file → val
    val_indices = set(range(2, len(files), 5))  # 2, 7, 12, 17, 22, 27
    train_files = [f for i, f in enumerate(files) if i not in val_indices]
    val_files = [f for i, f in enumerate(files) if i in val_indices]

    for i, f in enumerate(files):
        tag = "VAL " if i in val_indices else "TRAIN"
        fname = os.path.basename(os.path.dirname(f))
        log(f"  [{i:2d}] {tag} {fname}")

    log(f"\nTrain: {len(train_files)} files, Val: {len(val_files)} files")

    # Load into contiguous arrays
    log("Loading training data...")
    neural_train, labels_train = load_files_contiguous(train_files)
    log(f"  Train: {neural_train.shape} = {neural_train.nbytes/1e9:.1f} GB")

    log("Loading validation data...")
    neural_val, labels_val = load_files_contiguous(val_files)
    log(f"  Val: {neural_val.shape} = {neural_val.nbytes/1e9:.1f} GB")

    # Channel normalization from training data
    ch_mean = neural_train.mean(0).astype(np.float32)
    ch_std = (neural_train.std(0) + 1e-8).astype(np.float32)
    np.savez('analysis/decoder_v5/channel_norm.npz', mean=ch_mean, std=ch_std)
    log(f"  Channel mean: [{ch_mean.min():.1f}, {ch_mean.max():.1f}]")
    log(f"  Channel std:  [{ch_std.min():.1f}, {ch_std.max():.1f}]")

    # Create datasets
    train_ds = ContiguousWindowDataset(neural_train, labels_train, window_samples, stride_samples, ch_mean, ch_std)
    val_ds = ContiguousWindowDataset(neural_val, labels_val, window_samples, stride_samples, ch_mean, ch_std)
    log(f"  Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    # Memory check
    import psutil
    mem = psutil.virtual_memory()
    log(f"  Memory used: {mem.used/1e9:.1f} GB / {mem.total/1e9:.1f} GB ({mem.percent}%)")

    # ── 1. Train Joystick ──
    encoder_joy = Conv1DEncoder(N_NEURAL, window_samples)
    log(f"  Conv feat_dim: {encoder_joy.feat_dim}")
    joy_model = JoystickDecoder(encoder_joy)
    log(f"Joystick params: {sum(p.numel() for p in joy_model.parameters()):,}")

    joy_model, joy_r2 = train_joystick(joy_model, train_dl, val_dl)
    joy_model.cpu()
    torch.save(joy_model.state_dict(), 'analysis/decoder_v5/joystick.pt')
    log(f"  Saved joystick.pt (R²={joy_r2:.4f})")
    gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

    # ── 2. Train Triggers ──
    encoder_trig = Conv1DEncoder(N_NEURAL, window_samples)
    trig_model = TriggerDecoder(encoder_trig)
    log(f"Trigger params: {sum(p.numel() for p in trig_model.parameters()):,}")

    trig_model, trig_r2 = train_triggers(trig_model, train_dl, val_dl)
    trig_model.cpu()
    torch.save(trig_model.state_dict(), 'analysis/decoder_v5/trigger.pt')
    log(f"  Saved trigger.pt (R²={trig_r2:.4f})")
    gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

    # ── 3. Train Button Gate ──
    encoder_gate = Conv1DEncoder(N_NEURAL, window_samples)
    gate_model = ButtonGate(encoder_gate)
    log(f"Gate params: {sum(p.numel() for p in gate_model.parameters()):,}")

    gate_model = train_gate(gate_model, train_dl, val_dl)
    gate_model.cpu()
    torch.save(gate_model.state_dict(), 'analysis/decoder_v5/gate.pt')
    log(f"  Saved gate.pt")
    gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

    # ── Final Evaluation ──
    log(f"\n{'='*60}")
    log(f"  FINAL RESULTS (file-based split, no leakage)")
    log(f"{'='*60}")

    joy_model.eval().to(DEVICE)
    trig_model.eval().to(DEVICE)
    gate_model.eval().to(DEVICE)

    eval_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    jp_all, tp_all, bp_all, gp_all, y_all = [], [], [], [], []
    with torch.no_grad():
        for xb, yb in eval_dl:
            xb = xb.to(DEVICE)
            jp_all.append(joy_model(xb).cpu().numpy())
            tp_all.append(trig_model(xb).cpu().numpy())
            bl, g = gate_model(xb)
            bp_all.append((torch.sigmoid(bl) > 0.5).cpu().numpy().astype(float))
            gp_all.append(g.cpu().numpy())
            y_all.append(yb.numpy())

    jp = np.concatenate(jp_all)
    tp = np.concatenate(tp_all)
    bp = np.concatenate(bp_all)
    gp = np.concatenate(gp_all)
    Y_raw = np.concatenate(y_all)

    Yval_joy, Yval_trig, Yval_btn = process_labels_array(Y_raw)
    jp_gated = jp * gp
    gate_true = (Yval_btn.sum(1) == 0).astype(float)

    log("\nJoystick R² (raw / gated):")
    jnames = ['L-X (arm X)', 'L-Y (arm Y)', 'R-X (rotation)', 'R-Y (depth)']
    for i, n in enumerate(jnames):
        r2_raw = r2_score(Yval_joy[:, i], jp[:, i])
        r2_gated = r2_score(Yval_joy[:, i], jp_gated[:, i])
        log(f"  {n}: raw={r2_raw:.4f}  gated={r2_gated:.4f}")
    log(f"  Avg raw:   {r2_score(Yval_joy, jp, multioutput='uniform_average'):.4f}")
    log(f"  Avg gated: {r2_score(Yval_joy, jp_gated, multioutput='uniform_average'):.4f}")

    log(f"\nTrigger R² (analog):")
    for i, n in enumerate(['LT (open)', 'RT (close)']):
        log(f"  {n}: {r2_score(Yval_trig[:, i], tp[:, i]):.4f}")
    log(f"  Avg: {r2_score(Yval_trig, tp, multioutput='uniform_average'):.4f}")

    log(f"\nButton Gate:")
    bnames = ['A', 'B', 'X', 'Y', 'LB', 'RB']
    for i, n in enumerate(bnames):
        log(f"  {n}: {accuracy_score(Yval_btn[:, i], bp[:, i]):.4f} ({int(Yval_btn[:, i].sum())})")
    log(f"  Avg: {accuracy_score(Yval_btn.flatten(), bp.flatten()):.4f}")
    log(f"  Gate accuracy: {accuracy_score(gate_true, (gp > 0.5).flatten()):.4f}")

    # ONNX
    class RobotArmDecoder(nn.Module):
        def __init__(self, joy, trig, gate):
            super().__init__()
            self.joy = joy; self.trig = trig; self.gate_net = gate
        def forward(self, x):
            j = self.joy(x)
            t = self.trig(x)
            bl, g = self.gate_net(x)
            b = torch.sigmoid(bl)
            return torch.cat([j * g, t, b, g], dim=1)

    combined = RobotArmDecoder(joy_model.cpu(), trig_model.cpu(), gate_model.cpu())
    combined.eval()
    dummy = torch.randn(1, 64, window_samples)
    torch.onnx.export(combined, dummy, 'models/decoder_v5.onnx',
        input_names=['raw_neural'], output_names=['arm_control'],
        dynamic_axes={'raw_neural': {0: 'batch'}, 'arm_control': {0: 'batch'}},
        opset_version=13, dynamo=False)
    log(f"\nONNX saved: models/decoder_v5.onnx")
    log(f"Input: (batch, 64, {window_samples}) raw neural")
    log("Output: [joy_x, joy_y, rot, depth, lt, rt, A, B, X, Y, LB, RB, gate]")

    # Plots
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    t_axis = np.arange(len(Yval_joy)) * STRIDE_MS / 1000
    for i, (ax, n) in enumerate(zip(axes, jnames)):
        ax.plot(t_axis, Yval_joy[:, i], 'b-', alpha=.5, lw=.5, label='True')
        ax.plot(t_axis, jp[:, i], 'r-', alpha=.5, lw=.5, label='Pred')
        r2 = r2_score(Yval_joy[:, i], jp[:, i])
        ax.set_title(f'{n} — R² = {r2:.4f}'); ax.legend(loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('v5 1D-CNN Decoder (file-split, no leakage)', fontsize=14)
    plt.tight_layout(); plt.savefig('analysis/decoder_v5/joystick.png', dpi=150); plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
    for i, (ax, n) in enumerate(zip(axes, ['LT (Open Clamp)', 'RT (Close Clamp)'])):
        ax.plot(t_axis, Yval_trig[:, i], 'b-', alpha=.5, lw=.5, label='True')
        ax.plot(t_axis, tp[:, i], 'r-', alpha=.5, lw=.5, label='Predicted')
        r2 = r2_score(Yval_trig[:, i], tp[:, i])
        ax.set_title(f'{n} — R² = {r2:.4f}'); ax.legend()
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('v5 1D-CNN Trigger Decoder', fontsize=14)
    plt.tight_layout(); plt.savefig('analysis/decoder_v5/triggers.png', dpi=150); plt.close()

    log("Plots saved to analysis/decoder_v5/")
    log("Done!")
