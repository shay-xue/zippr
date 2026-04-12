"""
Decoder v4 — Gated architecture for robot arm control.

Robot arm mapping:
  L-Stick X → arm X (side-to-side)       [CRITICAL]
  L-Stick Y → arm Y (up/down)            [CRITICAL]
  R-Stick X → clamp rotation             [CRITICAL]
  R-Stick Y → arm Z (depth)              [CRITICAL]
  LT → open clamp (analog)              [CRITICAL]
  RT → close clamp (analog)             [CRITICAL]
  A, B, X, Y, LB, RB → unused           [SUPPRESS]

Architecture:
  1. Primary decoder: predicts all 12 outputs from neural features
  2. Gate network: detects unused button presses (A/B/X/Y/LB/RB)
  3. When unused buttons are detected, gate suppresses joystick outputs
     to prevent false arm movements caused by button press neural patterns

Also: LT/RT treated as analog regression (not binary classification)
"""

import h5py, numpy as np, glob, os, gc, sys, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

SAMPLE_RATE = 32000
N_NEURAL = 64
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)


def log(msg):
    print(msg, flush=True)


def extract_file_features(filepath, bin_ms=10):
    """Extract features from a single HDF5 file — keeps memory bounded."""
    bs = int(SAMPLE_RATE * bin_ms / 1000)
    with h5py.File(filepath, 'r') as hf:
        raw = hf['acquisition/ElectricalSeries'][:]
    n = len(raw) // 76
    d = raw[:n * 76].reshape(n, 76)
    neural = d[:, :64].astype(np.float32)
    labels = d[:, 64:76].astype(np.float32)
    del raw, d; gc.collect()

    nb = len(neural) // bs
    neural = neural[:nb * bs]
    labels = labels[:nb * bs]

    cm, cs = neural.mean(0), neural.std(0) + 1e-8
    tr = neural.reshape(nb, bs, 64)

    parts = []
    for t in [3.0, 4.0, 5.0]:
        above = neural > (cm + t * cs)
        below = neural < (cm - t * cs)
        spikes = (above | below).astype(np.float32)
        del above, below
        parts.append(spikes.reshape(nb, bs, 64).sum(1))
        del spikes

    parts.extend([tr.mean(1), tr.std(1), tr.max(1) - tr.min(1), np.abs(tr).mean(1)])
    feat = np.concatenate(parts, 1)
    bl = labels.reshape(nb, bs, 12).mean(1)
    del neural, labels, tr, parts; gc.collect()
    return feat, bl


def load_and_extract():
    """Load all files, extract features per-file, then combine. Memory efficient."""
    files = sorted(glob.glob('data/recordings/train_*/broadband_data_*.h5'))
    all_feat, all_bl = [], []
    for i, f in enumerate(files):
        feat, bl = extract_file_features(f)
        all_feat.append(feat)
        all_bl.append(bl)
        if (i + 1) % 8 == 0:
            log(f"  Loaded {i+1}/{len(files)} files")
    log(f"  Loaded {len(files)}/{len(files)} files")
    feat = np.concatenate(all_feat); del all_feat
    bl = np.concatenate(all_bl); del all_bl
    gc.collect()

    total_sec = feat.shape[0] * 0.01
    log(f"{len(files)} files, {total_sec:.1f}s ({total_sec/60:.1f} min), {feat.shape[0]} bins")

    fm, fs = feat.mean(0), feat.std(0) + 1e-8
    feat = (feat - fm) / fs
    log(f"Features: {feat.shape}")
    return feat, bl, fm, fs


def make_windows(feat, bl, w=15):
    X = np.lib.stride_tricks.sliding_window_view(feat, w, axis=0).transpose(0, 2, 1).copy()
    Y = bl[w - 1:]
    del feat, bl; gc.collect()
    return X, Y


def process_labels(Y):
    """
    Returns:
      Y_joy: 4 joystick axes normalized to [-1, 1]
      Y_trig: 2 trigger values normalized to [0, 1] (analog!)
      Y_btn: 6 binary buttons (A, B, X, Y, LB, RB)
    """
    Y_joy = Y[:, :4] / 32767.0           # joystick [-1, 1]
    Y_trig = Y[:, 10:12] / 32767.0       # LT(ch74), RT(ch75) → [0, 1] analog
    Y_trig = np.clip(Y_trig, 0, 1)
    Y_btn = (Y[:, 4:10] > 16000).astype(np.float32)  # A,B,X,Y,LB,RB → binary
    return Y_joy, Y_trig, Y_btn


# ── Models ──────────────────────────────────────────────────────────────────

class JoystickDecoder(nn.Module):
    """Dedicated joystick regression decoder."""
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inp, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 4), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class TriggerDecoder(nn.Module):
    """Dedicated analog trigger decoder (LT open, RT close)."""
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inp, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 2), nn.Sigmoid()  # output in [0, 1]
        )

    def forward(self, x):
        return self.net(x)


class ButtonGate(nn.Module):
    """
    Detects unused button presses (A/B/X/Y/LB/RB).
    Outputs 6 button probabilities AND a gate signal.
    Gate signal is high (=1) when NO unused buttons are pressed → pass through.
    Gate signal is low (→0) when unused buttons ARE pressed → suppress joystick.
    """
    def __init__(self, inp):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inp, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.btn_head = nn.Linear(128, 6)  # 6 button logits
        self.gate_head = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()  # gate in [0, 1]
        )

    def forward(self, x):
        s = self.shared(x)
        btn_logits = self.btn_head(s)
        gate = self.gate_head(s)
        return btn_logits, gate


# ── Training Functions ──────────────────────────────────────────────────────

def train_joystick(model, Xt, Yjt, Xv, Yjv, epochs=400):
    log("\nTraining Joystick Decoder...")
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    tdl = DataLoader(TensorDataset(torch.FloatTensor(Xt), torch.FloatTensor(Yjt)),
                     batch_size=128, shuffle=True, drop_last=True)
    vdl = DataLoader(TensorDataset(torch.FloatTensor(Xv), torch.FloatTensor(Yjv)), batch_size=128)

    best_r2 = -999; best_state = None; no_imp = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tdl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in vdl:
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

        if (ep + 1) % 25 == 0:
            per_ax = [r2_score(t[:, i], p[:, i]) for i in range(4)]
            log(f"  Ep {ep+1}: R²={r2:.4f} best={best_r2:.4f} "
                f"[{per_ax[0]:.3f},{per_ax[1]:.3f},{per_ax[2]:.3f},{per_ax[3]:.3f}] "
                f"lr={opt.param_groups[0]['lr']:.1e} pat={no_imp}")
        if no_imp >= 60:
            log(f"  Early stop at {ep+1}"); break

    model.load_state_dict(best_state)
    return model, best_r2


def train_triggers(model, Xt, Ytt, Xv, Ytv, epochs=400):
    log("\nTraining Trigger Decoder (analog)...")
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    tdl = DataLoader(TensorDataset(torch.FloatTensor(Xt), torch.FloatTensor(Ytt)),
                     batch_size=128, shuffle=True, drop_last=True)
    vdl = DataLoader(TensorDataset(torch.FloatTensor(Xv), torch.FloatTensor(Ytv)), batch_size=128)

    best_r2 = -999; best_state = None; no_imp = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tdl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in vdl:
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

        if (ep + 1) % 25 == 0:
            log(f"  Ep {ep+1}: R²={r2:.4f} best={best_r2:.4f} "
                f"[LT={r2_score(t[:,0],p[:,0]):.3f}, RT={r2_score(t[:,1],p[:,1]):.3f}] "
                f"lr={opt.param_groups[0]['lr']:.1e} pat={no_imp}")
        if no_imp >= 60:
            log(f"  Early stop at {ep+1}"); break

    model.load_state_dict(best_state)
    return model, best_r2


def train_gate(model, Xt, Ybt, Yjt, Xv, Ybv, Yjv, epochs=300):
    """
    Train button gate. Gate target: 1 when NO buttons pressed, 0 when any button pressed.
    """
    log("\nTraining Button Gate...")
    model = model.to(DEVICE)

    # Gate target: 1 if no buttons pressed, 0 if any pressed
    gate_t = (Ybt.sum(1) == 0).astype(np.float32).reshape(-1, 1)
    gate_v = (Ybv.sum(1) == 0).astype(np.float32).reshape(-1, 1)
    log(f"  Gate=1 (no buttons): {gate_t.mean():.1%} train, {gate_v.mean():.1%} val")

    pw = torch.FloatTensor((1 - Ybt.mean(0)) / (Ybt.mean(0) + 1e-6)).clamp(max=10.0)
    log(f"  Button pos_weight: {pw.numpy().round(1)}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5, min_lr=1e-6)
    btn_fn = nn.BCEWithLogitsLoss(pos_weight=pw.to(DEVICE))
    gate_fn = nn.BCELoss()

    tdl = DataLoader(TensorDataset(torch.FloatTensor(Xt), torch.FloatTensor(Ybt), torch.FloatTensor(gate_t)),
                     batch_size=128, shuffle=True, drop_last=True)
    vdl = DataLoader(TensorDataset(torch.FloatTensor(Xv), torch.FloatTensor(Ybv), torch.FloatTensor(gate_v)),
                     batch_size=128)

    best_acc = 0; best_state = None; no_imp = 0
    for ep in range(epochs):
        model.train()
        for xb, bb, gb in tdl:
            xb, bb, gb = xb.to(DEVICE), bb.to(DEVICE), gb.to(DEVICE)
            btn_logits, gate = model(xb)
            loss = btn_fn(btn_logits, bb) + gate_fn(gate, gb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        all_bp, all_bt, all_gp, all_gt = [], [], [], []
        vl = 0
        with torch.no_grad():
            for xb, bb, gb in vdl:
                xb, bb, gb = xb.to(DEVICE), bb.to(DEVICE), gb.to(DEVICE)
                bl, g = model(xb)
                vl += (btn_fn(bl, bb) + gate_fn(g, gb)).item()
                all_bp.append((torch.sigmoid(bl) > 0.5).cpu().numpy())
                all_bt.append(bb.cpu().numpy())
                all_gp.append((g > 0.5).cpu().numpy())
                all_gt.append(gb.cpu().numpy())
        vl /= len(vdl); sched.step(vl)
        bp, bt = np.concatenate(all_bp), np.concatenate(all_bt)
        gp, gt = np.concatenate(all_gp), np.concatenate(all_gt)
        btn_acc = accuracy_score(bt.flatten(), bp.flatten())
        gate_acc = accuracy_score(gt.flatten(), gp.flatten())
        combined = (btn_acc + gate_acc) / 2

        if combined > best_acc:
            best_acc = combined; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 25 == 0:
            log(f"  Ep {ep+1}: btn_acc={btn_acc:.4f} gate_acc={gate_acc:.4f} "
                f"lr={opt.param_groups[0]['lr']:.1e} pat={no_imp}")
        if no_imp >= 60:
            log(f"  Early stop at {ep+1}"); break

    model.load_state_dict(best_state)
    return model


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('analysis/decoder_v4', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load and extract features per-file (memory efficient)
    feat, bl, fm, fs = load_and_extract()
    np.savez('analysis/decoder_v4/feat_norm.npz', mean=fm, std=fs)
    w = 15; nf = feat.shape[1]; inp = w * nf
    X, Y = make_windows(feat, bl, w)
    Y_joy, Y_trig, Y_btn = process_labels(Y)

    log(f"\nJoystick: {Y_joy.shape}, range [{Y_joy.min():.2f}, {Y_joy.max():.2f}]")
    log(f"Triggers: {Y_trig.shape}, range [{Y_trig.min():.2f}, {Y_trig.max():.2f}]")
    log(f"Buttons:  {Y_btn.shape}, pos_rate: {Y_btn.mean():.3f}")

    # Split
    idx = np.arange(len(X))
    ti, vi = train_test_split(idx, test_size=0.2, random_state=42)

    # 1. Train joystick decoder
    joy_model = JoystickDecoder(inp)
    log(f"Joystick params: {sum(p.numel() for p in joy_model.parameters()):,}")
    joy_model, joy_r2 = train_joystick(joy_model, X[ti], Y_joy[ti], X[vi], Y_joy[vi])
    joy_model.cpu()
    torch.save(joy_model.state_dict(), 'analysis/decoder_v4/joystick.pt')
    log(f"  Saved joystick.pt (R²={joy_r2:.4f})")
    gc.collect(); torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    # 2. Train trigger decoder
    trig_model = TriggerDecoder(inp)
    log(f"Trigger params: {sum(p.numel() for p in trig_model.parameters()):,}")
    trig_model, trig_r2 = train_triggers(trig_model, X[ti], Y_trig[ti], X[vi], Y_trig[vi])
    trig_model.cpu()
    torch.save(trig_model.state_dict(), 'analysis/decoder_v4/trigger.pt')
    log(f"  Saved trigger.pt (R²={trig_r2:.4f})")
    gc.collect(); torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    # 3. Train button gate
    gate_model = ButtonGate(inp)
    log(f"Gate params: {sum(p.numel() for p in gate_model.parameters()):,}")
    gate_model = train_gate(gate_model, X[ti], Y_btn[ti], Y_joy[ti], X[vi], Y_btn[vi], Y_joy[vi])
    gate_model.cpu()
    torch.save(gate_model.state_dict(), 'analysis/decoder_v4/gate.pt')
    log(f"  Saved gate.pt")
    gc.collect(); torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    # Final evaluation
    joy_model.eval().to(DEVICE); trig_model.eval().to(DEVICE); gate_model.eval().to(DEVICE)
    Xv = torch.FloatTensor(X[vi]).to(DEVICE)
    with torch.no_grad():
        jp = joy_model(Xv).cpu().numpy()
        tp = trig_model(Xv).cpu().numpy()
        btn_logits, gate = gate_model(Xv)
        bp = (torch.sigmoid(btn_logits) > 0.5).cpu().numpy().astype(float)
        gp = gate.cpu().numpy()

    # Apply gate: suppress joystick when unused buttons detected
    jp_gated = jp * gp

    Yjv = Y_joy[vi]; Ytv = Y_trig[vi]; Ybv = Y_btn[vi]
    gate_true = (Ybv.sum(1) == 0).astype(float)

    log(f"\n{'='*60}")
    log(f"  FINAL RESULTS")
    log(f"{'='*60}")

    log("\nJoystick R² (raw / gated):")
    jnames = ['L-X (arm X)', 'L-Y (arm Y)', 'R-X (rotation)', 'R-Y (depth)']
    for i, n in enumerate(jnames):
        r2_raw = r2_score(Yjv[:, i], jp[:, i])
        r2_gated = r2_score(Yjv[:, i], jp_gated[:, i])
        log(f"  {n}: raw={r2_raw:.4f}  gated={r2_gated:.4f}")
    log(f"  Avg raw:   {r2_score(Yjv, jp, multioutput='uniform_average'):.4f}")
    log(f"  Avg gated: {r2_score(Yjv, jp_gated, multioutput='uniform_average'):.4f}")

    log(f"\nTrigger R² (analog):")
    for i, n in enumerate(['LT (open)', 'RT (close)']):
        log(f"  {n}: {r2_score(Ytv[:, i], tp[:, i]):.4f}")
    log(f"  Avg: {r2_score(Ytv, tp, multioutput='uniform_average'):.4f}")

    log(f"\nButton Gate:")
    bnames = ['A', 'B', 'X', 'Y', 'LB', 'RB']
    for i, n in enumerate(bnames):
        log(f"  {n}: {accuracy_score(Ybv[:, i], bp[:, i]):.4f} ({int(Ybv[:, i].sum())})")
    log(f"  Avg: {accuracy_score(Ybv.flatten(), bp.flatten()):.4f}")
    log(f"  Gate accuracy: {accuracy_score(gate_true, (gp > 0.5).flatten()):.4f}")

    # Norm params already saved; model checkpoints saved after each component above

    # ONNX — combined model for deployment
    class RobotArmDecoder(nn.Module):
        """Output: [4 joystick, 2 triggers, 6 button probs, 1 gate]"""
        def __init__(self, joy, trig, gate):
            super().__init__()
            self.joy = joy; self.trig = trig; self.gate_net = gate

        def forward(self, x):
            j = self.joy(x)                        # 4 joystick axes
            t = self.trig(x)                       # 2 trigger values
            btn_logits, g = self.gate_net(x)       # 6 buttons + gate
            b = torch.sigmoid(btn_logits)
            # Apply gate to joystick
            j_gated = j * g
            return torch.cat([j_gated, t, b, g], dim=1)  # 4+2+6+1 = 13

    combined = RobotArmDecoder(joy_model, trig_model, gate_model)
    combined.eval()
    dummy = torch.randn(1, w, nf)
    torch.onnx.export(combined, dummy, 'models/decoder.onnx',
        input_names=['neural_features'],
        output_names=['arm_control'],
        dynamic_axes={'neural_features': {0: 'batch'}, 'arm_control': {0: 'batch'}},
        opset_version=13, dynamo=False)
    log("\nONNX saved: models/decoder.onnx")
    log("Output format: [joy_x, joy_y, rot, depth, lt_open, rt_close, A, B, X, Y, LB, RB, gate]")

    # Plots
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    t = np.arange(len(Yjv)) / 100
    for i, (ax, n) in enumerate(zip(axes, jnames)):
        ax.plot(t, Yjv[:, i], 'b-', alpha=.5, lw=.5, label='True')
        ax.plot(t, jp[:, i], 'r-', alpha=.5, lw=.5, label='Pred (raw)')
        ax.plot(t, jp_gated[:, i], 'g-', alpha=.3, lw=.5, label='Pred (gated)')
        r2 = r2_score(Yjv[:, i], jp[:, i])
        ax.set_title(f'{n} — R² = {r2:.4f}'); ax.legend(loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('v4 Joystick Decoder (Robot Arm)', fontsize=14)
    plt.tight_layout(); plt.savefig('analysis/decoder_v4/joystick.png', dpi=150); plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
    for i, (ax, n) in enumerate(zip(axes, ['LT (Open Clamp)', 'RT (Close Clamp)'])):
        ax.plot(t, Ytv[:, i], 'b-', alpha=.5, lw=.5, label='True')
        ax.plot(t, tp[:, i], 'r-', alpha=.5, lw=.5, label='Predicted')
        r2 = r2_score(Ytv[:, i], tp[:, i])
        ax.set_title(f'{n} — R² = {r2:.4f}'); ax.legend()
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('v4 Trigger Decoder (Clamp Control)', fontsize=14)
    plt.tight_layout(); plt.savefig('analysis/decoder_v4/triggers.png', dpi=150); plt.close()

    log("Plots saved to analysis/decoder_v4/")
