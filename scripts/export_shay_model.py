#!/usr/bin/env python3
"""Export Shay's trained GRU checkpoint to ONNX + extract normalization stats.

Loads best_model.pt, recomputes feat_norm from the training data,
exports decoder.onnx, and prints C++ arrays for the synapse app.
"""

import h5py, numpy as np, glob, os, json, gc
import torch, torch.nn as nn
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 32_000
N_NEURAL = 64
N_LABELS = 12
BIN_MS = 10
BIN_SAMPLES = int(SAMPLE_RATE * BIN_MS / 1000)  # 320
N_FEATURES = 192
SEQ_LEN = 30
HIDDEN_SIZE = 192
N_LAYERS = 2
DROPOUT = 0.3

CHECKPOINT = 'analysis/decoder_rnn_shay/best_model.pt'
OUT_DIR = 'analysis/decoder_rnn_shay'
ONNX_PATH = 'models/decoder.onnx'
DATA_DIR = 'data/recordings'


def bandpass_filter(neural):
    sos = butter(2, [200, 5000], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    return sosfilt(sos, neural, axis=0).astype(np.float32)


def extract_features_track_a(neural_filtered):
    ch_std = neural_filtered.std(0) + 1e-8
    n_bins = len(neural_filtered) // BIN_SAMPLES
    neural_tr = neural_filtered[:n_bins * BIN_SAMPLES]

    parts = []
    for sigma in [3.0, 4.0, 5.0]:
        thresh = sigma * ch_std
        above = neural_tr > thresh
        below = neural_tr < -thresh
        spikes = (above | below).astype(np.float32)
        counts = spikes.reshape(n_bins, BIN_SAMPLES, N_NEURAL).sum(1)
        parts.append(counts)
        del above, below, spikes

    return np.concatenate(parts, axis=1).astype(np.float32), ch_std


class GRUDecoder(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = (nn.Linear(input_size, hidden_size)
                           if input_size != hidden_size else nn.Identity())
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.joy_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.GELU(), nn.Linear(64, 4), nn.Tanh())
        self.trig_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.GELU(), nn.Linear(32, 2), nn.Sigmoid())
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.input_proj(x)
        _, h_n = self.gru(x)
        z = h_n[-1]
        return self.joy_head(z), self.trig_head(z), self.gate_head(z)


class GRUDecoderForExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 192, 30) → (B, 30, 192)
        joy, trig, gate = self.model(x)
        return torch.cat([joy, trig, gate], dim=1)


def main():
    print("=" * 60)
    print("  Export Shay's GRU Model → ONNX + Normalization Stats")
    print("=" * 60)

    # 1. Load checkpoint
    print("\n[1/4] Loading checkpoint...")
    model = GRUDecoder()
    state_dict = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {CHECKPOINT} ({n_params:,} params)")

    # 2. Extract features from training data to get normalization stats
    print("\n[2/4] Computing normalization stats from training data...")
    # Find all actual HDF5 files (some are nested in directories named *.h5)
    files = sorted(glob.glob(os.path.join(DATA_DIR, '**/*.h5'), recursive=True))
    # Filter to actual files, not directories
    files = [f for f in files if os.path.isfile(f)]
    print(f"  Found {len(files)} recording files")

    all_feats = []
    all_ch_stds = []
    for i, f in enumerate(files):
        with h5py.File(f, 'r') as hf:
            raw = hf['acquisition/ElectricalSeries'][:]
        n = len(raw) // 76
        if n < BIN_SAMPLES * 2:
            print(f"  Skipping {os.path.basename(f)} (too short: {n} samples)")
            continue
        try:
            d = raw[:n * 76].reshape(n, 76).astype(np.float32)
        except Exception as e:
            print(f"  Skipping {os.path.basename(f)} (reshape error: {e})")
            continue
        neural = d[:, :N_NEURAL]
        del raw, d

        neural_filt = bandpass_filter(neural)
        del neural

        features, ch_std = extract_features_track_a(neural_filt)
        del neural_filt

        all_feats.append(features)
        all_ch_stds.append(ch_std)
        gc.collect()

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"  Processed {i+1}/{len(files)} files")

    all_feats_cat = np.concatenate(all_feats)
    feat_mean = all_feats_cat.mean(0)
    feat_std = all_feats_cat.std(0) + 1e-8
    channel_stds = np.mean(np.stack(all_ch_stds), axis=0)  # average across files
    print(f"  Total bins: {len(all_feats_cat):,}")
    print(f"  feat_mean range: [{feat_mean.min():.3f}, {feat_mean.max():.3f}]")
    print(f"  feat_std range:  [{feat_std.min():.3f}, {feat_std.max():.3f}]")
    print(f"  channel_stds range: [{channel_stds.min():.1f}, {channel_stds.max():.1f}]")

    # Save normalization stats
    np.savez(os.path.join(OUT_DIR, 'feat_norm.npz'),
             feat_mean=feat_mean, feat_std=feat_std, channel_stds=channel_stds)
    print(f"  Saved feat_norm.npz")

    # 3. Export ONNX
    print("\n[3/4] Exporting ONNX...")
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
    export_model = GRUDecoderForExport(model.cpu().eval())
    dummy = torch.randn(1, N_FEATURES, SEQ_LEN)

    torch.onnx.export(
        export_model, dummy, ONNX_PATH,
        input_names=['neural_features'],
        output_names=['arm_control'],
        dynamic_axes={
            'neural_features': {0: 'batch'},
            'arm_control': {0: 'batch'},
        },
        opset_version=13, dynamo=False,
    )
    onnx_size = os.path.getsize(ONNX_PATH)
    print(f"  Saved ONNX: {ONNX_PATH} ({onnx_size / 1e6:.1f} MB)")
    print(f"  Input:  (1, {N_FEATURES}, {SEQ_LEN})")
    print(f"  Output: (1, 7) = [LStX, LStY, RStX, RStY, LT, RT, gate]")

    # Verify
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH)
        diffs = []
        for _ in range(10):
            x_np = np.random.randn(1, N_FEATURES, SEQ_LEN).astype(np.float32)
            with torch.no_grad():
                pt_out = export_model(torch.from_numpy(x_np)).numpy()
            ort_out = sess.run(None, {'neural_features': x_np})[0]
            diffs.append(np.abs(pt_out - ort_out).max())
        print(f"  ONNX verification: max diff = {max(diffs):.2e} ✓")
    except Exception as e:
        print(f"  ONNX verification skipped: {e}")

    # 4. Print C++ arrays for synapse app
    print("\n[4/4] C++ constants for synapse app:")
    print("=" * 60)

    print("\n// Channel standard deviations (64 channels)")
    print("static constexpr float kChannelStds[64] = {")
    for i in range(0, 64, 8):
        vals = ', '.join(f'{channel_stds[j]:.6f}f' for j in range(i, min(i+8, 64)))
        print(f"  {vals},")
    print("};")

    print("\n// Feature means (192 features)")
    print("static constexpr float kFeatMean[192] = {")
    for i in range(0, 192, 8):
        vals = ', '.join(f'{feat_mean[j]:.6f}f' for j in range(i, min(i+8, 192)))
        print(f"  {vals},")
    print("};")

    print("\n// Feature stds (192 features)")
    print("static constexpr float kFeatStd[192] = {")
    for i in range(0, 192, 8):
        vals = ', '.join(f'{feat_std[j]:.6f}f' for j in range(i, min(i+8, 192)))
        print(f"  {vals},")
    print("};")

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print(f"  1. Copy C++ arrays above into fixed_weight_decoder.cpp")
    print(f"  2. synapsectl apps build synapse_app/synapse-example-app")
    print(f"  3. synapsectl -u <device> deploy-model {ONNX_PATH} --name decoder --force")
    print(f"  4. synapsectl -u <device> apps deploy synapse_app/synapse-example-app")
    print(f"  5. synapsectl -u <device> start synapse_app/synapse-example-app/config/gru_decoder_64ch.json")


if __name__ == '__main__':
    main()
