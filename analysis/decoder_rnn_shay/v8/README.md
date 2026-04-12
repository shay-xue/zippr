# GRU Decoder v8

Retrain of v7 with two improvements: optimal bin size from sweep applied, and Run 007 data added to fill coverage gaps.

---

## What changed from v7

| | v7 | v8 |
|--|----|----|
| Bin size | 10ms | **100ms** |
| Context window | 30 bins × 10ms = 300ms | **15 bins × 100ms = 1500ms** |
| Training data | Run 006 only | **Run 006 + Run 007** |
| LStY coverage | Underrepresented (val std 2.25× gap) | Fixed by Run 007 |
| RT coverage | Nearly absent in val | Fixed by Run 007 |

### Why 100ms bins?

A bin size sweep was run across [20, 30, 40, 50, 60, 70, 80, 100] ms using the same v7 architecture and preprocessing pipeline (`scripts/bin_size_sweep_v7.py`). Each bin size was evaluated with SEQ_LEN=15 bins, 50 epochs, early stopping patience=15.

Results showed 100ms as the clear winner (mean joystick R²=+0.652). The v7 model was trained with 10ms bins and never had this applied — v8 is the first model to use the optimal bin size.

Likely reason: at 10ms, each bin captures only ~3–6 spikes per channel on average, which is very noisy. At 100ms, each bin captures ~30–60 spikes, giving a much more stable signal estimate.

### Why Run 007?

After training v7 on Run 006, label distribution analysis showed two critical gaps in the validation set:

| Label | Train std | Val std | Ratio |
|-------|-----------|---------|-------|
| LStX | 0.430 | 0.435 | 1.01× |
| LStY | 0.430 | 0.152 | **2.25×** |
| RT | 0.326 | 0.111 | **2.94×** |

Run 007 added 26 dedicated sessions targeting these gaps:
- 8 sessions of isolated left stick vertical movement (up, down, sweep, circles)
- 4 sessions of RT/LT isolated sweeps and pumps
- 3 combined sessions (both sticks + RT/LT)
- 1 freestyle session (natural play)
- 2 rest baselines

---

## Training data

| Source | Description | Files |
|--------|-------------|-------|
| Run 006 | 32 structured sessions, diverse inputs | `data/recordings/train_*/` |
| Run 007 | 26 targeted sessions, fills LStY/RT gaps | `data/recordings/run007_*/` |

---

## Preprocessing pipeline

### 1. Load HDF5
- Path: `acquisition/ElectricalSeries` — flattened int16 array
- Reshape to `(n_samples, 76)`: channels 0–63 = neural, 64–75 = labels
- Labels: `[LStX, LStY, RStX, RStY, A, B, X, Y, LB, RB, LT, RT]`

### 2. Bandpass filter
- 2nd-order Butterworth bandpass: **200–5000 Hz**
- Isolates spike-band activity, removes LFP and high-frequency noise
- Applied per file along the time axis using `scipy.signal.sosfilt`

### 3. Bin into 100ms windows
- Bin size: **100ms = 3,200 samples** at 32,000 Hz
- Each 2-minute session produces ~1,200 bins

### 4. Spike counting — Track A (192 features/bin)
- Compute per-channel std from the full filtered recording
- Count threshold crossings at **3σ, 4σ, 5σ** — both positive and negative
- Each threshold gives 64 counts/bin (one per neural channel)
- Concatenated: **192 features/bin** (64 channels × 3 thresholds)

### 5. Label binning
- Labels averaged over each 100ms window → `(n_bins, 12)`

### 6. Label normalisation
| Channels | Labels | Normalisation |
|----------|--------|---------------|
| 64–67 | LStX, LStY, RStX, RStY | ÷ 32767 → [-1, +1] |
| 74–75 | LT, RT | ÷ 32767 → [0, +1] |
| 68–73 | A, B, X, Y, LB, RB | threshold at 16000 → binary 0/1 |
| — | gate | 1 if no buttons pressed, else 0 |

### 7. Feature normalisation
- Global mean and std computed from **training bins only**
- Applied to both train and val: `(x - mean) / std`
- Stats saved to `feat_norm.npz` for use at inference time

---

## Architecture

```
Input: (batch, 15, 192)              — 1500ms of spike-count features
  → Linear(192 → 192)               — input projection
  → GRU(hidden=192, layers=2, dropout=0.3)
  → last hidden state (batch, 192)
  → joy_head:  Linear(192→64) → GELU → Linear(64→4)  → Tanh    → (batch, 4)  joystick axes
  → trig_head: Linear(192→32) → GELU → Linear(32→2)  → Sigmoid → (batch, 2)  LT/RT
  → gate_head: Linear(192→32) → GELU → Linear(32→1)  → Sigmoid → (batch, 1)  button gate
```

Total parameters: ~330k

---

## Sequence construction

- **Sequence length**: 15 bins = 1500ms context
- **Training stride**: 1 (every possible overlapping window)
- **Val stride**: 15 (non-overlapping windows, no data leakage)
- Sequences never cross file boundaries

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts (T₀=30, T_mult=2) |
| Batch size | 128 |
| Max epochs | 300 |
| Early stopping patience | 40 epochs |
| Input noise augmentation | Gaussian σ=0.05 |
| Gradient clipping | max norm 1.0 |

**Loss**: `MSE(joystick) + MSE(triggers) + 0.5 × BCE(gate)`  
**Best model checkpoint**: selected by highest val joystick R²

---

## Train/val split

File-based stratified split — one file per recording category held out for val. Val file = middle file of each category (avoids boundary effects).

| Category | Matched by |
|----------|-----------|
| rest | `rest_start`, `rest_end`, `_rest_` |
| lstick | `lsty`, `lstx`, `lst_circle`, `lstick` |
| rstick | `rsty`, `rstx`, `rst_circle`, `rstick` |
| triggers | `lt_sweep`, `lt_pump`, `rt_sweep`, `rt_pump`, `lt_analog`, `rt_analog` |
| bothsticks | `both_sticks`, `sticks_and`, `bothsticks` |
| freestyle | `natural_play`, `freestyle`, `slow_all` |
| buttons | `btn_`, `buttons`, `ab_alt`, `xy_alt` |
| bumpers | `bumpers`, `lb_spam`, `rb_spam` |

---

## ONNX export

Wrapped in `GRUDecoderForExport` for Synapse App compatibility:
- **Input**: `(1, 192, 15)` — channels-first (transposed internally for GRU)
- **Output**: `(1, 7)` — `[LStX, LStY, RStX, RStY, LT, RT, gate]`

Saved to: `models/decoder_v8.onnx`  
Normalisation stats: `analysis/decoder_rnn_shay/v8/feat_norm.npz`  
Full metadata: `analysis/decoder_rnn_shay/v8/model_meta.json`
