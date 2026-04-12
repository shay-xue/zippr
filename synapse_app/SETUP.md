# Synapse App Setup

## Prerequisites
- Python 3.10+
- Docker (for cross-compilation to ARM64)
- `pip install --pre science-synapse`

## Development Workflow

### 1. Fork the example app
Fork the `calvin/inference` branch of the synapse-example-app repo:
https://github.com/sciencecorp/synapse-example-app

```bash
# Clone your fork
git clone https://github.com/<your-fork>/synapse-example-app.git
cd synapse-example-app
git checkout calvin/inference
```

### 2. Modify the decoder
Edit `src/fixed_weight_decoder.cpp` to load your ONNX model.

Key changes:
- Set `inference_enabled: true` in manifest.json
- Place your `decoder.onnx` in the models directory
- Adjust input/output dimensions to match your model

### 3. Build
```bash
synapsectl build
```

### 4. Deploy to SciFi
```bash
synapsectl deploy -u <device-ip>
```

### 5. Start with config
```bash
synapsectl start -u <device-ip> manifest.json
```

### 6. Verify output
```bash
# From the client directory
python client/listen_to_joystick.py --device <device-ip>
```

## Model Requirements
- Format: ONNX (`.onnx`) or quantized DLC (`.dlc`)
- Input: float tensor of shape (1, n_channels, window_size)
- Output: float tensor of shape (1, n_classes)
- Models deploy to `/opt/scifi/data/models/` on device
- System tries `.dlc` first (DSP), falls back to `.onnx` (CPU)

## Configuration (manifest.json)
```json
{
  "nodes": {
    "1": {
      "type": "kBroadbandSource",
      "peripheral_id": 100,
      "sample_rate_hz": 32000,
      "channels": [0, 1, ..., 31]
    },
    "2": {
      "type": "kApplication",
      "executable": "synapse-example-app",
      "inference_enabled": true,
      "model_name": "decoder"
    }
  }
}
```
