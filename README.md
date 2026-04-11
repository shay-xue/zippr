# Science NeuroTech — Global NeuroHack 2026

BCI decoder and game built on the Science Corporation SciFi headstage for the Global NeuroHack hackathon.

## Team
- Manraj
- Shay
- Yoyo
- Medha

## Project Overview

Maximizing bit rate through a neural decoder pipeline:
1. **Decoder**: ONNX model trained to decode controller inputs from mock neural data
2. **Synapse App**: On-device inference running on the SciFi headstage
3. **Game**: Interactive game scored by achieved bit rate over 60 seconds
4. **Hardware Integration**: Robotic arm for physical game interaction

## Architecture

```
Controller → SciFi (mock neural encoder) → Synapse App (ONNX decoder) → Game/Robotic Arm
```

## Getting Started

### Prerequisites
- Python 3.10+
- Docker (for Synapse App builds)
- `pip install --pre science-synapse`

### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

With `uv`, you can install the root project environment directly:

```bash
uv sync
```

### SciFi Device Setup
```bash
# Discover device
synapsectl discover

# Check device info
synapsectl -u <device-ip> info

# Connect controller: hold B button while plugging USB
# Verify encoder peripherals appear
synapsectl -u <device-ip> info
```

### Data Collection
```bash
python scripts/collect_data.py --device <device-ip> --mode easy --duration 300
```

### Training
```bash
python scripts/train_decoder.py --data data/recordings/ --output models/decoder.onnx
```

## Project Structure
```
├── data/               # Training data and recordings
├── models/             # Trained ONNX models
├── scripts/            # Data collection, training, utilities
├── synapse_app/        # On-device Synapse App (C++)
├── game/               # Game application
└── arm/                # Robotic arm control
```

## Arm Control

The `arm/` package contains the SO-101 follower arm control path used for hardware validation.

### Local arm config

Create `arm/local_config.py` for machine-specific settings such as the serial port:

```python
from arm.config import ArmSettings

SETTINGS = ArmSettings(
    port="/dev/tty.usbmodem0000001",
    robot_id="so101-local",
)
```

`arm/local_config.py` is gitignored so local hardware details stay out of the repo.

### Keyboard teleop

Run the phase-1 joint-space teleop loop with:

```bash
uv run arm-keyboard-teleop
```

### Browser teleop server

If you want a local server you can drive from a browser keyboard page, run:

```bash
uv run arm-web-teleop
```

Then open `http://127.0.0.1:8765` in a browser, click the page to focus it, and use the same keys.
The server binds only to `127.0.0.1`.
The page shows live arm connection status and any startup error, and teleop startup skips interactive calibration unless you set `calibrate_on_connect=True`.

### FastAPI command server

If you want to send abstract end-effector delta commands over HTTP, run:

```bash
uv run arm-api-server --dry-run
```

For real hardware, make sure `arm/local_config.py` contains the serial `port`, and place a local
copy of `so101.urdf` at `models/so101.urdf` or pass it explicitly:

```bash
uv run arm-api-server --urdf-path models/so101.urdf
```

Then open `http://127.0.0.1:8000/` in a browser, click the page to focus it, and use:
- `w/s`: positive/negative forward reach `dx`
- `a/d`: negative/positive shoulder pan `dy` in radians
- `i/k`: positive/negative `dz`
- `j/l`: negative/positive `d_rot`
- `t/g`: open/close gripper via `d_jaw`

The UI sends repeated `POST /api/end-effector-delta` requests while keys are held and lets you tune
the reach/z, shoulder-pan, roll, jaw, and repeat-step sizes from the page.

Example commands:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/joints
curl http://127.0.0.1:8000/api/pose
curl -X POST http://127.0.0.1:8000/api/end-effector-delta \
  -H 'Content-Type: application/json' \
  -d '{"dx": -0.01, "dy": 0.1, "dz": 0.01, "d_rot": 5.0, "d_jaw": 0.0}'
```

### Scan connected motor IDs

To check which motor IDs respond on the configured serial port:

```bash
uv run arm-check-motor-ids
```

Or scan a specific port directly:

```bash
uv run arm-check-motor-ids --port /dev/tty.usbmodem0000001
```

Default bindings:
- `q/a`: shoulder pan
- `w/s`: shoulder lift
- `e/d`: elbow flex
- `r/f`: wrist flex
- `t/g`: wrist roll
- `y/h`: gripper open/close
- `space`: freeze current target
- `esc`: exit cleanly

## Resources
- [SciFi Docs](https://science.xyz/docs/d/scifi1/index)
- [Synapse Docs](https://science.xyz/docs/c/synapse)
- [Synapse Apps Docs](https://science.xyz/docs/d/synapse-app/index)
- [Synapse Example App](https://github.com/sciencecorp/synapse-example-app)
