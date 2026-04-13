#!/usr/bin/env python3
"""brain_to_arm.py — Bridge between SciFi neural decoder and SO-101 robot arm.

Subscribes to the `joystick_out` tap from the Synapse device, maps the 7-element
decoder output to EndEffectorDeltaCommands, and POSTs them to the SO-101 FastAPI
server at a configurable rate.

Decoder outputs (from GRU model):
  [0] joy_x  → dx  (arm reach, forward/back)
  [1] joy_y  → dy  (shoulder pan, left/right)
  [2] rot    → d_rot (wrist roll)
  [3] depth  → dz  (arm height, up/down)
  [4] lt     → d_jaw (open gripper)
  [5] rt     → d_jaw (close gripper)
  [6] gate   → motion enable/disable (< 0.5 = suppress)

Usage:
  python3 scripts/brain_to_arm.py --device-ip 192.168.16.227 --arm-url http://127.0.0.1:8000
"""

import argparse
import struct
import sys
import time

import numpy as np
import requests
from synapse.api.datatype_pb2 import Tensor
from synapse.client.taps import Tap

# ── Mapping constants ──────────────────────────────────────────────────────
# These scale raw decoder outputs [-1,1] or [0,1] into physical deltas.
# Tune these based on how responsive you want the arm to be.

DX_SCALE = 0.10       # meters per unit — forward/back reach
DY_SCALE = 0.40       # radians per unit — shoulder pan
DZ_SCALE = 0.10       # meters per unit — up/down height
DROT_SCALE = 25.0     # degrees per unit — wrist roll
DJAW_SCALE = 10.0     # jaw units per unit — gripper open/close

GATE_THRESHOLD = 0.5  # suppress motion when gate < this
DEADZONE = 0.2       # ignore joystick values below this magnitude

COMMAND_RATE_HZ = 10  # max commands per second to the arm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge SciFi neural decoder to SO-101 robot arm"
    )
    parser.add_argument(
        "--device-ip", required=True,
        help="IP address of the Synapse device (e.g. 192.168.16.227)"
    )
    parser.add_argument(
        "--tap-name", default="joystick_out",
        help="Synapse tap name (default: joystick_out)"
    )
    parser.add_argument(
        "--arm-url", default="http://127.0.0.1:8000",
        help="SO-101 FastAPI server URL (default: http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without sending to the arm"
    )
    parser.add_argument(
        "--dx-scale", type=float, default=DX_SCALE,
        help=f"Reach scale in meters (default: {DX_SCALE})"
    )
    parser.add_argument(
        "--dy-scale", type=float, default=DY_SCALE,
        help=f"Pan scale in radians (default: {DY_SCALE})"
    )
    parser.add_argument(
        "--dz-scale", type=float, default=DZ_SCALE,
        help=f"Height scale in meters (default: {DZ_SCALE})"
    )
    parser.add_argument(
        "--drot-scale", type=float, default=DROT_SCALE,
        help=f"Wrist roll scale in degrees (default: {DROT_SCALE})"
    )
    parser.add_argument(
        "--djaw-scale", type=float, default=DJAW_SCALE,
        help=f"Jaw scale (default: {DJAW_SCALE})"
    )
    return parser.parse_args()


def parse_tensor(raw: bytes) -> np.ndarray:
    """Parse a Synapse Tensor protobuf into a float32 numpy array."""
    tensor = Tensor()
    tensor.ParseFromString(raw)
    data = tensor.data
    if tensor.endianness == Tensor.Endianness.TENSOR_BIG_ENDIAN:
        values = struct.unpack(f">{len(data) // 4}f", data)
        return np.array(values, dtype=np.float32)
    return np.frombuffer(data, dtype=np.float32)


def apply_deadzone(value: float, deadzone: float) -> float:
    """Zero out values within the deadzone, rescale the rest."""
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def decoder_to_arm_command(
    outputs: np.ndarray,
    dx_scale: float,
    dy_scale: float,
    dz_scale: float,
    drot_scale: float,
    djaw_scale: float,
) -> dict:
    """Map 7 decoder outputs to an EndEffectorDeltaCommand dict.

    Returns None if gate is below threshold (motion suppressed).
    """
    if outputs.size < 7:
        return None

    joy_x = float(outputs[0])   # left stick X
    joy_y = float(outputs[1])   # left stick Y
    rot   = float(outputs[2])   # right stick X → wrist roll
    depth = float(outputs[3])   # right stick Y → reach (extend/curl)
    lt    = float(outputs[4])   # left trigger → open gripper
    rt    = float(outputs[5])   # right trigger → close gripper
    gate  = float(outputs[6])   # motion gate

    # Gate check — if gate is low, buttons/triggers are being pressed,
    # suppress all joystick motion
    if gate < GATE_THRESHOLD:
        return {"dx": 0.0, "dy": 0.0, "dz": 0.0, "d_rot": 0.0, "d_jaw": 0.0}

    # Apply deadzone to joystick axes
    joy_x = apply_deadzone(joy_x, DEADZONE)
    joy_y = apply_deadzone(joy_y, DEADZONE)
    rot   = apply_deadzone(rot, DEADZONE)
    depth = apply_deadzone(depth, DEADZONE)

    # Gripper: lt opens (positive d_jaw), rt closes (negative d_jaw)
    d_jaw = (lt - rt) * djaw_scale

    # Mapping:
    #   Left stick X  (joy_x) → dy  (shoulder pan: left=turn left, right=turn right)
    #   Left stick Y  (joy_y) → dz  (height: up=elbow up, down=elbow down)
    #   Right stick X (rot)   → d_rot (wrist roll: left=wrist left, right=wrist right)
    #   Right stick Y (depth) → dx  (reach: up=extend out, down=curl back)
    return {
        "dx": depth * dx_scale,
        "dy": joy_x * dy_scale,
        "dz": joy_y * dz_scale,
        "d_rot": rot * drot_scale,
        "d_jaw": d_jaw,
    }


def main() -> None:
    args = parse_args()

    # Verify arm server is reachable
    if not args.dry_run:
        try:
            resp = requests.get(f"{args.arm_url}/health", timeout=3)
            resp.raise_for_status()
            print(f"Arm server OK at {args.arm_url}")
        except Exception as e:
            print(f"WARNING: Arm server not reachable at {args.arm_url}: {e}",
                  file=sys.stderr)
            print("Continuing anyway — will retry on each command.", file=sys.stderr)

    # Connect to Synapse tap
    tap = Tap(args.device_ip)
    if not tap.connect(args.tap_name):
        print(f"Failed to connect to tap '{args.tap_name}' at {args.device_ip}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Connected to tap '{args.tap_name}' at {args.device_ip}")
    print(f"Arm target: {args.arm_url}")
    print(f"Scales: dx={args.dx_scale}m dy={args.dy_scale}rad dz={args.dz_scale}m "
          f"rot={args.drot_scale}° jaw={args.djaw_scale}")
    print(f"Gate threshold: {GATE_THRESHOLD}, Deadzone: {DEADZONE}")
    print("Press Ctrl-C to stop.\n")

    min_interval = 1.0 / COMMAND_RATE_HZ
    last_send = 0.0
    cmd_count = 0
    err_count = 0
    session = requests.Session()

    try:
        while True:
            raw = tap.read()
            if raw is None:
                time.sleep(0.001)
                continue

            # Rate limit
            now = time.monotonic()
            if now - last_send < min_interval:
                continue

            # Parse decoder output
            try:
                outputs = parse_tensor(raw)
            except Exception as e:
                print(f"Parse error: {e}", file=sys.stderr)
                continue

            # Map to arm command
            command = decoder_to_arm_command(
                outputs,
                args.dx_scale, args.dy_scale, args.dz_scale,
                args.drot_scale, args.djaw_scale,
            )
            if command is None:
                continue

            cmd_count += 1

            if args.dry_run:
                print(f"[{cmd_count:5d}] dx={command['dx']:+.4f} dy={command['dy']:+.4f} "
                      f"dz={command['dz']:+.4f} rot={command['d_rot']:+.3f} "
                      f"jaw={command['d_jaw']:+.3f}  "
                      f"raw=[{' '.join(f'{v:+.2f}' for v in outputs[:7])}]")
            else:
                try:
                    resp = session.post(
                        f"{args.arm_url}/api/end-effector-delta",
                        json=command,
                        timeout=1,
                    )
                    resp.raise_for_status()

                    if cmd_count % 50 == 0:
                        data = resp.json()
                        print(f"[{cmd_count:5d}] dx={command['dx']:+.4f} "
                              f"dy={command['dy']:+.4f} dz={command['dz']:+.4f} "
                              f"→ OK (errors: {err_count})")
                except requests.RequestException as e:
                    err_count += 1
                    if err_count % 10 == 1:
                        print(f"Arm send error #{err_count}: {e}", file=sys.stderr)

            last_send = now

    except KeyboardInterrupt:
        print(f"\nStopped. Sent {cmd_count} commands, {err_count} errors.")
    finally:
        tap.disconnect()


if __name__ == "__main__":
    main()
