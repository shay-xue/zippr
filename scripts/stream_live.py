"""
Live streaming script — connects to SciFi and prints decoded output in real-time.

Used to test the on-device decoder by reading from the Synapse App's output tap.

Usage:
    python scripts/stream_live.py --device <ip> --tap joystick_out
"""

import argparse
import struct
import time
import numpy as np

try:
    import synapse as syn
except ImportError:
    print("Install synapse: pip install --pre science-synapse")
    raise


def parse_args():
    parser = argparse.ArgumentParser(description="Stream live decoded output from SciFi")
    parser.add_argument("--device", "-d", required=True, help="Device IP or hostname")
    parser.add_argument("--port", type=int, default=647, help="Synapse API port")
    parser.add_argument("--tap", default="joystick_out",
                        help="Name of the tap to stream from")
    parser.add_argument("--raw", action="store_true",
                        help="Stream raw broadband instead of decoded output")
    return parser.parse_args()


def stream_decoded(device_addr, port, tap_name):
    """Stream decoded output from the Synapse App's tap."""
    addr = f"{device_addr}:{port}"
    print(f"Connecting to {addr}, tap: {tap_name}")

    device = syn.Device(addr)
    info = device.info()
    print(f"Device: {info}")

    print(f"\nStreaming from tap '{tap_name}'... (Ctrl+C to stop)\n")

    try:
        while True:
            # Read from tap — exact API depends on synapse implementation
            # This is based on the listen_to_joystick.py pattern from the example app
            data = device.read_tap(tap_name)
            if data is not None:
                # Expecting (x, y) or class probabilities depending on decoder output
                print(f"\rDecoded: {data}", end="", flush=True)
            time.sleep(0.05)  # ~20 Hz polling

    except KeyboardInterrupt:
        print("\n\nStreaming stopped.")


def stream_raw_broadband(device_addr, port, n_channels=34):
    """Stream raw broadband data (neural + labels in training mode)."""
    addr = f"{device_addr}:{port}"
    print(f"Connecting to {addr} for raw broadband ({n_channels} ch)")

    device = syn.Device(addr)

    channels = [
        syn.Channel(id=i, electrode_id=i, reference_id=0)
        for i in range(n_channels)
    ]

    broadband = syn.BroadbandSource(
        peripheral_id=0,
        sample_rate_hz=32000,
        bit_width=12,
        signal=syn.SignalConfig(
            electrode=syn.ElectrodeConfig(channels=channels)
        ),
    )

    stream_out = syn.StreamOut(multicast_group="")

    config = syn.Config()
    config.add_node(broadband)
    config.add_node(stream_out)
    config.connect(broadband, stream_out)

    device.configure(config)
    device.start()

    print("Streaming raw broadband... (Ctrl+C to stop)\n")

    try:
        sample_count = 0
        start = time.time()
        while True:
            data = stream_out.read()
            if data is not None:
                if isinstance(data, np.ndarray):
                    sample_count += data.shape[-1] if data.ndim > 1 else 1
                elapsed = time.time() - start
                rate = sample_count / elapsed if elapsed > 0 else 0
                print(f"\rSamples: {sample_count:,} | Rate: {rate:.0f} sps | "
                      f"Elapsed: {elapsed:.1f}s", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        device.stop()
        print("Done.")


def main():
    args = parse_args()
    if args.raw:
        stream_raw_broadband(args.device, args.port)
    else:
        stream_decoded(args.device, args.port, args.tap)


if __name__ == "__main__":
    main()
