"""
Data collection script for SciFi headstage.

Connects to the SciFi device, streams broadband neural data in training mode
(which includes ground truth controller labels), and saves to HDF5.

Usage:
    python scripts/collect_data.py --device <ip> --mode easy --duration 300
    python scripts/collect_data.py --device <ip> --mode hard --duration 300
"""

import argparse
import time
import numpy as np
import h5py
import synapse as syn


def parse_args():
    parser = argparse.ArgumentParser(description="Collect training data from SciFi")
    parser.add_argument("--device", "-d", required=True, help="SciFi device IP or hostname")
    parser.add_argument("--port", type=int, default=647, help="Synapse API port")
    parser.add_argument("--mode", choices=["easy", "hard"], default="easy",
                        help="Encoder mode (easy=32ch+2labels, hard=64ch+12labels)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Recording duration in seconds")
    parser.add_argument("--output", "-o", default=None,
                        help="Output HDF5 file path")
    return parser.parse_args()


def get_channel_config(mode: str):
    """Return channel counts based on encoder mode."""
    if mode == "easy":
        return {
            "neural_channels": 32,
            "label_channels": 2,  # joystick X, Y (+ possibly A button)
            "total_channels": 34,
        }
    else:
        return {
            "neural_channels": 64,
            "label_channels": 12,  # 2 joysticks (4) + ABXY (4) + 2 triggers + 2 bumpers
            "total_channels": 76,
        }


def connect_and_stream(device_addr: str, port: int, mode: str, duration: int, output_path: str):
    """Connect to SciFi, configure broadband streaming, and record data."""

    ch_config = get_channel_config(mode)
    n_total = ch_config["total_channels"]
    n_neural = ch_config["neural_channels"]
    n_labels = ch_config["label_channels"]
    sample_rate = 32000  # 32 kSps as per challenge spec

    print(f"Connecting to SciFi at {device_addr}:{port}...")
    device = syn.Device(f"{device_addr}:{port}")

    info = device.info()
    print(f"Device info: {info}")
    print(f"Mode: {mode} | Neural channels: {n_neural} | Label channels: {n_labels}")

    # Configure broadband source for all channels (neural + labels in training mode)
    channels = [
        syn.Channel(id=i, electrode_id=i, reference_id=0)
        for i in range(n_total)
    ]

    broadband = syn.BroadbandSource(
        peripheral_id=0,  # Will need to adjust based on device.info() output
        sample_rate_hz=sample_rate,
        bit_width=12,
        signal=syn.SignalConfig(
            electrode=syn.ElectrodeConfig(channels=channels)
        ),
    )

    stream_out = syn.StreamOut(
        multicast_group="",  # unicast
    )

    config = syn.Config()
    config.add_node(broadband)
    config.add_node(stream_out)
    config.connect(broadband, stream_out)

    print("Configuring device...")
    device.configure(config)

    print(f"Starting {duration}s recording...")
    device.start()

    # Pre-allocate buffer for recording
    total_samples = duration * sample_rate
    data_buffer = np.zeros((n_total, total_samples), dtype=np.float32)

    samples_collected = 0
    start_time = time.time()

    try:
        while samples_collected < total_samples:
            elapsed = time.time() - start_time
            if elapsed > duration + 5:  # timeout safety
                print(f"Timeout after {elapsed:.1f}s")
                break

            # Read data from stream
            # The exact API for reading depends on the stream configuration
            # This will need to be adapted based on actual Synapse API behavior
            data = stream_out.read()

            if data is not None and len(data) > 0:
                n_new = data.shape[-1] if data.ndim > 1 else 1
                end_idx = min(samples_collected + n_new, total_samples)
                actual_new = end_idx - samples_collected

                if data.ndim == 1:
                    data = data.reshape(n_total, -1)

                data_buffer[:, samples_collected:end_idx] = data[:, :actual_new]
                samples_collected = end_idx

                if samples_collected % (sample_rate * 10) < n_new:
                    pct = 100 * samples_collected / total_samples
                    print(f"  Progress: {pct:.1f}% ({samples_collected}/{total_samples} samples)")

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")

    finally:
        device.stop()
        elapsed = time.time() - start_time
        print(f"Recording complete: {samples_collected} samples in {elapsed:.1f}s")

    # Trim to actual collected samples
    data_buffer = data_buffer[:, :samples_collected]

    # Save to HDF5
    print(f"Saving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.attrs["mode"] = mode
        f.attrs["sample_rate"] = sample_rate
        f.attrs["n_neural_channels"] = n_neural
        f.attrs["n_label_channels"] = n_labels
        f.attrs["duration_seconds"] = elapsed
        f.attrs["total_samples"] = samples_collected

        # Store neural data and labels separately for convenience
        f.create_dataset("neural", data=data_buffer[:n_neural, :], compression="gzip")
        f.create_dataset("labels", data=data_buffer[n_neural:, :], compression="gzip")
        f.create_dataset("raw", data=data_buffer, compression="gzip")

    print(f"Saved: neural={n_neural}ch x {samples_collected} samples, "
          f"labels={n_labels}ch x {samples_collected} samples")
    return output_path


def main():
    args = parse_args()

    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"data/recordings/{args.mode}_{timestamp}.h5"

    connect_and_stream(args.device, args.port, args.mode, args.duration, args.output)


if __name__ == "__main__":
    main()
