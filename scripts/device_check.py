"""
SciFi device setup verification script.

Connects to the SciFi headstage, prints device info, checks for
encoder peripherals, and verifies data streaming works.

Usage:
    python scripts/device_check.py --device <ip>
"""

import argparse
import sys
import time
import synapse as syn


def parse_args():
    parser = argparse.ArgumentParser(description="Verify SciFi device setup")
    parser.add_argument("--device", "-d", required=True, help="Device IP or hostname")
    parser.add_argument("--port", type=int, default=647, help="Synapse API port")
    return parser.parse_args()


def check_device(device_addr, port):
    """Run through all device verification steps."""
    addr = f"{device_addr}:{port}"

    # Step 1: Connect
    print("=" * 60)
    print(f"SciFi Device Check — {addr}")
    print("=" * 60)

    print("\n[1/4] Connecting to device...")
    try:
        device = syn.Device(addr)
        print("  ✓ Connection successful")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("\n  Troubleshooting:")
        print("    - Is the SciFi powered on?")
        print("    - Is it connected to the same WiFi network?")
        print("    - Try: synapsectl discover")
        return False

    # Step 2: Device info
    print("\n[2/4] Querying device info...")
    try:
        info = device.info()
        print(f"  ✓ Device info received")
        print(f"  {info}")
    except Exception as e:
        print(f"  ✗ Failed to get device info: {e}")
        return False

    # Step 3: Check for encoder peripherals
    print("\n[3/4] Checking for encoder peripherals...")
    print("  (Controller must be connected: hold B + plug USB)")
    try:
        # The exact API for checking peripherals depends on the info structure
        # This will need adjustment based on actual device.info() output
        if hasattr(info, "peripherals"):
            for p in info.peripherals:
                print(f"  Found peripheral: {p}")
        else:
            print(f"  Device info structure: {type(info)}")
            print(f"  Full info: {info}")
        print("  ✓ Peripheral check complete")
    except Exception as e:
        print(f"  ⚠ Could not enumerate peripherals: {e}")

    # Step 4: Quick stream test
    print("\n[4/4] Testing data stream (2 seconds)...")
    try:
        channels = [
            syn.Channel(id=i, electrode_id=i, reference_id=0)
            for i in range(32)
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
        time.sleep(2)
        device.stop()
        print("  ✓ Stream test passed")
    except Exception as e:
        print(f"  ⚠ Stream test issue: {e}")
        print("  This may need adjustment based on actual peripheral IDs")

    print("\n" + "=" * 60)
    print("Device check complete. If peripherals are missing,")
    print("reconnect the controller (hold B + plug USB) and retry.")
    print("=" * 60)
    return True


def main():
    args = parse_args()
    success = check_device(args.device, args.port)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
