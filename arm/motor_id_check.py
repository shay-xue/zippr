"""Scan a Feetech bus for responding motor IDs on a serial port."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from lerobot.motors.feetech import FeetechMotorsBus

from .config import load_settings


def _model_number_to_name_map() -> dict[int, str]:
    return {value: key for key, value in FeetechMotorsBus.model_number_table.items()}


def scan_motor_ids(port: str) -> dict[int, dict[int, int]]:
    """Return responding motor IDs grouped by baudrate."""

    bus = FeetechMotorsBus(port=port, motors={})
    bus.connect(handshake=False)
    found: dict[int, dict[int, int]] = {}
    try:
        for baudrate in bus.available_baudrates:
            bus.set_baudrate(baudrate)
            ids_to_model_numbers = bus.broadcast_ping()
            if ids_to_model_numbers:
                found[baudrate] = dict(sorted(ids_to_model_numbers.items()))
    finally:
        bus.port_handler.closePort()

    return found


def format_scan_results(port: str, scan_results: dict[int, dict[int, int]]) -> str:
    """Build a readable report for the scan results."""

    lines = [f"Motor scan results for port: {port}"]
    model_name_by_number = _model_number_to_name_map()

    if not scan_results:
        lines.append("No responding motors found on any supported baudrate.")
        return "\n".join(lines)

    for baudrate in sorted(scan_results):
        lines.append(f"")
        lines.append(f"Baudrate {baudrate}:")
        for motor_id, model_number in scan_results[baudrate].items():
            model_name = model_name_by_number.get(model_number, "unknown_model")
            lines.append(f"  ID {motor_id}: model_number={model_number} ({model_name})")

    return "\n".join(lines)


def _resolve_port(explicit_port: str | None) -> str:
    if explicit_port:
        return explicit_port

    settings = load_settings()
    if settings.port:
        return settings.port

    raise ValueError("No port provided. Pass --port or set `port` in arm/local_config.py.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan a serial port for connected Feetech motor IDs.")
    parser.add_argument(
        "--port",
        help="Serial port to scan. Defaults to the port configured in arm/local_config.py.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    port = _resolve_port(args.port)
    results = scan_motor_ids(port)
    print(format_scan_results(port, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
