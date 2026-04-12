"""
Websocket client for the BCI decoder and a mock server for testing.

The decoder sends raw 12-channel one-hot controller vectors (JSON lists)
over ``ws://localhost:8765``.  Each message is a JSON object::

    {
        "timestamp": 1234567890.123,
        "vector": [0, 0, 0, 32767, 0, 0, 0, 0, 0, 0, 0, 0]
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

try:
    import websockets
    import websockets.server
except ImportError:
    websockets = None  # type: ignore[assignment]

from config import (
    CH_LEFT_STICK_X,
    CH_LEFT_STICK_Y,
    CH_RIGHT_STICK_X,
    CH_RIGHT_STICK_Y,
    CH_TRIGGER_LEFT,
    CH_TRIGGER_RIGHT,
    DECODER_WS_URL,
    NUM_ONEHOT_CHANNELS,
    RAW_STICK_MAX,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)

# ── Target direction hint (set by main app so mock biases toward correct) ─────

_TARGET_DIRECTION: Optional[str] = None
_TARGET_LOCK: threading.Lock = threading.Lock()


def set_target_direction_hint(direction: Optional[str]) -> None:
    """Called by the main app to hint the mock decoder toward the correct direction.

    When set, the MockDecoderServer will emit the hinted direction ~65% of the
    time, simulating a realistic BCI accuracy level so the bit rate is non-zero.
    """
    global _TARGET_DIRECTION
    with _TARGET_LOCK:
        _TARGET_DIRECTION = direction


# ── Decoded frame ─────────────────────────────────────────────────────────────

@dataclass
class DecodedFrame:
    """One decoded output from the BCI decoder."""

    timestamp: float
    vector: List[float]                  # raw 12-channel one-hot
    direction: Optional[str] = None      # UP / DOWN / LEFT / RIGHT / None
    confidence: float = 1.0

    @property
    def is_movement(self) -> bool:
        return self.direction is not None


def vector_to_direction(vec: list[float] | np.ndarray) -> Optional[str]:
    """Map a 12-channel one-hot vector to a grid direction.

    Only the two grid-relevant analog sticks are considered:
      - LEFT_STICK_X  (ch 0): positive → RIGHT, negative → LEFT
      - RIGHT_STICK_Y (ch 3): positive → UP,    negative → DOWN

    Returns None for non-movement inputs (buttons, triggers, all-zero).
    """
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (NUM_ONEHOT_CHANNELS,):
        return None

    active = np.flatnonzero(arr)
    if active.size != 1:
        return None

    idx = int(active[0])
    val = float(arr[idx])

    if idx == CH_LEFT_STICK_X:
        return "RIGHT" if val > 0 else "LEFT"
    if idx == CH_RIGHT_STICK_Y:
        return "UP" if val > 0 else "DOWN"

    # z-axis, rotation, buttons, triggers → not a grid move
    return None


# ── Decoder client ────────────────────────────────────────────────────────────

class DecoderClient:
    """Connects to the decoder websocket and buffers received frames."""

    def __init__(self, url: str = DECODER_WS_URL, maxlen: int = 500) -> None:
        self.url = url
        self._buffer: Deque[DecodedFrame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Start the background receive loop."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_latest(self) -> Optional[DecodedFrame]:
        """Return the most recent frame, or None."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_all_since(self, t: float) -> list[DecodedFrame]:
        """Return all frames with timestamp > *t*."""
        with self._lock:
            return [f for f in self._buffer if f.timestamp > t]

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._receive_loop())

    async def _receive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url) as ws:  # type: ignore[union-attr]
                    logger.info("Connected to decoder at %s", self.url)
                    async for raw_msg in ws:
                        if self._stop.is_set():
                            break
                        self._handle_message(raw_msg)
            except Exception as exc:
                logger.warning("Decoder connection failed (%s), retrying in 2s…", exc)
                await asyncio.sleep(2)

    def _handle_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
            vec = data.get("vector", data)  # accept {vector:[...]} or bare list
            if isinstance(vec, list):
                ts = data.get("timestamp", time.time())
                direction = vector_to_direction(vec)
                frame = DecodedFrame(
                    timestamp=ts,
                    vector=vec,
                    direction=direction,
                    confidence=data.get("confidence", 1.0),
                )
                with self._lock:
                    self._buffer.append(frame)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.debug("Bad decoder message: %s", exc)


# ── Mock decoder server ───────────────────────────────────────────────────────

class MockDecoderServer:
    """Emits one-hot controller vectors for testing.

    When a target direction hint is set via ``set_target_direction_hint()``,
    the server biases ~65% of emitted vectors toward that direction to
    simulate realistic BCI accuracy and produce non-zero bit rates.
    """

    # Maps direction string to (channel_index, sign) for one-hot encoding
    _DIR_TO_CH: dict[str, tuple[int, int]] = {
        "RIGHT": (CH_LEFT_STICK_X,  +1),
        "LEFT":  (CH_LEFT_STICK_X,  -1),
        "UP":    (CH_RIGHT_STICK_Y, +1),
        "DOWN":  (CH_RIGHT_STICK_Y, -1),
    }

    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._rng = random.Random(RANDOM_SEED)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        try:
            async with websockets.serve(  # type: ignore[union-attr]
                self._handler,
                self.host,
                self.port,
            ):
                logger.info("Mock decoder server running on ws://%s:%s", self.host, self.port)
                while not self._stop.is_set():
                    await asyncio.sleep(0.1)
        except OSError as exc:
            # Port still held by a previous session — reuse existing server.
            logger.warning(
                "Mock server bind failed on port %s (%s) — reusing existing server.",
                self.port, exc,
            )

    async def _handler(self, ws: object) -> None:
        """Send a movement vector every 1–2.5 seconds."""
        try:
            while not self._stop.is_set():
                vec = self._random_vector()
                direction = vector_to_direction(vec)
                msg = json.dumps({
                    "timestamp": time.time(),
                    "vector": vec,
                    "direction": direction,
                    "confidence": round(self._rng.uniform(0.65, 1.0), 2),
                })
                await ws.send(msg)  # type: ignore[union-attr]
                await asyncio.sleep(self._rng.uniform(1.0, 2.5))
        except Exception:
            pass  # client disconnected

    def _random_vector(self) -> list[float]:
        """Generate a one-hot vector, biasing toward the current target direction.

        65% of the time: emit the hinted correct direction (if available).
        35% of the time: emit a uniformly random grid-movement direction.
        """
        with _TARGET_LOCK:
            hint = _TARGET_DIRECTION

        if hint is not None and hint in self._DIR_TO_CH and self._rng.random() < 0.65:
            ch, sign = self._DIR_TO_CH[hint]
        else:
            # Random grid-relevant direction
            direction = self._rng.choice(list(self._DIR_TO_CH.keys()))
            ch, sign = self._DIR_TO_CH[direction]

        vec = [0.0] * NUM_ONEHOT_CHANNELS
        vec[ch] = sign * RAW_STICK_MAX
        return vec


# ── Synapse tap decoder client (live SciFi device) ───────────────────────────

def decoder_outputs_to_direction(
    outputs: np.ndarray,
    threshold: float = 0.08,
    gate_threshold: float = 0.5,
) -> Optional[str]:
    """Map 7-channel v9 GRU decoder outputs to a grid direction.

    Decoder outputs:
      [0] joy_x  (left stick X) → LEFT/RIGHT
      [1] joy_y  (left stick Y) → UP/DOWN
      [2] rot, [3] depth, [4] lt, [5] rt — not used for grid
      [6] gate — suppress if < gate_threshold

    Returns the dominant direction (LEFT/RIGHT/UP/DOWN) or None.
    """
    if outputs.size < 7:
        return None

    gate = float(outputs[6])
    if gate < gate_threshold:
        return None

    joy_x = float(outputs[0])  # left stick X → LEFT/RIGHT
    joy_y = float(outputs[1])  # left stick Y → UP/DOWN

    # Apply threshold
    ax = joy_x if abs(joy_x) >= threshold else 0.0
    ay = joy_y if abs(joy_y) >= threshold else 0.0

    if ax == 0.0 and ay == 0.0:
        return None

    # Pick dominant axis
    if abs(ax) >= abs(ay):
        return "RIGHT" if ax > 0 else "LEFT"
    return "UP" if ay > 0 else "DOWN"


class SynapseDecoderClient:
    """Reads from the SciFi device joystick_out tap and buffers decoded frames.

    This replaces the WebSocket-based DecoderClient for live operation.
    Movement frames are rate-limited to at most one per ``min_move_interval``
    seconds so the chess game receives ~1 decision/sec instead of 10/sec.
    """

    def __init__(
        self,
        device_ip: str = "192.168.8.123",
        tap_name: str = "joystick_out",
        direction_threshold: float = 0.15,
        gate_threshold: float = 0.5,
        min_move_interval: float = 1.0,
        maxlen: int = 500,
    ) -> None:
        self.device_ip = device_ip
        self.tap_name = tap_name
        self.direction_threshold = direction_threshold
        self.gate_threshold = gate_threshold
        self.min_move_interval = min_move_interval
        self._buffer: Deque[DecodedFrame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_move_time: float = 0.0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_latest(self) -> Optional[DecodedFrame]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_all_since(self, t: float) -> list[DecodedFrame]:
        with self._lock:
            return [f for f in self._buffer if f.timestamp > t]

    def _run(self) -> None:
        try:
            import struct
            from synapse.api.datatype_pb2 import Tensor
            from synapse.client.taps import Tap
        except ImportError:
            logger.error("synapse SDK not installed — cannot use SynapseDecoderClient")
            return

        tap = Tap(self.device_ip)
        if not tap.connect(self.tap_name):
            logger.error("Failed to connect to tap '%s' at %s", self.tap_name, self.device_ip)
            return

        logger.info("Connected to Synapse tap '%s' at %s", self.tap_name, self.device_ip)

        try:
            while not self._stop.is_set():
                raw = tap.read()
                if raw is None:
                    time.sleep(0.01)
                    continue

                try:
                    tensor = Tensor()
                    tensor.ParseFromString(raw)
                    outputs = np.frombuffer(tensor.data, dtype=np.float32)
                except Exception:
                    continue

                direction = decoder_outputs_to_direction(
                    outputs, self.direction_threshold, self.gate_threshold,
                )

                # Rate-limit movement frames: only emit one per min_move_interval
                now = time.time()
                if direction is not None:
                    if now - self._last_move_time < self.min_move_interval:
                        continue  # skip this movement, too soon
                    self._last_move_time = now

                # Compute confidence from the magnitude of the dominant axis
                if outputs.size >= 2:
                    confidence = min(1.0, max(abs(float(outputs[0])), abs(float(outputs[1]))))
                else:
                    confidence = 0.0

                frame = DecodedFrame(
                    timestamp=time.time(),
                    vector=outputs.tolist(),
                    direction=direction,
                    confidence=confidence,
                )
                with self._lock:
                    self._buffer.append(frame)
        finally:
            tap.disconnect()
