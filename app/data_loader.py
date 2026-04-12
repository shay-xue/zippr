"""
H5 neural recording loader with real-time streaming support.

Each .h5 file contains a flat interleaved array that must be reshaped:
    raw.reshape(-1, 76)  →  (timepoints, 76 channels)
    channels 0–63   = neural signal
    channels 64–75  = raw controller labels (12 channels)
"""

from __future__ import annotations

import glob
import logging
import os
import threading
import time
from collections import deque
from typing import Deque, Generator, Optional

import h5py
import numpy as np

from config import (
    BUFFER_SECONDS,
    N_NEURAL_CHANNELS,
    N_TARGET_CHANNELS,
    N_TOTAL_CHANNELS,
    RECORDINGS_DIR,
    SAMPLE_RATE_HZ,
)

logger = logging.getLogger(__name__)


class H5DataLoader:
    """Load and stream neural data from .h5 recordings.

    Parameters
    ----------
    recordings_dir:
        Directory containing ``broadband_data*.h5`` files.
    buffer_seconds:
        Seconds of data to keep in the rolling display buffer.
    """

    def __init__(
        self,
        recordings_dir: str = RECORDINGS_DIR,
        buffer_seconds: int = BUFFER_SECONDS,
    ) -> None:
        self.recordings_dir = recordings_dir
        self.buffer_seconds = buffer_seconds

        # Loaded data
        self.neural: Optional[np.ndarray] = None    # (T, 64) float32
        self.targets: Optional[np.ndarray] = None   # (T, 12) float32
        self.timestamps: Optional[np.ndarray] = None # (T,) float64 seconds

        # Rolling buffer for waveform display (thread-safe via lock)
        self._buffer: Deque[np.ndarray] = deque()
        self._buffer_ts: Deque[float] = deque()
        self._lock = threading.Lock()

        # Z-score stats (computed once at load)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        # Background streaming
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stream_idx: int = 0

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load and concatenate all .h5 files in recordings_dir."""
        pattern = os.path.join(self.recordings_dir, "broadband_data*.h5")
        paths = sorted(glob.glob(pattern))
        if not paths:
            logger.warning("No .h5 files found in %s", self.recordings_dir)
            return

        all_neural, all_targets, all_ts = [], [], []

        for path in paths:
            logger.info("Loading %s", path)
            with h5py.File(path, "r") as f:
                raw = f["acquisition"]["ElectricalSeries"][:]
                data = raw.reshape(-1, N_TOTAL_CHANNELS).astype(np.float32)
                ts_ns = f["acquisition"]["timestamp_ns"][:]
                ts_sec = ts_ns.astype(np.float64) / 1e9

            neural = data[:, :N_NEURAL_CHANNELS]
            targets = data[:, N_NEURAL_CHANNELS : N_NEURAL_CHANNELS + N_TARGET_CHANNELS]

            all_neural.append(neural)
            all_targets.append(targets)
            all_ts.append(ts_sec)

        self.neural = np.concatenate(all_neural, axis=0)
        self.targets = np.concatenate(all_targets, axis=0)
        self.timestamps = np.concatenate(all_ts, axis=0)

        # Z-score normalisation (per channel)
        self._mean = self.neural.mean(axis=0)
        self._std = self.neural.std(axis=0)
        self._std[self._std == 0] = 1.0  # avoid div-by-zero
        self.neural = (self.neural - self._mean) / self._std

        logger.info(
            "Loaded %d timepoints (%.1f s) from %d files",
            len(self.timestamps),
            self.timestamps[-1] - self.timestamps[0],
            len(paths),
        )

    # ── Streaming generator ──────────────────────────────────────────────────

    def stream_chunk(
        self, chunk_size_ms: float = 100
    ) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Yield (neural_chunk, target_chunk, ts_chunk) at real-time pace.

        Each chunk has shape (channels, samples_in_chunk) for neural data,
        transposed from the stored (samples, channels) layout.
        """
        if self.neural is None:
            return

        chunk_samples = int(chunk_size_ms / 1000.0 * SAMPLE_RATE_HZ)
        n = len(self.neural)
        idx = 0

        while idx < n:
            end = min(idx + chunk_samples, n)
            neural_chunk = self.neural[idx:end].T       # (64, chunk)
            target_chunk = self.targets[idx:end].T      # (12, chunk)
            ts_chunk = self.timestamps[idx:end]

            yield neural_chunk, target_chunk, ts_chunk

            idx = end
            time.sleep(chunk_size_ms / 1000.0)

    # ── Background playback ──────────────────────────────────────────────────

    def start_playback(self, chunk_size_ms: float = 100) -> None:
        """Start streaming data in a background thread, filling the rolling buffer."""
        self._stop.clear()
        self._stream_idx = 0
        self._thread = threading.Thread(
            target=self._playback_loop, args=(chunk_size_ms,), daemon=True
        )
        self._thread.start()

    def stop_playback(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _playback_loop(self, chunk_size_ms: float) -> None:
        for neural_chunk, _, ts_chunk in self.stream_chunk(chunk_size_ms):
            if self._stop.is_set():
                break
            with self._lock:
                self._buffer.append(neural_chunk)
                self._buffer_ts.append(ts_chunk[-1])
                # Trim buffer to keep only last buffer_seconds
                while (
                    len(self._buffer_ts) > 1
                    and self._buffer_ts[-1] - self._buffer_ts[0]
                    > self.buffer_seconds
                ):
                    self._buffer.popleft()
                    self._buffer_ts.popleft()

    # ── Buffer access (for waveform display) ─────────────────────────────────

    def get_buffer(self) -> Optional[np.ndarray]:
        """Return the current rolling buffer as (channels, samples) or None."""
        with self._lock:
            if not self._buffer:
                return None
            return np.concatenate(list(self._buffer), axis=1)

    def get_buffer_time_range(self) -> tuple[float, float]:
        """Return (start_time, end_time) of the current buffer in seconds."""
        with self._lock:
            if not self._buffer_ts:
                return (0.0, 0.0)
            return (self._buffer_ts[0], self._buffer_ts[-1])
