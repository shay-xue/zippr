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


# ── Mock data loader (synthetic waveforms for demo mode) ────────────────────

class MockDataLoader:
    """Generates realistic-looking synthetic neural waveforms in real time.

    Produces 64 channels of band-limited noise with occasional spike-like
    transients, matching the visual character of real broadband neural data.
    Compatible with the same interface as H5DataLoader.
    """

    def __init__(
        self,
        n_channels: int = N_NEURAL_CHANNELS,
        sample_rate: int = SAMPLE_RATE_HZ,
        buffer_seconds: int = BUFFER_SECONDS,
    ) -> None:
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds

        self._buffer: Deque[np.ndarray] = deque()
        self._buffer_ts: Deque[float] = deque()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._rng = np.random.default_rng(42)

        # Per-channel characteristics (amplitude, spike rate)
        self._amplitudes = self._rng.uniform(0.3, 1.5, size=n_channels).astype(np.float32)
        self._spike_rates = self._rng.uniform(0.5, 5.0, size=n_channels)  # spikes/sec

    def load(self) -> None:
        """No-op — synthetic data needs no file loading."""
        pass

    def start_playback(self, chunk_size_ms: float = 100) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._generate_loop, args=(chunk_size_ms,), daemon=True
        )
        self._thread.start()

    def stop_playback(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _generate_loop(self, chunk_size_ms: float) -> None:
        chunk_samples = int(chunk_size_ms / 1000.0 * self.sample_rate)
        t0 = time.time()

        while not self._stop.is_set():
            now = time.time()
            # Base: band-limited Gaussian noise per channel
            noise = self._rng.standard_normal(
                (self.n_channels, chunk_samples)
            ).astype(np.float32)

            # Scale by per-channel amplitude
            chunk = noise * self._amplitudes[:, None]

            # Add sparse spike-like transients
            for ch in range(self.n_channels):
                n_spikes = self._rng.poisson(
                    self._spike_rates[ch] * chunk_size_ms / 1000.0
                )
                if n_spikes > 0:
                    spike_locs = self._rng.integers(0, chunk_samples, size=n_spikes)
                    spike_amps = self._rng.uniform(2.5, 5.0, size=n_spikes)
                    signs = self._rng.choice([-1.0, 1.0], size=n_spikes)
                    chunk[ch, spike_locs] += (spike_amps * signs).astype(np.float32)

            with self._lock:
                self._buffer.append(chunk)
                self._buffer_ts.append(now)
                # Trim to buffer_seconds
                while (
                    len(self._buffer_ts) > 1
                    and self._buffer_ts[-1] - self._buffer_ts[0]
                    > self.buffer_seconds
                ):
                    self._buffer.popleft()
                    self._buffer_ts.popleft()

            time.sleep(chunk_size_ms / 1000.0)

    def get_buffer(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._buffer:
                return None
            return np.concatenate(list(self._buffer), axis=1)

    def get_buffer_time_range(self) -> tuple[float, float]:
        with self._lock:
            if not self._buffer_ts:
                return (0.0, 0.0)
            return (self._buffer_ts[0], self._buffer_ts[-1])
