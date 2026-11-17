"""Shared temporal encoding. All timestamps use milliseconds."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class EncodedTrace:
    values: np.ndarray
    mask: np.ndarray
    feature_names: tuple[str, ...]
    channel: str
    start_ms: float
    end_ms: float
    truncated: bool = False

    def flatten(self) -> np.ndarray:
        """Append the observation mask so padding differs from idle observations."""
        return np.concatenate((self.values.ravel(), self.mask.astype(np.float32)))


def time_ms(row: dict) -> float:
    value = float(row["time_ms"] if "time_ms" in row else row["timestamp_ms"])
    if not math.isfinite(value):
        raise ValueError("timestamps must be finite")
    return value


def allocate(start: float, end: float, window_ms: float, max_windows: int,
             names: tuple[str, ...], channel: str) -> EncodedTrace:
    if not all(math.isfinite(x) for x in (start, end, window_ms)):
        raise ValueError("window bounds must be finite")
    if end <= start or window_ms <= 0 or max_windows < 1:
        raise ValueError("positive duration, window size, and window count required")
    n = min(max_windows, max(1, math.ceil((end - start) / window_ms)))
    mask = np.zeros(max_windows, dtype=bool)
    mask[:n] = True
    return EncodedTrace(np.zeros((max_windows, len(names)), dtype=np.float32),
                        mask, names, channel, start, end,
                        end - start > window_ms * max_windows)


def window_index(t: float, start: float, window_ms: float) -> int:
    """Right-closed windows (a,b], with the onset assigned to the first window."""
    return max(0, math.ceil((t - start) / window_ms) - 1)
