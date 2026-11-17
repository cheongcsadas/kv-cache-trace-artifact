"""Section 6.2: an input-independent Uniform(0, 5 ms) delay per load operation."""
import random
import math
import time


class DynamicTiming:
    def __init__(self, max_delay_ms=5.0, seed=None):
        self.max_delay_ms = float(max_delay_ms)
        if not math.isfinite(self.max_delay_ms) or self.max_delay_ms < 0:
            raise ValueError("Maximum delay must be finite and nonnegative")
        self.rng = random.Random(seed)

    def execute(self, load, *args, sleeper=time.sleep, **kwargs):
        """Call on the backend's loading path before queuing the eligible copy.

        Host sleep can overshoot; actual latency must be measured externally.
        Seed=None avoids a shared deterministic delay sequence across requests.
        """
        sleeper(self.rng.uniform(0.0, self.max_delay_ms)/1000)
        return load(*args, **kwargs)
