"""Block-aligned physical resident cache sizes, applied before allocation."""
import math


def budget(context_tokens, block_tokens, residency_fraction):
    if (isinstance(context_tokens, bool) or not isinstance(context_tokens, int)
            or context_tokens < 2 or block_tokens not in (8, 16, 32)
            or not 0 < residency_fraction < 1):
        raise ValueError("Invalid context length, logical page size, or residency fraction")
    blocks = math.ceil(context_tokens / block_tokens)
    if blocks < 2:
        raise ValueError("A CPU/GPU split requires at least two context pages")
    slots = min(blocks - 1, max(1, math.ceil(blocks * residency_fraction)))
    # Preserve the upstream sink/recent arrangement; scale its 128:512 ratio.
    sink = min(slots - 1, max(0, slots // 5))
    recent = slots - sink
    persistent = list(range(sink)) + list(range(blocks - recent, blocks))
    return dict(context_blocks=blocks, capacity_slots_per_layer=slots,
                sink_slots=sink, recent_slots=recent, persistent_blocks=persistent,
                persistent_tokens=sum(min(block_tokens, context_tokens-b*block_tokens)
                                      for b in persistent))
