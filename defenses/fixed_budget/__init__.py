"""Section 6.1: real padding transfers, shuffled with required loads."""
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class LoadPlan:
    transfer_order: tuple
    required: tuple
    padding: tuple
    within_budget: bool


class FixedBudget:
    def __init__(self, budget=65, block_tokens=16, seed=None):
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1 or block_tokens != 16:
            raise ValueError("Paper defense uses positive budgets at 16-token granularity")
        self.budget = budget
        self.rng = random.Random(seed)

    def plan(self, required, padding_candidates):
        required, candidates = tuple(required), tuple(padding_candidates)
        if len(set(required)) != len(required): raise ValueError("Duplicate required blocks")
        candidates = tuple(dict.fromkeys(x for x in candidates if x not in set(required)))
        count = max(0, self.budget-len(required))
        if len(candidates) < count:
            raise ValueError("Insufficient valid nonresident CPU-backed padding pages")
        padding = tuple(self.rng.sample(candidates, count))
        order = list(required+padding)
        self.rng.shuffle(order)
        return LoadPlan(tuple(order), required, padding, len(required) <= self.budget)

    def execute(self, plan, transfer, publish):
        """transfer(id) must finish an actual H2D copy; publish(id) uses identical schema.

        transfer must place padding in non-attention scratch storage, respecting
        backend residency bookkeeping. No padding flag is exposed to the consumer.
        Returned attention inputs contain required pages only.
        """
        if not isinstance(plan, LoadPlan):
            raise ValueError("Expected a LoadPlan created by this defense")
        required, padding, order = plan.required, plan.padding, plan.transfer_order
        if (len(set(required)) != len(required) or len(set(padding)) != len(padding)
                or set(required) & set(padding) or len(set(order)) != len(order)
                or set(order) != set(required) | set(padding)
                or len(padding) != max(0, self.budget - len(required))
                or plan.within_budget != (len(required) <= self.budget)):
            raise ValueError("Invalid plan: transfer counts, identities or budget disagree")
        transferred = {}
        for block in plan.transfer_order:
            transferred[block] = transfer(block)
            publish(block)
        return [transferred[block] for block in plan.required]


def eligible(system, phase):
    if system not in {"quest", "infinigen", "retrievalattention"}:
        raise ValueError("Unknown system")
    if phase not in {"prefill", "decode"}:
        raise ValueError("Unknown inference phase")
    return phase == ("prefill" if system == "retrievalattention" else "decode")
