"""Mistral RoPE/GQA adaptation of InfiniGen's published partial-key rehearsal."""
import math


def rotary(value, positions, theta):
    import torch
    dim = value.shape[-1]
    inv = theta ** (-torch.arange(0, dim, 2, device=value.device, dtype=torch.float32) / dim)
    phase = positions.float()[:, None] * inv[None, :]
    phase = torch.cat((phase, phase), -1)
    first, second = value.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), -1)
    return value * phase.cos().to(value.dtype) + rotated * phase.sin().to(value.dtype)


class Predictor:
    """Use upstream column selection and threshold/top-k selection unchanged.

    RoPE precedes the shared Q/K orthogonal transform, as in the upstream
    Llama path. GQA keys are expanded only in the reduced-dimensional cache.
    The full K/V remains shared across each group of query heads.
    """
    def __init__(self, primitives, query, key, maximum_tokens, ratio, alpha, budget):
        import torch
        self.t = torch
        self.h, self.d = query.shape[0], query.shape[-1]
        self.groups = self.h // key.shape[0]
        self.alpha, self.budget = float(alpha), int(budget)
        self.native = primitives["kv_selection_controller"].speculate_attention
        partial = primitives["partial_weight_generation_controller"]
        if not 0 < ratio <= 1 or int(self.d * ratio) < 1 or self.alpha <= 0 or self.budget < 1:
            raise ValueError("Invalid InfiniGen predictor parameters")
        repeated = key.repeat_interleave(self.groups, dim=0)
        # The right singular vectors are an orthogonal basis even for a short
        # prefill. Sort by joint Q/K singular magnitudes as in native skew().
        basis = []
        for head in range(self.h):
            _, sq, vq = torch.linalg.svd(query[head].float(), full_matrices=False)
            sk = torch.linalg.svdvals(repeated[head].float())
            if vq.shape[0] < self.d:
                # Complete a short-prompt basis without treating absent
                # singular directions as meaningful importance scores.
                _, _, vq = torch.linalg.svd(query[head].float(), full_matrices=True)
            importance = torch.zeros(self.d, device=query.device)
            importance[:sq.numel()] = sq * sk
            order = importance.argsort()
            matrix = torch.empty((self.d, self.d), device=query.device)
            matrix[:, order] = vq.transpose(0, 1)
            basis.append(matrix.to(query.dtype))
        self.basis = torch.stack(basis)
        skewed = torch.matmul(query, self.basis)
        flat = skewed.transpose(0, 1).reshape(1, query.shape[1], self.h*self.d)
        self.columns = partial.partial_weight_index_generation(flat, self.h, self.d, ratio)[0]
        self.width = self.columns.shape[-1]
        self.partial = torch.empty((maximum_tokens, self.h, self.width),
                                   device=query.device, dtype=query.dtype)
        self.length = 0
        self.identity = torch.eye(self.h*self.width, dtype=query.dtype, device=query.device)
        self.append(key)

    def _reduce(self, full):
        value = self.t.matmul(full, self.basis)
        return value.gather(-1, self.columns[:, None, :].expand(-1, value.shape[1], -1))

    def append(self, key):
        count = key.shape[1]
        reduced = self._reduce(key.repeat_interleave(self.groups, dim=0))
        self.partial[self.length:self.length+count].copy_(reduced.transpose(0, 1))
        self.length += count

    def select(self, query, *, earliest=0):
        """Return per-query-head indices in the original chronological cache."""
        if earliest >= self.length:
            return self.t.empty((self.h, 0), dtype=self.t.long, device=query.device)
        reduced = self._reduce(query) / math.sqrt(self.d)
        # Feed already projected/rotated partial Q through an identity linear
        # operator. This preserves the upstream speculation calculation while
        # supporting rotary position encoding and grouped-query projection.
        hidden = reduced.transpose(0, 1).reshape(1, 1, self.h*self.width)
        selected = self.native(hidden, self.identity, self.partial[earliest:self.length],
                               self.h, self.alpha, min(self.budget, self.length-earliest))
        return selected[:, 0, :].transpose(0, 1) + earliest

