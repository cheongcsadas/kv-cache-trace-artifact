"""Side-channel Semantic Recovery Attack."""
from ..models import SemanticAttack


class SSA(SemanticAttack):
    channel = "hw"


__all__ = ["SSA"]
