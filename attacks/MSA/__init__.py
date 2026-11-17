"""Metadata Semantic Recovery Attack."""
from ..models import SemanticAttack


class MSA(SemanticAttack):
    channel = "meta"


__all__ = ["MSA"]
