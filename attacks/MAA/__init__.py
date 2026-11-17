"""Metadata Attribute Inference Attack."""
from ..models import AttributeAttack


class MAA(AttributeAttack):
    channel = "meta"


__all__ = ["MAA"]
