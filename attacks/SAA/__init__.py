"""Side-channel Attribute Inference Attack."""
from ..models import AttributeAttack


class SAA(AttributeAttack):
    channel = "hw"


__all__ = ["SAA"]
