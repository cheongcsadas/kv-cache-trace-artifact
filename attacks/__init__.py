"""Channel-separated KvTrail feature extraction and inference."""

from .encoding import EncodedTrace
from .metadata import encode_metadata, reconstruct_metadata
from .hardware import Calibration, LocalizationError, encode_hardware, localize_hardware, hardware_activity_windows
from .ablation import encode_ablation, ABLATIONS

__all__ = ["EncodedTrace", "Calibration", "LocalizationError", "encode_metadata",
           "reconstruct_metadata", "encode_hardware", "localize_hardware", "hardware_activity_windows", "encode_ablation", "ABLATIONS"]
