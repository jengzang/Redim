"""Small, dependency-light helpers for the ReDimNet2 reproduction project."""

from .data_validation import ManifestError, validate_voxceleb_layout

__all__ = ["ManifestError", "validate_voxceleb_layout"]
