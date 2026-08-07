"""Stable entry points for WAL direct-packed inference."""

from .platform import RuntimePlatform, detect_platform

__all__ = ["RuntimePlatform", "detect_platform"]
