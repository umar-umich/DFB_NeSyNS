"""CEC certification gate: per (image, claim) -> CERTIFIED/REJECTED/UNTESTABLE."""
from .certify import CertificationGate  # noqa: F401
__all__ = ["CertificationGate"]
