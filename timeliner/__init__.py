"""Top‑level package for Timeliner."""
from importlib.metadata import version as _version

__all__ = [
    "__version__",
    "Timeliner",
    "IntervalSummariser",
]

try:
    __version__: str = _version(__name__)
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

# Re‑export primary façades
from .timeliner import Timeliner  # noqa: E402  pylint: disable=wrong-import-position
from .summariser import IntervalSummariser  # noqa: E402