# """Timeliner package scaffold.
#
# Usage:
#     >>> from timeliner.pipeline import build_timeline
#     >>> from timeliner.config import load_config
#     >>> cfg = load_config("sample_config.yml")
#     >>> timeline = build_timeline(cfg)
# """
#
# from .config import load_config, TimelinerConfig
# from .pipeline import build_timeline
#
# __all__ = [
#     "load_config",
#     "TimelinerConfig",
#     "build_timeline",
# ]

"""Top‑level package for Timeliner."""
from importlib.metadata import version as _version  # Python ≥ 3.10

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