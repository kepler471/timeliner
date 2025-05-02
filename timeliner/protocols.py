"""Prompt snippets (protocols) that steer the summariser."""
from __future__ import annotations

summary_config_protocols: list[str] = [
    "Only include information relevant to the selected theme.",
    "Emphasise *new* facts or developments not previously summarised.",
    "If a headline repeats earlier information with no update, note it as continuation only.",
]