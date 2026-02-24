"""
Backward-compatibility shim for metamachine.utils.evolution.

The evolutionary algorithm engine has moved to the dedicated package::

    metamachine.evolution

This module re-exports everything from there so existing code that imports
from ``metamachine.utils.evolution`` continues to work without modification.

Prefer importing directly from ``metamachine.evolution``:

    from metamachine.evolution import EvolutionEngine, Individual

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

import warnings as _warnings

_warnings.warn(
    "metamachine.utils.evolution is deprecated. "
    "Import from metamachine.evolution instead.",
    DeprecationWarning,
    stacklevel=2,
)

from metamachine.evolution.engine import (  # noqa: F401  (re-export)
    EvolutionEngine,
    Individual,
    tournament_selection,
    truncation_selection,
    roulette_selection,
)

__all__ = [
    "EvolutionEngine",
    "Individual",
    "tournament_selection",
    "truncation_selection",
    "roulette_selection",
]
