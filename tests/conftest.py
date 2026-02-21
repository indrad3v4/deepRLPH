# -*- coding: utf-8 -*-
"""Pytest configuration for deepRLPH tests.

This file normalizes sys.path so that the test environment matches how
`main.py` runs the app: treating `src/` as the top-level module root.

That allows imports like `from orchestrator import ...` and flat imports
inside orchestrator.py (e.g. `from context_ingestor import ...`) to work
under pytest on local machines.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Make src/ behave like the app's top-level module directory.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
