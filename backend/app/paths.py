"""
Centralized path resolution for AutoML Framework.

Handles both normal (development) and frozen (PyInstaller) execution modes.

Concepts:
  - BUNDLE_DIR : where the bundled read-only assets live
      * Normal  → backend/              (project root)
      * Frozen  → sys._MEIPASS          (temp extraction folder)
  - RUNTIME_DIR: writable directory for user data (uploads, models, DB files)
      * Normal  → backend/              (same as BUNDLE_DIR)
      * Frozen  → directory where the .exe resides
"""

import sys
import os
from pathlib import Path

# ── Detect frozen mode ───────────────────────────────────────────────
IS_FROZEN = getattr(sys, "frozen", False)

# ── Base directories ─────────────────────────────────────────────────
if IS_FROZEN:
    # PyInstaller stores extracted bundle here (read-only)
    BUNDLE_DIR = Path(sys._MEIPASS)
    # Writable directory = folder containing the .exe
    RUNTIME_DIR = Path(os.path.dirname(sys.executable))
else:
    # Development mode: both point to backend/
    BUNDLE_DIR = Path(__file__).resolve().parent.parent   # backend/
    RUNTIME_DIR = BUNDLE_DIR


def get_bundle_path(*parts: str) -> Path:
    """
    Resolve a path relative to the bundle (read-only assets).
    Use for: included static files, seed data shipped with the app.
    """
    return BUNDLE_DIR.joinpath(*parts)


def get_runtime_path(*parts: str) -> Path:
    """
    Resolve a path relative to the runtime directory (writable).
    Use for: uploads, trained_models, data (JSON stores), logs.
    """
    return RUNTIME_DIR.joinpath(*parts)
