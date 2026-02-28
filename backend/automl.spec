# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AutoML Framework
==========================================

Build command:
    cd backend
    pyinstaller automl.spec --clean

Or simply run:
    build.bat
"""
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# ── Paths ────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.abspath(".")
FRONTEND_DIST = os.path.join(BACKEND_DIR, "..", "frontend", "dist")

# ── Hidden imports ───────────────────────────────────────────────────
# PyInstaller misses many scikit-learn, imblearn, and optional modules
# because they are imported lazily inside functions.
hiddenimports = []

# Collect ALL submodules for packages with dynamic / lazy imports
for pkg in [
    "sklearn",
    "sklearn.cluster",
    "sklearn.compose",
    "sklearn.decomposition",
    "sklearn.ensemble",
    "sklearn.feature_extraction",
    "sklearn.feature_extraction.text",
    "sklearn.feature_selection",
    "sklearn.impute",
    "sklearn.linear_model",
    "sklearn.metrics",
    "sklearn.model_selection",
    "sklearn.neighbors",
    "sklearn.pipeline",
    "sklearn.preprocessing",
    "sklearn.svm",
    "sklearn.tree",
    "sklearn.utils",
    "imblearn",
    "xgboost",
    "lightgbm",
    "optuna",
    "scipy",
    "scipy.sparse",
    "scipy.stats",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "starlette",
    "pydantic",
    "pydantic_settings",
]:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        hiddenimports.append(pkg)

# Explicit additions that collect_submodules sometimes misses
hiddenimports += [
    "multiprocessing",
    "multiprocessing.process",
    "multiprocessing.queues",
    "multiprocessing.pool",
    "chardet",
    "psutil",
    "joblib",
    "aiofiles",
    "reportlab",
    "reportlab.lib",
    "reportlab.lib.pagesizes",
    "reportlab.platypus",
    "encodings",
    "encodings.utf_8",
    "encodings.ascii",
    "encodings.latin_1",
    "encodings.cp1252",
    "email.mime",              # sometimes needed by urllib
    "pkg_resources.py2_warn",  # suppress setuptools warning
]

# Optional packages — ignore if not installed
for optional_pkg in ["hdbscan", "umap", "statsmodels", "statsmodels.stats",
                      "statsmodels.stats.outliers_influence"]:
    try:
        hiddenimports += collect_submodules(optional_pkg)
    except Exception:
        pass

# De-duplicate
hiddenimports = sorted(set(hiddenimports))

# ── Data files to bundle ─────────────────────────────────────────────
datas = []

# Seed JSON data files (read-only copies; app copies them to writable dir on first run)
datas.append((os.path.join(BACKEND_DIR, "data"), "data"))

# Frontend build — only if it exists (run `npm run build` first)
if os.path.isdir(FRONTEND_DIST):
    datas.append((FRONTEND_DIST, "frontend_dist"))
else:
    print("⚠  WARNING: frontend/dist not found. Build frontend first: cd frontend && npm run build")

# Collect data files for packages that ship .json / .txt metadata
for pkg in ["xgboost", "lightgbm", "sklearn", "certifi", "pydantic"]:
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

# ── Analysis ─────────────────────────────────────────────────────────
a = Analysis(
    ["main.py"],
    pathex=[BACKEND_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",       # not needed, saves ~20 MB
        "test",
        "tests",
        "unittest",
        "IPython",
        "notebook",
        "jupyter",
        "pytest",
    ],
    noarchive=False,
    optimize=0,
)

# ── Build targets ────────────────────────────────────────────────────
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoML",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,        # Keep console visible for logs; set False for GUI-only
    icon=None,           # Set to 'icon.ico' if you add one
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AutoML",
)
