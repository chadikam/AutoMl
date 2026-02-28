# AutoML Framework — Packaging as Windows Executable

## Quick Start

```bash
cd backend
build.bat
```

Output: `backend/dist/AutoML/AutoML.exe`

---

## What the Build Does

1. **Builds the frontend** (`cd frontend && npm run build`) → produces `frontend/dist/`
2. **Installs PyInstaller** (if missing)
3. **Runs PyInstaller** with `automl.spec` → produces `backend/dist/AutoML/`

---

## Architecture When Frozen

```
dist/AutoML/
├── AutoML.exe              ← entry point
├── _internal/              ← PyInstaller bundle (read-only)
│   ├── data/               ← seed JSON files (copied to runtime on first run)
│   ├── frontend_dist/      ← built React SPA
│   └── ... (Python + deps)
├── data/                   ← writable JSON DB (created on first run)
├── uploads/                ← user-uploaded CSV files
└── trained_models/         ← saved .joblib models & plots
```

**Key concept:**
- `_internal/` = read-only bundled assets (PyInstaller's `_MEIPASS`)
- Everything else in the exe's directory = writable at runtime

---

## Files Modified for Packaging

| File | Change |
|------|--------|
| `app/paths.py` | **NEW** — Centralized path resolution (`BUNDLE_DIR`, `RUNTIME_DIR`) |
| `app/config.py` | Paths resolved to absolute via `RUNTIME_DIR` after loading `.env` |
| `app/storage.py` | `DATA_DIR` uses `get_runtime_path()` instead of `__file__` |
| `main.py` | Writable dir creation, frontend serving, browser auto-open, frozen uvicorn |
| `automl.spec` | **NEW** — PyInstaller spec with all hidden imports |
| `build.bat` | **NEW** — One-click build script |

---

## Potential Runtime Issues When Frozen

### 1. Missing sklearn / imblearn submodules
**Symptom:** `ModuleNotFoundError: No module named 'sklearn.xxx'`
**Cause:** PyInstaller can't trace lazy imports inside functions.
**Fix:** Already handled in `automl.spec` via `collect_submodules()`. If a new submodule is added later, add it to the `hiddenimports` list.

### 2. Matplotlib backend
**Symptom:** `RuntimeError: Matplotlib requires a GUI framework`
**Fix:** The app uses `matplotlib.pyplot.savefig()` (headless). Add to the top of `main.py` if needed:
```python
import matplotlib
matplotlib.use("Agg")
```

### 3. Multiprocessing freeze support
**Symptom:** Infinite spawn loop or hang on Windows.
**Fix:** Already mitigated by PyInstaller, but if you use `multiprocessing.Process`, add at the very top of `main.py`:
```python
import multiprocessing
multiprocessing.freeze_support()
```

### 4. Anti-virus false positives
**Symptom:** Windows Defender quarantines `AutoML.exe`.
**Fix:** Code-sign the executable, or add a folder exclusion.

### 5. hdbscan / umap / statsmodels not installed
**Symptom:** Features silently disabled (already handled via `try/except`).
**Fix:** Install them before building if you want those features:
```bash
pip install hdbscan umap-learn statsmodels
```

### 6. LightGBM DLL
**Symptom:** `OSError: cannot load library 'lib_lightgbm.dll'`
**Fix:** Already handled by `collect_data_files("lightgbm")` in the spec. If it persists, manually copy the DLL.

### 7. Large executable size (~500MB+)
**Cause:** NumPy, SciPy, sklearn, XGBoost, LightGBM are large.
**Mitigation:** The spec excludes tkinter, test suites, IPython, Jupyter. Further size reduction requires UPX compression (enabled in spec).

### 8. First-run startup time
**Cause:** PyInstaller extracts `_internal/` on first run.
**Mitigation:** Use `--onedir` mode (default in spec) instead of `--onefile` to avoid re-extraction.

### 9. Temp file cleanup
**Symptom:** Orphaned temp files in `%TEMP%`.
**Cause:** If the exe is killed (not gracefully shut down), PyInstaller temp dirs may remain.
**Fix:** Normal behavior; OS cleans them periodically.

### 10. Port already in use
**Symptom:** `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`
**Fix:** Kill the existing process or change the port via environment variable:
```bash
set PORT=9000
AutoML.exe
```

---

## Environment Variable Overrides

When running the exe, you can override settings via environment variables:

```bash
set PORT=9000
set HOST=0.0.0.0
set UPLOAD_DIR=D:\my_uploads
AutoML.exe
```

---

## Rebuilding After Code Changes

1. Make your changes in `backend/` or `frontend/`
2. Run `build.bat` again
3. The `dist/AutoML/` folder is fully replaced

---

## Manual PyInstaller Command (Alternative)

```bash
cd backend
pyinstaller automl.spec --clean --noconfirm
```

Or without a spec file (not recommended — misses hidden imports):
```bash
pyinstaller main.py --name AutoML --console --noconfirm ^
  --add-data "../frontend/dist;frontend_dist" ^
  --add-data "data;data" ^
  --hidden-import sklearn --hidden-import uvicorn ^
  --collect-submodules sklearn --collect-submodules uvicorn
```
