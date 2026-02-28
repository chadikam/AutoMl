"""
Main FastAPI application for AutoML Framework
Initializes the app, configures middleware, and registers routes
"""
import multiprocessing
multiprocessing.freeze_support()  # Required for PyInstaller on Windows

import matplotlib
matplotlib.use("Agg")  # Headless backend — no GUI needed for plot generation

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import os
import sys

from app.paths import IS_FROZEN, BUNDLE_DIR, RUNTIME_DIR, get_runtime_path, get_bundle_path
from app.config import settings
from app.storage import init_storage
from app.routes import datasets, models, automl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _ensure_runtime_dirs():
    """Create writable directories next to the .exe (or in dev CWD)."""
    for d in (settings.upload_dir, settings.models_dir, settings.data_dir):
        os.makedirs(d, exist_ok=True)


def _seed_data_files():
    """
    On first run of the frozen exe, copy seed JSON files from the bundle
    into the writable data/ directory so the app has its initial state.
    """
    if not IS_FROZEN:
        return
    bundle_data = get_bundle_path("data")
    runtime_data = get_runtime_path("data")
    if not bundle_data.exists():
        return
    for src_file in bundle_data.glob("*.json"):
        dest_file = runtime_data / src_file.name
        if not dest_file.exists():
            import shutil
            shutil.copy2(src_file, dest_file)
            logger.info(f"Seeded {dest_file.name} from bundle")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting AutoML Framework API...")
    _ensure_runtime_dirs()
    _seed_data_files()
    init_storage()
    logger.info(f"📂 Runtime directory: {RUNTIME_DIR}")
    logger.info("✅ Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AutoML Framework API...")
    logger.info("✅ Application shut down successfully")


# Initialize FastAPI application
app = FastAPI(
    title="AutoML Framework API",
    description="Open-source automated machine learning framework with adaptive preprocessing and Optuna optimization",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS - Must be before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Register routers
app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(automl.router)

# Mount static files for uploads (writable runtime path)
uploads_dir = Path(settings.upload_dir)
uploads_dir.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

# ── Serve frontend build (when frozen, the dist/ is bundled) ─────────
_frontend_dir = get_bundle_path("frontend_dist")
if _frontend_dir.is_dir():
    # Serve static assets (JS/CSS/images) under /assets
    _assets_dir = _frontend_dir / "assets"
    if _assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="frontend_assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Catch-all: serve the SPA index.html for any non-API route."""
        file_path = _frontend_dir / full_path
        if full_path and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_frontend_dir / "index.html"))
else:
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "name": "AutoML Framework API",
            "version": "2.0.0",
            "status": "running",
            "documentation": "/docs"
        }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "AutoML Framework API"
    }


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    host = settings.host
    port = settings.port

    if IS_FROZEN:
        # When frozen: open browser automatically, no reload
        def _open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(f"http://127.0.0.1:{port}")

        threading.Thread(target=_open_browser, daemon=True).start()
        uvicorn.run(
            app,            # pass the app object directly (no import string)
            host=host,
            port=port,
            log_level="info",
        )
    else:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
        )
