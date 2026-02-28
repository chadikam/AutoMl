"""
Configuration module for AutoML Framework
Manages environment variables and application settings
"""
import os
from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import Optional

from app.paths import RUNTIME_DIR


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Relative directory paths are resolved against RUNTIME_DIR
    (the .exe folder when frozen, or the backend/ folder in dev).
    """
    # Application Settings
    upload_dir: str = "uploads"
    models_dir: str = "trained_models"
    data_dir: str = "data"
    max_upload_size: int = 52428800  # 50MB in bytes
    
    # CORS Settings
    cors_origins: str = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:8000,http://localhost:8000"
    
    # Server Settings
    host: str = "127.0.0.1"
    port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    @model_validator(mode="after")
    def _resolve_dirs(self) -> "Settings":
        """Make all directory settings absolute (anchored to RUNTIME_DIR)."""
        for field in ("upload_dir", "models_dir", "data_dir"):
            val = getattr(self, field)
            if not os.path.isabs(val):
                setattr(self, field, str(RUNTIME_DIR / val))
        return self

    @property
    def cors_origins_list(self) -> list:
        """Convert cors_origins string to list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Create settings instance
settings = Settings()
