"""
Configuration module for DRES (Distributed Retrieval Evaluation Server).
Reads settings from environment variables or .env file.
"""

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class DresSettings:
    """Settings for DRES connection and credentials."""

    def __init__(self):
        self.base_url: str = os.getenv("DRES_BASE_URL", "http://192.168.28.151:5000").rstrip("/")
        self.username: str = os.getenv("DRES_USERNAME", "")
        self.password: str = os.getenv("DRES_PASSWORD", "")
        self.timeout: float = float(os.getenv("DRES_TIMEOUT", "10.0"))
        self.retry_count: int = int(os.getenv("DRES_RETRY_COUNT", "1"))
        # Explicit run selection (optional). Priority: run_id > run_name > auto.
        self.run_id: Optional[str] = os.getenv("DRES_RUN_ID", "").strip() or None
        self.run_name: Optional[str] = os.getenv("DRES_RUN_NAME", "").strip() or None

    def __repr__(self) -> str:
        return (
            f"DresSettings(base_url='{self.base_url}', "
            f"username='{self.username}', timeout={self.timeout}, "
            f"run_id='{self.run_id}', run_name='{self.run_name}')"
        )


# Singleton — NOT lru_cached so that runtime reloads can call DresSettings() fresh.
_settings_instance: Optional[DresSettings] = None


def get_dres_settings() -> DresSettings:
    """Return singleton DresSettings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = DresSettings()
    return _settings_instance
