"""Settings loaded from environment variables."""
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    api_key: str = os.environ.get("API_KEY", "dev-secret-key")
    model_dir: str = os.environ.get("MODEL_DIR", "/app/models")
    redis_url: str = os.environ.get("REDIS_URL", "")
    alert_cooldown_secs: int = int(os.environ.get("ALERT_COOLDOWN_SECS", "300"))
    alert_max_repeat: int = int(os.environ.get("ALERT_MAX_REPEAT", "3"))
    debug_mode: bool = os.environ.get("DEBUG_MODE", "false").lower() == "true"
    model_version: str = os.environ.get("MODEL_VERSION", "v1.0.0")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
