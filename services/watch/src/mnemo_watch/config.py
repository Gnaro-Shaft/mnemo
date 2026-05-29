"""Configuration du service mnemo-watch."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WatchSettings(BaseSettings):
    """Settings du watcher (env vars prefixées MNEMO_WATCH_, ou .env)."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMO_WATCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    debounce_seconds: float = Field(
        default=3.0,
        description=(
            "Délai de calme après le dernier event sur un path avant de déclencher "
            "le reindex. 3s = compromis entre réactivité et coalescing des rafales "
            "(Obsidian save toutes les 1-2s pendant la frappe)."
        ),
    )


watch_settings = WatchSettings()
