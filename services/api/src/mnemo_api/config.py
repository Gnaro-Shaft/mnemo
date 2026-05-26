"""Configuration du service mnemo-api."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Configuration du service API.

    Toutes les variables sont surchargeables via env vars (préfixe MNEMO_API_)
    ou via un fichier .env à la racine du projet.

    Cas particulier `api_key` : on utilise validation_alias='MNEMO_API_KEY'
    pour éviter la collision sémantique "MNEMO_API_API_KEY" (double 'API').
    """

    model_config = SettingsConfigDict(
        env_prefix="MNEMO_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="Adresse d'écoute uvicorn")
    port: int = Field(default=8000, description="Port d'écoute uvicorn")
    api_key: str = Field(
        default="change-me",
        validation_alias="MNEMO_API_KEY",
        description="API key requise dans le header X-API-Key (sauf /healthz)",
    )

    # Limites pour /query
    default_limit: int = Field(default=5)
    max_limit: int = Field(default=20)
    default_score_threshold: float = Field(default=0.6)


api_settings = ApiSettings()
