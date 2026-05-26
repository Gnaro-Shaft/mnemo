"""Configuration du service mnemo-ingest.

Toutes les variables peuvent être surchargées via env vars (préfixe MNEMO_)
ou via un fichier .env à la racine du projet.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration globale du pipeline d'ingestion."""

    model_config = SettingsConfigDict(
        env_prefix="MNEMO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    vault_path: Path = Field(
        default=Path("/home/dgnaro/storage/mnemo/vault-mirror"),
        description="Chemin absolu vers le vault matérialisé (output Phase 1.A)",
    )

    # --- Ollama (embeddings) ---
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="URL de l'API Ollama",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Nom du modèle Ollama pour les embeddings",
    )
    embedding_dim: int = Field(
        default=768,
        description="Dimension des vecteurs (768 pour nomic-embed-text)",
    )

    # --- Qdrant ---
    qdrant_url: str = Field(
        default="http://127.0.0.1:6333",
        description="URL de l'API Qdrant",
    )
    collection_name: str = Field(
        default="vault",
        description="Nom de la collection Qdrant",
    )

    # --- Chunking ---
    chunk_target_tokens: int = Field(
        default=600,
        description="Taille cible d'un chunk en tokens (approximation: chars/4)",
    )
    chunk_overlap_tokens: int = Field(
        default=100,
        description="Overlap entre chunks consécutifs pour le sliding window fallback",
    )
    chunk_min_tokens: int = Field(
        default=50,
        description="Taille minimale d'un chunk avant merge avec le suivant",
    )


settings = Settings()
