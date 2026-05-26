"""Modèles de données pour le pipeline d'ingestion."""

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5, NAMESPACE_URL

from pydantic import BaseModel, Field, computed_field


# Namespace UUID pour générer des IDs déterministes
# (un même path+heading donnera toujours le même UUID v5)
MNEMO_UUID_NAMESPACE = uuid5(NAMESPACE_URL, "mnemo://vault")


class Document(BaseModel):
    """Un fichier markdown parsé : frontmatter + contenu."""

    path: Path = Field(description="Chemin relatif au vault root")
    abs_path: Path = Field(description="Chemin absolu sur le filesystem")
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    content: str = Field(description="Markdown brut sans le frontmatter")
    mtime: float = Field(description="Timestamp de dernière modification")

    @computed_field
    @property
    def title(self) -> str:
        """Titre du document, depuis le frontmatter ou le filename."""
        return self.frontmatter.get("nom") or self.path.stem

    @computed_field
    @property
    def tags(self) -> list[str]:
        """Tags du frontmatter (toujours une liste, même si frontmatter mal formé)."""
        raw = self.frontmatter.get("tags", [])
        if isinstance(raw, list):
            return [str(t) for t in raw]
        if isinstance(raw, str):
            return [raw]
        return []


class Chunk(BaseModel):
    """Un morceau de document prêt à être embed et upsert dans Qdrant."""

    id: UUID = Field(description="UUID v5 déterministe depuis path + heading_path + index")
    content: str
    path: str = Field(description="Chemin relatif au vault root (en string pour Qdrant)")
    title: str
    headings_path: list[str] = Field(
        default_factory=list,
        description="Chemin hiérarchique des headings (ex: ['Architecture', 'Stack'])",
    )
    tags: list[str] = Field(default_factory=list)
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    char_count: int = 0
    token_count: int = 0  # approximation: char_count / 4
    mtime: float = 0.0
    chunk_index: int = Field(description="Index du chunk dans son document (0-based)")

    @classmethod
    def make_id(cls, path: str, headings_path: list[str], chunk_index: int) -> UUID:
        """Génère un UUID v5 déterministe pour un chunk.

        Re-ingérer le même document produira les mêmes IDs, ce qui permet
        l'upsert (remplacement) au lieu de créer des doublons.
        """
        name = f"{path}::{':'.join(headings_path)}::{chunk_index}"
        return uuid5(MNEMO_UUID_NAMESPACE, name)


class IngestResult(BaseModel):
    """Stats d'un run d'ingestion."""

    started_at: datetime
    finished_at: datetime | None = None
    vault_path: Path
    documents_processed: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    chunks_upserted: int = 0
    errors: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def duration_seconds(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()
