"""Pydantic schemas pour l'API mnemo-api."""

from pydantic import BaseModel, Field


# ---------- Requests ----------


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1, max_length=500, description="Requête en langage naturel"
    )
    limit: int = Field(default=5, ge=1, le=20, description="Nombre de résultats")
    score_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Score cosine minimum"
    )
    path_prefix: str | None = Field(
        default=None,
        description="Filtre : ne retourner que les chunks dont path commence par",
    )


# ---------- Responses ----------


class SearchHit(BaseModel):
    score: float = Field(description="Score de similarité cosine [0, 1]")
    path: str
    title: str
    section: str | None = Field(
        default=None, description="Headings hiérarchiques joints par ' > '"
    )
    content_preview: str = Field(description="Premiers 500 chars du chunk")
    chunk_index: int


class QueryResponse(BaseModel):
    query: str
    results: list[SearchHit]
    took_ms: int


class HealthService(BaseModel):
    reachable: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    version: str = "0.1.0"
    ollama: HealthService
    qdrant: HealthService


class StatsResponse(BaseModel):
    collection: str
    points_count: int
    vector_size: int
    distance: str = "Cosine"
