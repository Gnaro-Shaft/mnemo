"""FastAPI dependencies : auth + clients singletons."""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from mnemo_core.embedder import OllamaEmbedder
from mnemo_core.indexer import QdrantIndexer

from .config import api_settings


async def verify_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Vérifie l'API key dans le header X-API-Key."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    if x_api_key != api_settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid X-API-Key"
        )


def get_embedder(request: Request) -> OllamaEmbedder:
    """Récupère l'instance OllamaEmbedder partagée (initialisée au startup)."""
    return request.app.state.embedder


def get_indexer(request: Request) -> QdrantIndexer:
    """Récupère l'instance QdrantIndexer partagée (initialisée au startup)."""
    return request.app.state.indexer


# Type aliases pour des signatures de routes plus lisibles
RequireAuth = Annotated[None, Depends(verify_api_key)]
Embedder = Annotated[OllamaEmbedder, Depends(get_embedder)]
Indexer = Annotated[QdrantIndexer, Depends(get_indexer)]
