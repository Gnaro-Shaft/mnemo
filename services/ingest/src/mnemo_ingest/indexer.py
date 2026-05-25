"""Indexer : upsert et search dans Qdrant."""

import logging
from types import TracebackType
from typing import Any, Self

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from mnemo_ingest.config import Settings, settings
from mnemo_ingest.models import Chunk

logger = logging.getLogger(__name__)


class QdrantIndexer:
    """Client Qdrant pour upsert et search de chunks."""

    def __init__(self, conf: Settings = settings) -> None:
        self.conf = conf
        self.client = QdrantClient(url=conf.qdrant_url)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.conf.collection_name)

    def ensure_collection(self, reset: bool = False) -> None:
        """Crée la collection si elle n'existe pas (ou la recrée si reset=True).

        Avec reset=True, supprime la collection existante et toutes ses données.
        Utile pour repartir from scratch (changement de modèle, schéma payload, etc.).
        """
        if self.collection_exists():
            if reset:
                logger.info("Resetting collection %s", self.conf.collection_name)
                self.client.delete_collection(self.conf.collection_name)
            else:
                logger.debug("Collection %s already exists", self.conf.collection_name)
                return

        logger.info(
            "Creating collection %s (size=%d, distance=Cosine)",
            self.conf.collection_name,
            self.conf.embedding_dim,
        )
        self.client.create_collection(
            collection_name=self.conf.collection_name,
            vectors_config=VectorParams(
                size=self.conf.embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self, chunks: list[Chunk], vectors: list[list[float]]
    ) -> int:
        """Upsert une liste de chunks avec leurs vecteurs.

        Args:
            chunks: les Chunk à indexer
            vectors: les embeddings (même ordre que chunks)

        Returns:
            Nombre de points upserted.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatched lengths: {len(chunks)} chunks vs {len(vectors)} vectors"
            )
        if not chunks:
            return 0

        points = [
            PointStruct(
                id=str(chunk.id),
                vector=vector,
                payload={
                    "content": chunk.content,
                    "path": chunk.path,
                    "title": chunk.title,
                    "headings_path": chunk.headings_path,
                    "tags": chunk.tags,
                    "frontmatter": chunk.frontmatter,
                    "char_count": chunk.char_count,
                    "token_count": chunk.token_count,
                    "mtime": chunk.mtime,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        self.client.upsert(
            collection_name=self.conf.collection_name,
            points=points,
        )
        return len(points)

    def search(self, query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        """Search par similarité (cosine). Retourne top-K avec score et payload."""
        results = self.client.query_points(
            collection_name=self.conf.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return [
            {"id": str(p.id), "score": p.score, "payload": p.payload}
            for p in results.points
        ]

    def count(self) -> int:
        """Nombre de points dans la collection."""
        return self.client.count(self.conf.collection_name).count

    def delete_by_path(self, path: str) -> None:
        """Supprime tous les chunks d'un fichier (pour réindex incrémentale Phase 3).

        Note: pour de bonnes perfs sur grosse collection, le payload field 'path'
        devrait être indexé via create_payload_index. À faire en Phase 3.
        """
        self.client.delete(
            collection_name=self.conf.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="path", match=MatchValue(value=path))]
            ),
        )
