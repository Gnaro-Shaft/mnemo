"""Embedder : appelle Ollama pour générer des embeddings vectoriels."""

import logging
from collections.abc import Iterable
from types import TracebackType
from typing import Self

import httpx

from mnemo_core.config import Settings, settings

logger = logging.getLogger(__name__)

# Prefixes recommandés par Nomic pour leur modèle multilingue.
# Voir : https://blog.nomic.ai/posts/nomic-embed-text-v1
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbedderError(Exception):
    """Erreur durant la génération d'embeddings."""


class OllamaEmbedder:
    """Client Ollama pour la génération d'embeddings.

    Utilise une session HTTP persistante (réutilisation de connexion = perf).
    Sync. Pour async, voir Phase 3 (auto-ingestion temps réel).
    """

    def __init__(self, conf: Settings = settings, timeout: float = 300.0) -> None:
        self.conf = conf
        self.client = httpx.Client(base_url=conf.ollama_url, timeout=timeout)

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

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        """Appel bas-niveau à l'API Ollama /api/embed."""
        try:
            response = self.client.post(
                "/api/embed",
                json={"model": self.conf.embedding_model, "input": inputs},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbedderError(f"Ollama call failed: {exc}") from exc

        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(inputs):
            raise EmbedderError(
                f"Unexpected response: expected {len(inputs)} embeddings, "
                f"got {len(embeddings) if embeddings else 0}"
            )
        return embeddings

    def embed_documents(
        self, texts: Iterable[str], batch_size: int = 16
    ) -> list[list[float]]:
        """Embed une liste de documents (pour indexation).

        Ajoute automatiquement le prefix 'search_document:'. Batch par groupes
        de batch_size pour limiter les round-trips réseau.
        """
        texts = list(texts)
        if not texts:
            return []

        if self.conf.use_embedding_prefixes:
            prefixed = [DOCUMENT_PREFIX + t for t in texts]
        else:
            prefixed = list(texts)
        all_embeddings: list[list[float]] = []

        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            logger.debug("Embedding batch %d-%d / %d", i, i + len(batch), len(prefixed))
            all_embeddings.extend(self._embed(batch))

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed une requête utilisateur (pour search).

        Ajoute automatiquement le prefix 'search_query:'. Pas de batch (typiquement 1 query).
        """
        prefix = QUERY_PREFIX if self.conf.use_embedding_prefixes else ""
        result = self._embed([prefix + text])
        return result[0]
