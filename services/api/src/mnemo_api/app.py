"""FastAPI app : endpoints /healthz, /stats, /query."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from mnemo_core.config import settings as core_settings
from mnemo_core.embedder import OllamaEmbedder
from mnemo_core.indexer import QdrantIndexer

from .deps import Embedder, Indexer, RequireAuth
from .schemas import (
    HealthResponse,
    HealthService,
    QueryRequest,
    QueryResponse,
    SearchHit,
    StatsResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Init des clients Ollama + Qdrant au startup, cleanup au shutdown."""
    logger.info("Starting mnemo-api...")
    app.state.embedder = OllamaEmbedder()
    app.state.indexer = QdrantIndexer()
    logger.info(
        "mnemo-api ready (collection=%s, ollama=%s, qdrant=%s)",
        core_settings.collection_name,
        core_settings.ollama_url,
        core_settings.qdrant_url,
    )
    yield
    logger.info("Shutting down mnemo-api...")
    app.state.embedder.close()
    app.state.indexer.close()


app = FastAPI(
    title="Mnemo API",
    description="Self-hosted RAG over an Obsidian vault (Qdrant + Ollama).",
    version="0.1.0",
    lifespan=lifespan,
)


# ----------------------- /healthz (no auth) -----------------------


@app.get("/healthz", response_model=HealthResponse, tags=["health"])
async def healthz(embedder: Embedder, indexer: Indexer) -> HealthResponse:
    """Healthcheck : vérifie Ollama + Qdrant. Pas d'auth (monitoring)."""
    ollama_health = HealthService(reachable=False)
    qdrant_health = HealthService(reachable=False)

    # Ollama : test d'embed d'une short string
    try:
        await asyncio.to_thread(embedder.embed_query, "ping")
        ollama_health.reachable = True
    except Exception as exc:
        ollama_health.detail = str(exc)[:200]

    # Qdrant : test collection_exists
    try:
        exists = await asyncio.to_thread(indexer.collection_exists)
        qdrant_health.reachable = True
        if not exists:
            qdrant_health.detail = (
                f"Collection '{core_settings.collection_name}' not found"
            )
    except Exception as exc:
        qdrant_health.detail = str(exc)[:200]

    overall = (
        "ok" if (ollama_health.reachable and qdrant_health.reachable) else "degraded"
    )

    return HealthResponse(status=overall, ollama=ollama_health, qdrant=qdrant_health)


# ----------------------- /stats (auth) -----------------------


@app.get("/stats", response_model=StatsResponse, tags=["meta"])
async def stats_endpoint(_: RequireAuth, indexer: Indexer) -> StatsResponse:
    count = await asyncio.to_thread(indexer.count)
    return StatsResponse(
        collection=core_settings.collection_name,
        points_count=count,
        vector_size=core_settings.embedding_dim,
        distance="Cosine",
    )


# ----------------------- /query (auth) -----------------------


@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query_endpoint(
    request: QueryRequest,
    _: RequireAuth,
    embedder: Embedder,
    indexer: Indexer,
) -> QueryResponse:
    t0 = time.perf_counter()

    # 1. Embed la query (offload sur thread pool car sync)
    query_vector = await asyncio.to_thread(embedder.embed_query, request.query)

    # 2. Search Qdrant (over-fetch un peu pour absorber le filtre score/path)
    raw_hits = await asyncio.to_thread(
        indexer.search, query_vector, min(request.limit * 3, 30)
    )

    # 3. Filtre score_threshold + path_prefix
    results: list[SearchHit] = []
    for hit in raw_hits:
        if hit["score"] < request.score_threshold:
            continue
        payload = hit["payload"] or {}
        path = payload.get("path", "")
        if request.path_prefix and not path.startswith(request.path_prefix):
            continue

        headings_path = payload.get("headings_path") or []
        section = " > ".join(headings_path) if headings_path else None

        content = payload.get("content", "")
        preview = content[:500] + ("…" if len(content) > 500 else "")

        results.append(
            SearchHit(
                score=hit["score"],
                path=path,
                title=payload.get("title", ""),
                section=section,
                content_preview=preview,
                chunk_index=payload.get("chunk_index", 0),
            )
        )
        if len(results) >= request.limit:
            break

    took_ms = int((time.perf_counter() - t0) * 1000)
    return QueryResponse(query=request.query, results=results, took_ms=took_ms)
