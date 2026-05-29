"""Entry point: python -m mnemo_watch → service de surveillance du vault."""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time

from mnemo_core.config import settings as core_settings
from mnemo_core.embedder import OllamaEmbedder
from mnemo_core.indexer import QdrantIndexer

from .config import watch_settings
from .handler import start_observer
from .reindex import Reindexer

logger = logging.getLogger("mnemo_watch")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    vault = core_settings.vault_path.resolve()
    if not vault.exists():
        logger.error("Vault path does not exist: %s", vault)
        sys.exit(1)

    logger.info("Starting mnemo-watch")
    logger.info("  vault     = %s", vault)
    logger.info("  debounce  = %.1fs", watch_settings.debounce_seconds)
    logger.info("  qdrant    = %s (collection=%s)",
                core_settings.qdrant_url, core_settings.collection_name)
    logger.info("  ollama    = %s (model=%s)",
                core_settings.ollama_url, core_settings.embedding_model)

    embedder = OllamaEmbedder()
    indexer = QdrantIndexer()
    reindexer = Reindexer(vault, embedder, indexer)

    if not indexer.collection_exists():
        logger.error(
            "Qdrant collection '%s' does not exist — "
            "run `uv run python -m mnemo_ingest ingest --reset` first",
            core_settings.collection_name,
        )
        embedder.close()
        indexer.close()
        sys.exit(1)

    observer, debouncer = start_observer(
        vault, reindexer, watch_settings.debounce_seconds,
    )
    logger.info("Watching for changes... (Ctrl+C or SIGTERM to stop)")

    shutdown = threading.Event()

    def _handle_signal(signum: int, _frame) -> None:
        logger.info("Received signal %d, shutting down...", signum)
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while not shutdown.is_set():
            time.sleep(0.5)
    finally:
        logger.info("Stopping observer...")
        observer.stop()
        observer.join(timeout=5)
        debouncer.shutdown()
        embedder.close()
        indexer.close()
        logger.info("Bye.")


if __name__ == "__main__":
    main()
