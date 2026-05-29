"""Réindex incrémental d'un fichier .md vers Qdrant.

Pattern : delete_by_path + ré-embed + upsert. Idempotent grâce aux UUID v5
déterministes côté Chunk (path+heading+index → même UUID toujours).
"""

from __future__ import annotations

import logging
from pathlib import Path

import frontmatter

from mnemo_core.embedder import OllamaEmbedder
from mnemo_core.indexer import QdrantIndexer
from mnemo_core.models import Document
from mnemo_ingest.chunker import chunk_document

logger = logging.getLogger(__name__)


class Reindexer:
    """Réindex incrémental : 1 fichier → delete_by_path + ré-embed + upsert."""

    def __init__(
        self,
        vault_root: Path,
        embedder: OllamaEmbedder,
        indexer: QdrantIndexer,
    ) -> None:
        self.vault_root = vault_root.resolve()
        self.embedder = embedder
        self.indexer = indexer

    def _to_relative(self, abs_path: Path) -> str:
        """Chemin relatif au vault root, en str (format payload Qdrant)."""
        return str(abs_path.resolve().relative_to(self.vault_root))

    def unindex(self, abs_path: Path) -> None:
        """Supprime tous les chunks d'un fichier de Qdrant."""
        try:
            rel = self._to_relative(abs_path)
        except ValueError:
            logger.warning("Path outside vault, ignoring: %s", abs_path)
            return
        logger.info("Unindexing %s", rel)
        try:
            self.indexer.delete_by_path(rel)
        except Exception:
            logger.exception("delete_by_path failed for %s", rel)

    def reindex(self, abs_path: Path) -> None:
        """Re-parse + chunk + embed + upsert un fichier (idempotent)."""
        # File supprimé entre l'event et le debounce → traiter comme delete
        if not abs_path.exists():
            self.unindex(abs_path)
            return

        try:
            rel = self._to_relative(abs_path)
        except ValueError:
            logger.warning("Path outside vault, ignoring: %s", abs_path)
            return

        logger.info("Reindexing %s", rel)

        # 1. Parse
        try:
            post = frontmatter.load(abs_path)
            doc = Document(
                path=Path(rel),
                abs_path=abs_path,
                frontmatter=dict(post.metadata),
                content=post.content,
                mtime=abs_path.stat().st_mtime,
            )
        except Exception:
            logger.exception("Failed to parse %s", rel)
            return

        # 2. Chunk
        chunks = list(chunk_document(doc))
        if not chunks:
            # Document vide après strip frontmatter → supprimer ce qui était indexé
            logger.info("  → empty document, unindexing")
            self.unindex(abs_path)
            return

        # 3. Embed
        try:
            vectors = self.embedder.embed_documents([c.content for c in chunks])
        except Exception:
            logger.exception("Failed to embed %s", rel)
            return

        # 4. Delete (avant upsert) : si le chunking a produit moins de chunks qu'avant
        #    (ex: doc raccourci), les anciens orphelins seraient gardés sinon.
        try:
            self.indexer.delete_by_path(rel)
        except Exception:
            logger.exception("delete_by_path failed for %s, continuing", rel)

        # 5. Upsert
        try:
            n = self.indexer.upsert_chunks(chunks, vectors)
            logger.info("  → %d chunks upserted", n)
        except Exception:
            logger.exception("upsert_chunks failed for %s", rel)
