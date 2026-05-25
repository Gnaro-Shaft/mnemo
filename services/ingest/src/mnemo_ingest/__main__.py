"""CLI orchestrateur pour le pipeline d'ingestion Mnemo."""

import argparse
import logging
import sys
import time
from datetime import datetime

from mnemo_ingest.chunker import chunk_document
from mnemo_ingest.config import settings
from mnemo_ingest.embedder import OllamaEmbedder
from mnemo_ingest.indexer import QdrantIndexer
from mnemo_ingest.models import IngestResult
from mnemo_ingest.parser import iter_documents


def setup_logging(verbose: bool = False) -> None:
    """Configure le logging racine."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run le pipeline d'ingestion complet."""
    log = logging.getLogger("mnemo_ingest.cli")

    result = IngestResult(
        started_at=datetime.now(),
        vault_path=settings.vault_path,
    )

    with OllamaEmbedder(settings) as embedder, QdrantIndexer(settings) as indexer:
        indexer.ensure_collection(reset=args.reset)
        log.info(
            "Collection %s ready (count=%d, reset=%s)",
            settings.collection_name,
            indexer.count(),
            args.reset,
        )

        # Phase 1 : parse + chunk
        log.info("Parsing vault: %s", settings.vault_path)
        docs = list(iter_documents(settings.vault_path))
        if args.limit:
            docs = docs[: args.limit]
            log.info("Limiting to first %d documents", args.limit)

        all_chunks = []
        for i, doc in enumerate(docs, 1):
            try:
                chunks = chunk_document(doc)
                all_chunks.extend(chunks)
                result.documents_processed += 1
                if i % 25 == 0 or i == len(docs):
                    log.info(
                        "Parsed %d/%d documents → %d chunks",
                        i, len(docs), len(all_chunks),
                    )
            except Exception as exc:
                log.warning("Failed to chunk %s: %s", doc.path, exc)
                result.documents_failed += 1
                result.errors.append(f"{doc.path}: {exc}")

        result.chunks_created = len(all_chunks)
        log.info(
            "Total: %d chunks from %d documents",
            len(all_chunks), result.documents_processed,
        )

        if not all_chunks:
            log.warning("No chunks to embed, exiting")
            result.finished_at = datetime.now()
            print(result.model_dump_json(indent=2))
            return 0

        # Phase 2 : embed (le plus long sur CPU)
        eta = len(all_chunks) * 2.5
        log.info(
            "Embedding %d chunks (~%.0fs estimated on CPU)...",
            len(all_chunks), eta,
        )
        embed_start = time.time()
        vectors = embedder.embed_documents([c.content for c in all_chunks])
        embed_elapsed = time.time() - embed_start
        log.info(
            "Embedded in %.1fs (%.1f chunks/s)",
            embed_elapsed, len(all_chunks) / embed_elapsed,
        )

        # Phase 3 : upsert
        log.info("Upserting %d points to Qdrant...", len(all_chunks))
        upsert_start = time.time()
        n = indexer.upsert_chunks(all_chunks, vectors)
        upsert_elapsed = time.time() - upsert_start
        log.info("Upserted %d points in %.2fs", n, upsert_elapsed)
        result.chunks_upserted = n

        # Stats finales
        result.finished_at = datetime.now()
        log.info("Final count in collection: %d", indexer.count())

    print()
    print("=" * 60)
    print("INGEST COMPLETE")
    print("=" * 60)
    print(result.model_dump_json(indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Run une query de search."""
    log = logging.getLogger("mnemo_ingest.cli")

    with OllamaEmbedder(settings) as embedder, QdrantIndexer(settings) as indexer:
        if not indexer.collection_exists():
            log.error(
                "Collection %s does not exist. Run 'ingest' first.",
                settings.collection_name,
            )
            return 1

        log.info("Query: %s", args.query)
        qvec = embedder.embed_query(args.query)
        results = indexer.search(qvec, limit=args.limit)

        print()
        print(f"Top {len(results)} results for: {args.query!r}")
        print("=" * 60)
        for i, r in enumerate(results, 1):
            payload = r["payload"]
            print(f"\n[{i}] score={r['score']:.3f}  {payload['title']}")
            print(f"    path: {payload['path']}")
            if payload.get("headings_path"):
                print(f"    section: {' > '.join(payload['headings_path'])}")
            print(f"    preview: {payload['content'][:200]}...")

    return 0


def cmd_count(args: argparse.Namespace) -> int:
    """Affiche le nombre de points dans la collection."""
    with QdrantIndexer(settings) as indexer:
        if not indexer.collection_exists():
            print(f"Collection '{settings.collection_name}' does not exist.")
            return 1
        print(f"Collection '{settings.collection_name}': {indexer.count()} points")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mnemo-ingest",
        description="Mnemo ingest: parse vault → chunk → embed → upsert to Qdrant",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Run the full ingest pipeline")
    p_ingest.add_argument(
        "--reset", action="store_true", help="Drop and recreate collection"
    )
    p_ingest.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N documents (useful for testing)",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # search
    p_search = subparsers.add_parser("search", help="Search the vault")
    p_search.add_argument("query", type=str, help="The search query")
    p_search.add_argument("--limit", type=int, default=5, help="Number of results")
    p_search.set_defaults(func=cmd_search)

    # count
    p_count = subparsers.add_parser(
        "count", help="Show number of points in the collection"
    )
    p_count.set_defaults(func=cmd_count)

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
