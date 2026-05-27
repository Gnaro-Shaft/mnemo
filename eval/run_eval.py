"""Mnemo — Évaluation du retrieval RAG.

Usage:
    cd ~/projects/mnemo
    uv run python eval/run_eval.py --tag baseline
    uv run python eval/run_eval.py --tag bge-m3 --limit 10

Calcule Hit@1, Hit@3, Hit@5, MRR, et score moyen top-1.
Génère un rapport markdown dans eval/results/<timestamp>-<tag>.md.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import Any

import yaml

from mnemo_core.config import settings
from mnemo_core.embedder import OllamaEmbedder
from mnemo_core.indexer import QdrantIndexer


logger = logging.getLogger("mnemo_eval")
EVAL_DIR = Path(__file__).parent
QUERIES_FILE = EVAL_DIR / "queries.yaml"
RESULTS_DIR = EVAL_DIR / "results"


def load_queries() -> list[dict[str, Any]]:
    with QUERIES_FILE.open() as f:
        data = yaml.safe_load(f)
    return data["queries"]


def reciprocal_rank(hit_paths: list[str], expected: list[str]) -> float:
    """1 / rank du premier expected trouvé dans hit_paths (ou 0 si aucun)."""
    for i, p in enumerate(hit_paths, start=1):
        if p in expected:
            return 1.0 / i
    return 0.0


def hit_at_k(hit_paths: list[str], expected: list[str], k: int) -> int:
    """1 si au moins un expected est dans les K premiers hits, sinon 0."""
    return int(any(p in expected for p in hit_paths[:k]))


def first_rank(hit_paths: list[str], expected: list[str]) -> int | None:
    """Rang (1-indexed) du premier expected trouvé, ou None."""
    for i, p in enumerate(hit_paths, start=1):
        if p in expected:
            return i
    return None


def run_eval(tag: str, limit: int = 10) -> None:
    queries = load_queries()
    logger.info("Loaded %d queries", len(queries))

    embedder = OllamaEmbedder()
    indexer = QdrantIndexer()

    rows: list[dict[str, Any]] = []
    by_cat: dict[str, list[dict[str, Any]]] = {}

    for q in queries:
        text = q["query"]
        expected = q["expected_paths"]
        cat = q.get("category", "uncategorized")

        vec = embedder.embed_query(text)
        raw = indexer.search(vec, limit=limit)
        hit_paths = [r["payload"].get("path", "") for r in raw]
        top_score = raw[0]["score"] if raw else 0.0

        row = {
            "id": q["id"],
            "category": cat,
            "query": text,
            "expected_paths": expected,
            "hit_paths_top5": hit_paths[:5],
            "rank": first_rank(hit_paths, expected),
            "hit@1": hit_at_k(hit_paths, expected, 1),
            "hit@3": hit_at_k(hit_paths, expected, 3),
            "hit@5": hit_at_k(hit_paths, expected, 5),
            "rr": reciprocal_rank(hit_paths, expected),
            "top_score": top_score,
        }
        rows.append(row)
        by_cat.setdefault(cat, []).append(row)

    embedder.close()
    indexer.close()

    # ---------- Métriques globales ----------
    n = len(rows)
    mrr = sum(r["rr"] for r in rows) / n
    hit1 = sum(r["hit@1"] for r in rows) / n
    hit3 = sum(r["hit@3"] for r in rows) / n
    hit5 = sum(r["hit@5"] for r in rows) / n
    avg_top = sum(r["top_score"] for r in rows) / n

    # ---------- Rapport ----------
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = RESULTS_DIR / f"{ts}-{tag}.md"

    lines: list[str] = []
    lines.append(f"# Eval Mnemo — `{tag}` ({ts})\n")
    lines.append(f"**Settings** : model=`{settings.embedding_model}` "
                 f"dim={settings.embedding_dim} collection=`{settings.collection_name}` "
                 f"queries={n}\n")

    lines.append("## Métriques globales\n")
    lines.append(f"| Métrique | Valeur |")
    lines.append(f"|---|---|")
    lines.append(f"| Hit@1 | **{hit1:.2%}** ({sum(r['hit@1'] for r in rows)}/{n}) |")
    lines.append(f"| Hit@3 | **{hit3:.2%}** ({sum(r['hit@3'] for r in rows)}/{n}) |")
    lines.append(f"| Hit@5 | **{hit5:.2%}** ({sum(r['hit@5'] for r in rows)}/{n}) |")
    lines.append(f"| MRR   | **{mrr:.3f}** |")
    lines.append(f"| Score moyen top-1 | **{avg_top:.3f}** |\n")

    lines.append("## Métriques par catégorie\n")
    lines.append("| Catégorie | N | Hit@1 | Hit@3 | MRR | Score top-1 |")
    lines.append("|---|---|---|---|---|---|")
    for cat, items in sorted(by_cat.items()):
        nc = len(items)
        c_hit1 = sum(r["hit@1"] for r in items) / nc
        c_hit3 = sum(r["hit@3"] for r in items) / nc
        c_mrr = sum(r["rr"] for r in items) / nc
        c_score = sum(r["top_score"] for r in items) / nc
        lines.append(f"| {cat} | {nc} | {c_hit1:.0%} | {c_hit3:.0%} | "
                     f"{c_mrr:.3f} | {c_score:.3f} |")
    lines.append("")

    lines.append("## Détail par query\n")
    lines.append("| ID | Cat | Query | Rank | Hit@1 | Hit@5 | Score |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        rank_str = str(r["rank"]) if r["rank"] is not None else "—"
        emoji_h1 = "✅" if r["hit@1"] else "❌"
        emoji_h5 = "✅" if r["hit@5"] else "❌"
        lines.append(f"| `{r['id']}` | {r['category']} | {r['query']} | "
                     f"{rank_str} | {emoji_h1} | {emoji_h5} | {r['top_score']:.3f} |")
    lines.append("")

    lines.append("## Misses (queries qui ratent Hit@5)\n")
    for r in rows:
        if r["hit@5"]:
            continue
        lines.append(f"### `{r['id']}` — {r['query']}\n")
        lines.append(f"- **Expected** :")
        for p in r["expected_paths"]:
            lines.append(f"  - `{p}`")
        lines.append(f"- **Actual top-5** :")
        for p in r["hit_paths_top5"]:
            lines.append(f"  - `{p}`")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Affichage console
    print()
    print(f"╔══ Eval `{tag}` ══════════════════════════════════════╗")
    print(f"║  Model: {settings.embedding_model:<40}     ║")
    print(f"║  Hit@1: {hit1:.2%}  Hit@3: {hit3:.2%}  Hit@5: {hit5:.2%}             ║")
    print(f"║  MRR:   {mrr:.3f}             Score top-1: {avg_top:.3f}     ║")
    print(f"╚════════════════════════════════════════════════════╝")
    print(f"Report: {report_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run Mnemo RAG evaluation suite")
    p.add_argument("--tag", required=True,
                   help="Tag for this run (e.g. 'baseline', 'bge-m3', 'no-prefix')")
    p.add_argument("--limit", type=int, default=10,
                   help="Top-K to fetch from Qdrant (default 10)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_eval(args.tag, args.limit)


if __name__ == "__main__":
    main()
