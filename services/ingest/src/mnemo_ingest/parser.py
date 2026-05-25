"""Parser : walk le vault et extrait frontmatter + content de chaque .md."""

import logging
from collections.abc import Iterator
from pathlib import Path

import frontmatter

from mnemo_ingest.models import Document

logger = logging.getLogger(__name__)

# Dossiers à ignorer pendant le walk (matchés sur n'importe quel niveau)
SKIP_DIRS = {".obsidian", ".trash", "Templates"}


def parse_document(abs_path: Path, vault_path: Path) -> Document | None:
    """Parse un fichier markdown en Document.

    Retourne None si le fichier est vide, hors vault, ou si le parsing échoue.
    """
    try:
        post = frontmatter.load(abs_path)
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", abs_path, exc)
        return None

    content = post.content.strip()
    if not content:
        logger.debug("Skipping empty file: %s", abs_path)
        return None

    try:
        relative = abs_path.relative_to(vault_path)
    except ValueError:
        logger.warning("File %s not under vault root %s", abs_path, vault_path)
        return None

    return Document(
        path=relative,
        abs_path=abs_path,
        frontmatter=dict(post.metadata),
        content=content,
        mtime=abs_path.stat().st_mtime,
    )


def iter_documents(vault_path: Path) -> Iterator[Document]:
    """Walk récursif du vault et yield des Documents parsés.

    - Skip les dossiers dans SKIP_DIRS (à n'importe quel niveau)
    - Skip les fichiers cachés (.xxx)
    - Skip les fichiers vides
    - Log + skip silencieux des erreurs de parsing
    """
    if not vault_path.exists():
        raise FileNotFoundError(f"Vault path does not exist: {vault_path}")
    if not vault_path.is_dir():
        raise NotADirectoryError(f"Vault path is not a directory: {vault_path}")

    for md_path in vault_path.rglob("*.md"):
        # Skip si un parent est dans SKIP_DIRS
        if any(part in SKIP_DIRS for part in md_path.relative_to(vault_path).parts):
            continue

        # Skip si filename commence par "."
        if md_path.name.startswith("."):
            continue

        doc = parse_document(md_path, vault_path)
        if doc is not None:
            yield doc
