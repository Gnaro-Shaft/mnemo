"""Chunker hybride : split par heading + sliding window fallback pour grosses sections."""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from mnemo_ingest.config import Settings, settings
from mnemo_ingest.models import Chunk, Document

# Regex pour matcher un heading markdown au début d'une ligne (1 à 6 #)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class Section:
    """Une section logique du document, identifiée par sa hiérarchie de headings."""

    headings_path: list[str]
    text: str


def estimate_tokens(text: str) -> int:
    """Approximation grossière : ~4 chars/token pour FR/EN avec un tokenizer BPE.

    Pour de la précision, utiliser tiktoken (mais c'est overkill pour le sizing de chunks).
    """
    return max(1, len(text) // 4)


def iter_sections(content: str) -> Iterator[Section]:
    """Split un document markdown en sections logiques par heading.

    Track la hiérarchie complète (H1 > H2 > H3) pour chaque section.
    Le contenu avant le premier heading est yieldé avec headings_path=[].
    """
    headings_stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def make_section() -> Section | None:
        text = "\n".join(buffer).strip()
        if not text:
            return None
        return Section(
            headings_path=[title for _, title in headings_stack],
            text=text,
        )

    for line in content.split("\n"):
        match = HEADING_RE.match(line)
        if match:
            # Flush la section précédente
            section = make_section()
            if section:
                yield section
            buffer.clear()

            # Update la stack
            level = len(match.group(1))
            title = match.group(2).strip()
            while headings_stack and headings_stack[-1][0] >= level:
                headings_stack.pop()
            headings_stack.append((level, title))
        else:
            buffer.append(line)

    # Flush la dernière section
    section = make_section()
    if section:
        yield section


def _split_by_chars(text: str, target_chars: int, overlap_chars: int) -> Iterator[str]:
    """Fallback brutal : split par caractères en essayant de couper sur les espaces."""
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start + (target_chars // 2):
                end = last_space
        yield text[start:end].strip()
        if end >= len(text):
            break
        start = end - overlap_chars


def sliding_window(text: str, target_chars: int, overlap_chars: int) -> Iterator[str]:
    """Découpe un texte en morceaux ~target_chars avec overlap.

    Privilégie les coupures sur paragraphes (double newline), puis sur espaces.
    """
    if len(text) <= target_chars:
        yield text
        return

    paragraphs = text.split("\n\n")
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        # Cas dégénéré : un seul paragraphe plus grand que target → split brutal
        if len(para) > target_chars and not current:
            yield from _split_by_chars(para, target_chars, overlap_chars)
            continue

        # Si ajouter ce para fait dépasser, flush et garde dernier para en overlap
        if current_len + len(para) > target_chars and current:
            yield "\n\n".join(current).strip()
            if overlap_chars > 0 and len(current[-1]) <= overlap_chars * 2:
                current = [current[-1]]
                current_len = len(current[-1])
            else:
                current = []
                current_len = 0

        current.append(para)
        current_len += len(para) + 2  # +2 pour les \n\n

    if current:
        yield "\n\n".join(current).strip()

def merge_small_chunks(chunks: list[Chunk], target_chars: int) -> list[Chunk]:
    """Merge consécutivement les petits chunks tant qu'on reste sous target_chars.

    L'ID, headings_path et chunk_index du chunk résultant viennent du PREMIER chunk
    de la séquence mergée. Garantit l'idempotence : ré-ingérer le même fichier
    produit la même séquence de chunks à merger, donc les mêmes IDs.
    """
    if not chunks:
        return chunks

    merged: list[Chunk] = []
    current = chunks[0]

    for nxt in chunks[1:]:
        combined_len = current.char_count + nxt.char_count + 2  # +2 pour "\n\n"
        if combined_len <= target_chars:
            current = current.model_copy(update={
                "content": current.content + "\n\n" + nxt.content,
                "char_count": combined_len,
                "token_count": current.token_count + nxt.token_count,
            })
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


def chunk_document(doc: Document, conf: Settings = settings) -> list[Chunk]:
    """Découpe un Document en liste de Chunks prêts à embed.

    Pipeline :
    1. Split par headings (iter_sections)
    2. Pour chaque section :
       - Si <= max_chars → 1 chunk
       - Sinon → sub-split via sliding window par paragraphes
    3. Préfixe chaque chunk avec son heading_path pour le contexte
    """
    target_chars = conf.chunk_target_tokens * 4
    overlap_chars = conf.chunk_overlap_tokens * 4
    max_chars = target_chars * 2  # seuil au-delà duquel on sub-split

    chunks: list[Chunk] = []
    chunk_index = 0

    for section in iter_sections(doc.content):
        # Préfixe le chunk avec son heading_path (contexte pour l'embedding)
        if section.headings_path:
            section_text = " > ".join(section.headings_path) + "\n\n" + section.text
        else:
            section_text = section.text

        if len(section_text) <= max_chars:
            chunk_texts = [section_text]
        else:
            chunk_texts = list(sliding_window(section_text, target_chars, overlap_chars))

        for text in chunk_texts:
            chunks.append(
                Chunk(
                    id=Chunk.make_id(
                        path=str(doc.path),
                        headings_path=section.headings_path,
                        chunk_index=chunk_index,
                    ),
                    content=text,
                    path=str(doc.path),
                    title=doc.title,
                    headings_path=section.headings_path,
                    tags=doc.tags,
                    frontmatter=doc.frontmatter,
                    char_count=len(text),
                    token_count=estimate_tokens(text),
                    mtime=doc.mtime,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    # Post-traitement : merge des petits chunks consécutifs pour atteindre target_chars
    chunks = merge_small_chunks(chunks, target_chars)
    return chunks
