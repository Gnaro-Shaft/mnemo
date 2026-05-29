"""Watchdog handler avec debouncing pour coalescer les rafales d'events."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from .reindex import Reindexer

logger = logging.getLogger(__name__)

SKIP_DIRS = {".obsidian", ".trash", "Templates"}


class Debouncer:
    """Coalesce des events rapides en un seul callback après N secondes de calme.

    Pour chaque path, on garde un timer. À chaque nouveau event sur ce path,
    on cancel le timer et on en relance un nouveau. Quand un timer expire
    sans être interrompu, on appelle le callback.
    """

    def __init__(self, delay_seconds: float, callback: Callable[[Path], None]) -> None:
        self.delay = delay_seconds
        self.callback = callback
        self.timers: dict[Path, threading.Timer] = {}
        self.lock = threading.Lock()

    def trigger(self, path: Path) -> None:
        with self.lock:
            existing = self.timers.pop(path, None)
            if existing is not None:
                existing.cancel()

            timer = threading.Timer(self.delay, self._fire, args=(path,))
            timer.daemon = True
            self.timers[path] = timer
            timer.start()

    def _fire(self, path: Path) -> None:
        with self.lock:
            self.timers.pop(path, None)
        try:
            self.callback(path)
        except Exception:
            logger.exception("Debouncer callback failed for %s", path)

    def shutdown(self) -> None:
        with self.lock:
            for t in self.timers.values():
                t.cancel()
            self.timers.clear()


class VaultEventHandler(FileSystemEventHandler):
    """Filtre + route les events watchdog vers reindex/unindex."""

    def __init__(
        self,
        vault_root: Path,
        reindexer: Reindexer,
        debouncer: Debouncer,
    ) -> None:
        super().__init__()
        self.vault_root = vault_root.resolve()
        self.reindexer = reindexer
        self.debouncer = debouncer

    def _is_relevant(self, path: str) -> bool:
        """Seuls les .md sous vault, hors .obsidian/.trash/Templates."""
        p = Path(path)
        if p.suffix.lower() != ".md":
            return False
        try:
            rel = p.resolve().relative_to(self.vault_root)
        except ValueError:
            return False
        return not any(part in SKIP_DIRS for part in rel.parts)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory or not self._is_relevant(event.src_path):
            return
        logger.info("created  → %s", event.src_path)
        self.debouncer.trigger(Path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory or not self._is_relevant(event.src_path):
            return
        logger.debug("modified → %s", event.src_path)
        self.debouncer.trigger(Path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory or not self._is_relevant(event.src_path):
            return
        logger.info("deleted  → %s", event.src_path)
        # Delete = immédiat (pas de risque de rafale)
        self.reindexer.unindex(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if self._is_relevant(event.src_path):
            logger.info("moved out → %s", event.src_path)
            self.reindexer.unindex(Path(event.src_path))
        if self._is_relevant(event.dest_path):
            logger.info("moved in  → %s", event.dest_path)
            self.debouncer.trigger(Path(event.dest_path))


def start_observer(
    vault_root: Path,
    reindexer: Reindexer,
    debounce_seconds: float,
) -> tuple[BaseObserver, Debouncer]:
    """Démarre le watcher. Retourne (observer, debouncer) à stopper proprement."""
    debouncer = Debouncer(debounce_seconds, reindexer.reindex)
    handler = VaultEventHandler(vault_root, reindexer, debouncer)
    observer = Observer()
    observer.schedule(handler, str(vault_root), recursive=True)
    observer.start()
    return observer, debouncer
