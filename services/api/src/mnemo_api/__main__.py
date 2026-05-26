"""Entry point: python -m mnemo_api → lance uvicorn."""

import logging

import uvicorn

from .config import api_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run(
        "mnemo_api.app:app",
        host=api_settings.host,
        port=api_settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
