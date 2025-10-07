"""Application entrypoint for running the AudiobookGen API."""
from __future__ import annotations

import logging
import os

import uvicorn

from audiobookgen import create_app


logging.basicConfig(level=os.getenv("AUDIOBOOKGEN_LOG_LEVEL", "INFO"))


def main() -> None:
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
