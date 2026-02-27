"""
Itqani — Real-time Arabic → French translation for mosque khutba.

Usage
─────
    cd itqani/
    python main.py

Environment
───────────
    OPENROUTER_API_KEY=sk-or-...   (or set in .env at project root)
"""

import logging
import signal
import sys

from itqani.pipeline.coordinator import Pipeline


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quieten noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.INFO)


def main():
    _setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Itqani starting up…")

    pipeline = Pipeline()

    def _handle_signal(signum, frame):
        logger.info("Received signal %d, shutting down…", signum)
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    pipeline.start()
    pipeline.wait()


if __name__ == "__main__":
    main()
