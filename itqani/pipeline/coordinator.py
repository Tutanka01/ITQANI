"""
Pipeline coordinator — wires up all queues and threads, then starts them.

Thread map
──────────
1. audio_thread   — AudioCapture (sounddevice callback)
2. vad_thread     — VADChunker   (silero-vad)
3. whisper_thread — Transcriber  (faster-whisper)
4. translate_thread — Translator (OpenRouter streaming)
5. server_thread  — uvicorn      (FastAPI + WebSocket)
"""

import asyncio
import logging
import queue
import signal
import threading

import uvicorn

from itqani.audio.capture import AudioCapture
from itqani.audio.vad import VADChunker
from itqani.context.manager import ContextManager
from itqani.server.app import app, make_sync_broadcast
from itqani.transcription.transcriber import Transcriber
from itqani.translation.translator import Translator
from itqani import config

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self):
        # Queues
        self.audio_queue = queue.Queue(maxsize=50)
        self.chunk_queue = queue.Queue(maxsize=5)
        self.transcript_queue = queue.Queue(maxsize=5)

        # Shared stop event
        self.stop_event = threading.Event()

        # Shared context window
        self.context_manager = ContextManager()

        # Will be set once the uvicorn event loop is known
        self._server_loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()

    # ------------------------------------------------------------------ #
    # Server thread                                                        #
    # ------------------------------------------------------------------ #

    def _run_server(self):
        """Run uvicorn in its own thread, capturing its event loop."""
        uv_config = uvicorn.Config(
            app,
            host=config.SERVER_HOST,
            port=config.SERVER_PORT,
            log_level="warning",
        )
        server = uvicorn.Server(uv_config)

        # Patch the startup to capture the event loop
        original_startup = server.startup

        async def _patched_startup(sockets=None):
            self._server_loop = asyncio.get_event_loop()
            self._loop_ready.set()
            await original_startup(sockets)

        server.startup = _patched_startup

        # Disable uvicorn's own signal handlers so our main handler works
        server.install_signal_handlers = lambda: None

        server.run()

    # ------------------------------------------------------------------ #
    # Start / stop                                                         #
    # ------------------------------------------------------------------ #

    def start(self):
        logger.info("Starting pipeline...")
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Logs directory: %s", config.LOGS_DIR)

        # 1. Server thread (must start first so we can get the event loop)
        server_thread = threading.Thread(
            target=self._run_server, name="server", daemon=True
        )
        server_thread.start()

        # Wait for server event loop to be ready (max 15s)
        if not self._loop_ready.wait(timeout=15):
            raise RuntimeError("Server failed to start within 15 seconds")
        logger.info("Server started on %s:%d", config.SERVER_HOST, config.SERVER_PORT)

        sync_broadcast = make_sync_broadcast(self._server_loop)

        # 2. Whisper thread
        transcriber = Transcriber(
            self.chunk_queue, self.transcript_queue, self.stop_event
        )
        whisper_thread = threading.Thread(
            target=transcriber.run, name="transcriber", daemon=True
        )

        # 3. Translator thread
        translator = Translator(
            self.transcript_queue, sync_broadcast, self.context_manager, self.stop_event
        )
        translate_thread = threading.Thread(
            target=translator.run, name="translator", daemon=True
        )

        # 4. VAD thread
        vad = VADChunker(self.audio_queue, self.chunk_queue, self.stop_event)
        vad_thread = threading.Thread(target=vad.run, name="vad", daemon=True)

        # 5. Audio capture thread
        capture = AudioCapture(self.audio_queue, self.stop_event)
        audio_thread = threading.Thread(
            target=capture.run, name="audio_capture", daemon=True
        )

        # Start processing threads (order: models first, then audio)
        whisper_thread.start()
        translate_thread.start()
        vad_thread.start()
        audio_thread.start()

        logger.info("All pipeline threads started. Press Ctrl+C to stop.")

        self._threads = [
            server_thread,
            whisper_thread,
            translate_thread,
            vad_thread,
            audio_thread,
        ]

    def wait(self):
        """Block until stop_event is set (e.g. by SIGINT)."""
        try:
            self.stop_event.wait()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        logger.info("Stopping pipeline...")
        self.stop_event.set()
