"""
Thread 1 — sounddevice audio capture.

Reads raw int16 frames from the default input device and puts them
into audio_queue as numpy arrays of shape (CHUNK_FRAMES,).
"""

import logging
import queue
import threading

import numpy as np
import sounddevice as sd

from itqani import config

logger = logging.getLogger(__name__)


class AudioCapture:
    def __init__(self, audio_queue: queue.Queue, stop_event: threading.Event):
        self._queue = audio_queue
        self._stop = stop_event

    def _callback(self, indata: np.ndarray, frames: int, time, status):
        if status:
            logger.warning("sounddevice status: %s", status)
        # indata shape: (frames, channels) — take first channel, copy to avoid aliasing
        chunk = indata[:, 0].copy().astype(np.int16)
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            logger.debug("audio_queue full — dropping frame")

    def run(self):
        logger.info(
            "AudioCapture starting (rate=%d, chunk=%d)",
            config.SAMPLE_RATE,
            config.CHUNK_FRAMES,
        )
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=config.CHUNK_FRAMES,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            self._stop.wait()
        logger.info("AudioCapture stopped")
