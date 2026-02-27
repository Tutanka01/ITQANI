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
        self._frame_count = 0
        self._peak_max = 0

    def _callback(self, indata: np.ndarray, frames: int, time, status):
        if status:
            logger.warning("sounddevice status: %s", status)
        # indata shape: (frames, channels) — take first channel, copy to avoid aliasing
        chunk = indata[:, 0].copy().astype(np.int16)

        # Monitoring : log les niveaux audio toutes les ~5s (~156 frames à 512 samples/16kHz)
        peak = int(np.max(np.abs(chunk)))
        self._peak_max = max(self._peak_max, peak)
        self._frame_count += 1
        if self._frame_count % 156 == 0:
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            logger.info(
                "🎙 Audio: peak=%d/%d (%.0f%%), rms=%.0f — %s",
                peak, 32768, peak / 32768 * 100, rms,
                "OK" if peak > 500 else "⚠ SIGNAL TRÈS FAIBLE — vérifier le micro",
            )
            self._peak_max = 0

        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            logger.debug("audio_queue full — dropping frame")

    def run(self):
        # Lister les devices pour le debug
        try:
            default_input = sd.query_devices(kind='input')
            logger.info("Micro par défaut : %s", default_input['name'])
        except Exception as exc:
            logger.warning("Impossible de lister le micro : %s", exc)

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
