"""
Thread 3 — faster-whisper transcription.

Reads float32 audio chunks from chunk_queue and puts Arabic transcription
strings into transcript_queue.
"""

import logging
import queue
import threading
import time

import numpy as np
from faster_whisper import WhisperModel

from itqani import config

logger = logging.getLogger(__name__)

# Hallucinations connues de Whisper sur les silences ou courts clips
_WHISPER_HALLUCINATIONS = {
    "اشتركوا في القناة",
    "ترجمة نانسي قنقر",
    "شكرا للمشاهدة",
    "شكراً للمشاهدة",
    "للمشاهدة",
    "اشتركوا",
}


def _load_model() -> WhisperModel:
    logger.info(
        "Loading faster-whisper %s on %s (%s)...",
        config.WHISPER_MODEL,
        config.WHISPER_DEVICE,
        config.WHISPER_COMPUTE_TYPE,
    )
    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        cpu_threads=config.WHISPER_CPU_THREADS,
    )
    logger.info("faster-whisper model loaded")
    return model


class Transcriber:
    def __init__(
        self,
        chunk_queue: queue.Queue,
        transcript_queue: queue.Queue,
        stop_event: threading.Event,
    ):
        self._chunk_q = chunk_queue
        self._transcript_q = transcript_queue
        self._stop = stop_event
        self._model = _load_model()

    def _transcribe(self, audio: np.ndarray) -> str:
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language="ar",
            beam_size=config.WHISPER_BEAM_SIZE,
            initial_prompt=config.WHISPER_INITIAL_PROMPT,
            condition_on_previous_text=True,
            vad_filter=False,
            temperature=0,
            word_timestamps=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed = time.perf_counter() - t0
        logger.info(
            "Transcribed (%.2fs, lang=%s prob=%.2f): %s",
            elapsed,
            info.language,
            info.language_probability,
            text[:120],
        )
        return text

    def run(self):
        logger.info("Transcriber started")
        while not self._stop.is_set():
            try:
                audio = self._chunk_q.get(timeout=0.5)
            except queue.Empty:
                continue

            text = self._transcribe(audio)
            if not text:
                logger.debug("Empty transcription, skipping")
                continue
            if text in _WHISPER_HALLUCINATIONS or any(h in text for h in _WHISPER_HALLUCINATIONS):
                logger.debug("Hallucination détectée, ignorée : %s", text)
                continue

            try:
                self._transcript_q.put(text, timeout=2)
            except queue.Full:
                logger.warning("transcript_queue full — dropping transcription")

        logger.info("Transcriber stopped")
