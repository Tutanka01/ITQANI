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


def _is_looping(text: str) -> bool:
    """Détecte une hallucination en boucle (même séquence répétée > 3 fois)."""
    words = text.split()
    if len(words) < 9:
        return False
    for n in range(2, 5):  # bigrammes, trigrammes, quadrigrammes
        for i in range(len(words) - n):
            seq = " ".join(words[i:i + n])
            if text.count(seq) > 3:
                return True
    return False


# Hallucinations connues de Whisper sur les silences ou courts clips
_WHISPER_HALLUCINATIONS = {
    "اشتركوا في القناة",
    "ترجمة نانسي قنقر",
    "شكرا للمشاهدة",
    "شكراً للمشاهدة",
    "للمشاهدة",
    "اشتركوا",
    "هذه خطبة جمعة",
    "هذا خطبة الجمعة",
    "بسم الله الرحمن الرحيم",
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
        # Rolling buffer des dernières transcriptions arabes
        # Injecté dans initial_prompt pour que Whisper connaisse le contexte du discours
        self._recent: list[str] = []
        self._last_lang_prob: float = 1.0

    def _build_prompt(self) -> str:
        """Construit un prompt dynamique à partir des dernières transcriptions."""
        base = config.WHISPER_INITIAL_PROMPT
        if not self._recent:
            return base
        context = " ".join(self._recent[-config.WHISPER_CONTEXT_SENTENCES:])
        # Limiter à 224 tokens max (limite Whisper) — ~900 caractères arabes
        combined = f"{base} {context}"
        return combined[-900:] if len(combined) > 900 else combined

    def _transcribe(self, audio: np.ndarray) -> str:
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language="ar",
            beam_size=config.WHISPER_BEAM_SIZE,
            initial_prompt=self._build_prompt(),
            condition_on_previous_text=False,
            vad_filter=False,
            temperature=0,
            word_timestamps=False,
            no_speech_threshold=0.5,
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
        self._last_lang_prob = info.language_probability
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
                logger.debug("Hallucination connue ignorée : %s", text)
                continue

            if _is_looping(text):
                logger.warning("Hallucination en boucle détectée, contexte réinitialisé : %s", text[:80])
                self._recent.clear()  # casser la cascade
                continue

            # Mémoriser pour enrichir le prochain prompt (seulement si confiance élevée)
            if self._last_lang_prob >= 0.8:
                self._recent.append(text)
                if len(self._recent) > config.WHISPER_CONTEXT_SENTENCES:
                    self._recent.pop(0)
            else:
                logger.debug("Low confidence (%.2f), not adding to context: %s", self._last_lang_prob, text[:60])

            try:
                self._transcript_q.put(text, timeout=2)
            except queue.Full:
                logger.warning("transcript_queue full — dropping transcription")

        logger.info("Transcriber stopped")
