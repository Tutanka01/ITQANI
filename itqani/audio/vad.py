"""
Thread 2 — Silero-VAD + chunk assembler.

Reads int16 frames from audio_queue, applies Silero-VAD, and puts
finalized float32 numpy audio chunks into chunk_queue.

Finalisation rules
──────────────────
1. Speech detected (prob ≥ VAD_THRESHOLD) → accumulate frames
2. SILENCE_DURATION_MS of continuous silence after speech → flush
3. Accumulated duration ≥ CHUNK_MAX_DURATION_S → forced flush
4. Chunks shorter than CHUNK_MIN_DURATION_S → discarded (noise)
"""

import logging
import queue
import threading
from typing import List

import numpy as np
import torch

from itqani import config

logger = logging.getLogger(__name__)

_SILENCE_FRAMES = int(
    config.SILENCE_DURATION_MS / 1000 * config.SAMPLE_RATE / config.CHUNK_FRAMES
)
_MAX_FRAMES = int(
    config.CHUNK_MAX_DURATION_S * config.SAMPLE_RATE / config.CHUNK_FRAMES
)
_MIN_FRAMES = int(
    config.CHUNK_MIN_DURATION_S * config.SAMPLE_RATE / config.CHUNK_FRAMES
)


def _load_silero_vad():
    logger.info("Loading Silero-VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    logger.info("Silero-VAD loaded")
    return model


class VADChunker:
    def __init__(
        self,
        audio_queue: queue.Queue,
        chunk_queue: queue.Queue,
        stop_event: threading.Event,
    ):
        self._audio_q = audio_queue
        self._chunk_q = chunk_queue
        self._stop = stop_event
        self._model = _load_silero_vad()
        self._model.eval()

    def _vad_prob(self, frame_int16: np.ndarray) -> float:
        """Return speech probability for a single 512-sample int16 frame."""
        audio_f32 = frame_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_f32).unsqueeze(0)  # (1, 512)
        with torch.no_grad():
            prob = self._model(tensor, config.SAMPLE_RATE).item()
        return prob

    def _flush(self, buffer: List[np.ndarray]) -> None:
        if len(buffer) < _MIN_FRAMES:
            logger.debug("VAD: chunk too short (%d frames), discarded", len(buffer))
            return
        audio = np.concatenate(buffer).astype(np.float32) / 32768.0
        duration = len(audio) / config.SAMPLE_RATE
        logger.info("VAD: flushing chunk (%.2fs)", duration)
        try:
            self._chunk_q.put(audio, timeout=2)
        except queue.Full:
            logger.warning("chunk_queue full — dropping audio chunk")

    def run(self):
        logger.info("VADChunker started")
        buffer: List[np.ndarray] = []
        in_speech = False
        silence_count = 0

        while not self._stop.is_set():
            try:
                frame = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                # If stop requested and we have accumulated audio, flush it
                if self._stop.is_set() and buffer:
                    self._flush(buffer)
                continue

            prob = self._vad_prob(frame)
            is_speech = prob >= config.VAD_THRESHOLD

            if is_speech:
                buffer.append(frame)
                in_speech = True
                silence_count = 0
            elif in_speech:
                # Silence after speech — keep accumulating until threshold
                buffer.append(frame)
                silence_count += 1
                if silence_count >= _SILENCE_FRAMES:
                    self._flush(buffer)
                    buffer = []
                    in_speech = False
                    silence_count = 0
            # else: pre-speech silence — discard

            # Forced flush on max duration
            if len(buffer) >= _MAX_FRAMES:
                logger.info("VAD: max duration reached, forced flush")
                self._flush(buffer)
                buffer = []
                in_speech = False
                silence_count = 0

        # Flush remaining on shutdown
        if buffer:
            self._flush(buffer)
        logger.info("VADChunker stopped")
