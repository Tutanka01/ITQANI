"""
Thread 2 — Silero-VAD + chunk assembler (hybrid temporal + dip-aligned).

Reads int16 frames from audio_queue, applies Silero-VAD, and puts
finalized float32 numpy audio chunks into chunk_queue.

Chunking algorithm
──────────────────
Optimised for continuous loud speech (khutba) where long silences never come.

1. Speech detected (prob ≥ VAD_THRESHOLD) → accumulate frames
2. When duration ≥ CHUNK_TARGET_S → ALWAYS cut:
   - If a dip (low prob) exists in last frames → cut at dip (carry-over)
   - Otherwise → cut everything anyway (never wait)
3. CHUNK_MAX_S is a safety net only
4. True silence (≥ CHUNK_SILENCE_S) → normal flush
5. Chunks shorter than CHUNK_MIN_S → discarded (noise)

Key rule: at TARGET, we ALWAYS flush. No waiting.
"""

import logging
import queue
import threading
from typing import List, Tuple

import numpy as np
import torch

from itqani import config

logger = logging.getLogger(__name__)

# Pre-computed frame counts
_FRAME_DURATION_S = config.CHUNK_FRAMES / config.SAMPLE_RATE  # ~0.032s per frame
_TARGET_FRAMES = int(config.CHUNK_TARGET_S / _FRAME_DURATION_S)
_MIN_FRAMES = int(config.CHUNK_MIN_S / _FRAME_DURATION_S)
_MAX_FRAMES = int(config.CHUNK_MAX_S / _FRAME_DURATION_S)
_SILENCE_FRAMES = int(config.CHUNK_SILENCE_S / _FRAME_DURATION_S)


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
        rms = np.sqrt(np.mean(audio ** 2))

        if rms < config.VAD_MIN_RMS:
            logger.debug("VAD: chunk trop silencieux (rms=%.4f), ignoré", rms)
            return

        # Normalisation par peak — préserve la dynamique, amène le signal
        # dans la plage que Whisper attend (proche de [-1, 1])
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.9 / peak)

        duration = len(audio) / config.SAMPLE_RATE
        logger.info("VAD: flushing chunk (%.2fs, rms=%.4f, peak=%.4f)", duration, rms, peak)
        try:
            self._chunk_q.put(audio, timeout=2)
        except queue.Full:
            logger.warning("chunk_queue full — dropping audio chunk")

    def _find_best_dip(self, probs: List[float]) -> Tuple[int, float]:
        """Find the frame with the lowest prob in the last DIP_WINDOW frames."""
        window = min(config.CHUNK_DIP_WINDOW, len(probs))
        search = probs[-window:]
        min_idx = int(np.argmin(search))
        min_prob = search[min_idx]
        abs_idx = len(probs) - window + min_idx
        return abs_idx, min_prob

    def run(self):
        logger.info(
            "VADChunker started (target=%.1fs, min=%.1fs, max=%.1fs, silence=%.1fs)",
            config.CHUNK_TARGET_S, config.CHUNK_MIN_S,
            config.CHUNK_MAX_S, config.CHUNK_SILENCE_S,
        )
        buffer: List[np.ndarray] = []
        probs: List[float] = []
        pre_roll: List[np.ndarray] = []
        in_speech = False
        silence_count = 0

        while not self._stop.is_set():
            try:
                frame = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set() and buffer:
                    self._flush(buffer)
                continue

            prob = self._vad_prob(frame)
            is_speech = prob >= config.VAD_THRESHOLD

            if is_speech:
                if not in_speech:
                    # Start of speech: include pre-roll
                    buffer.extend(pre_roll)
                    probs.extend([0.0] * len(pre_roll))
                    pre_roll = []
                buffer.append(frame)
                probs.append(prob)
                in_speech = True
                silence_count = 0
            elif in_speech:
                # Silence after speech — still accumulate
                buffer.append(frame)
                probs.append(prob)
                silence_count += 1

                # True silence → normal flush
                if silence_count >= _SILENCE_FRAMES:
                    logger.debug("VAD: silence (%.0fms), flushing", silence_count * _FRAME_DURATION_S * 1000)
                    pre_roll = list(buffer[-config.VAD_PRE_ROLL_FRAMES:])
                    self._flush(buffer)
                    buffer = []
                    probs = []
                    in_speech = False
                    silence_count = 0
                    continue
            else:
                # Silence before speech — circular pre-roll
                pre_roll.append(frame)
                if len(pre_roll) > config.VAD_PRE_ROLL_FRAMES:
                    pre_roll.pop(0)

            # --- Temporal flush: ALWAYS cut at TARGET, never wait ---
            n_frames = len(buffer)

            if n_frames >= _TARGET_FRAMES:
                dip_idx, dip_prob = self._find_best_dip(probs)

                if dip_prob < config.CHUNK_DIP_SOFT_THRESHOLD:
                    # Good dip found → cut there, carry-over the rest
                    cut_at = dip_idx + 1
                    to_flush = buffer[:cut_at]
                    carry_over = buffer[cut_at:]
                    carry_probs = probs[cut_at:]

                    logger.info(
                        "VAD: dip-cut at %.2fs (prob=%.2f), carry %.2fs",
                        len(to_flush) * _FRAME_DURATION_S, dip_prob,
                        len(carry_over) * _FRAME_DURATION_S,
                    )
                    self._flush(to_flush)
                    buffer = carry_over
                    probs = carry_probs
                else:
                    # No good dip → cut everything anyway, don't wait!
                    logger.info(
                        "VAD: target cut at %.2fs (no dip, best=%.2f)",
                        n_frames * _FRAME_DURATION_S, dip_prob,
                    )
                    self._flush(buffer)
                    buffer = []
                    probs = []

                # in_speech stays True — continuous speech keeps flowing
                silence_count = 0

        # Flush remaining on shutdown
        if buffer:
            self._flush(buffer)
        logger.info("VADChunker stopped")
