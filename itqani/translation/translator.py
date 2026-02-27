"""
Thread 4 — OpenRouter streaming translation.

Reads Arabic transcriptions from transcript_queue, calls the LLM via
OpenRouter with streaming, and pushes WebSocket messages to the
broadcast callback supplied by the server.

Optimisations vs version initiale
──────────────────────────────────
- Client httpx persistant (connection pooling + keep-alive) : évite le
  handshake TLS à chaque traduction (~100-300ms gagnés par chunk).
- Retry avec backoff exponentiel (3 tentatives) sur les erreurs réseau
  et les codes HTTP 429/503.
- Chaque traduction réussie est journalisée en JSONL dans LOGS_DIR.

WebSocket message types
───────────────────────
{"type": "token",     "content": "<token>"}   — streamed token
{"type": "chunk_end", "content": "<full>"}    — full translated chunk
{"type": "error",     "content": "<msg>"}     — translation error
"""

import asyncio
import datetime
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import httpx

from itqani import config
from itqani.context.manager import ContextManager

logger = logging.getLogger(__name__)


def _log_translation(arabic: str, french: str, latency: float) -> None:
    """Append one translation record to today's JSONL log file."""
    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.date.today().isoformat()
        log_path = config.LOGS_DIR / f"translations_{today}.jsonl"
        record = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "latency_s": round(latency, 3),
            "arabic": arabic,
            "french": french,
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to write translation log: %s", exc)


class Translator:
    def __init__(
        self,
        transcript_queue: queue.Queue,
        broadcast: Callable[[str], None],
        context_manager: ContextManager,
        stop_event: threading.Event,
    ):
        self._transcript_q = transcript_queue
        self._broadcast = broadcast
        self._ctx = context_manager
        self._stop = stop_event

        if not config.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                "Export it or add it to .env before starting."
            )

        self._headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://itqani.local",
            "X-Title": "Itqani Khutba Translator",
        }

    # ------------------------------------------------------------------ #
    # Internal async helpers                                               #
    # ------------------------------------------------------------------ #

    async def _stream_translation(
        self, arabic_text: str, client: httpx.AsyncClient, attempt: int = 0
    ) -> str:
        context_block = self._ctx.format_for_prompt()
        user_content = (
            f"{context_block}"
            f"Passage arabe à traduire :\n{arabic_text}"
        )

        payload = {
            "model": config.OPENROUTER_MODEL,
            "temperature": config.OPENROUTER_TEMPERATURE,
            "stream": True,
            "messages": [
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }

        full_text: list[str] = []
        t0 = time.perf_counter()

        try:
            async with client.stream(
                "POST",
                f"{config.OPENROUTER_BASE_URL}/chat/completions",
                headers=self._headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        full_text.append(token)
                        self._broadcast(
                            json.dumps({"type": "token", "content": token})
                        )

        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException) as exc:
            if attempt < config.TRANSLATION_MAX_RETRIES:
                delay = config.TRANSLATION_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Network error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, config.TRANSLATION_MAX_RETRIES, delay, exc,
                )
                await asyncio.sleep(delay)
                return await self._stream_translation(arabic_text, client, attempt + 1)
            raise

        elapsed = time.perf_counter() - t0
        result = "".join(full_text).strip()
        logger.info(
            "Translation complete (%.2fs, %d tokens): %s",
            elapsed,
            len(full_text),
            result[:120],
        )
        return result

    async def _process_one(self, arabic_text: str, client: httpx.AsyncClient) -> None:
        for attempt in range(config.TRANSLATION_MAX_RETRIES + 1):
            try:
                t0 = time.perf_counter()
                french_text = await self._stream_translation(arabic_text, client)
                latency = time.perf_counter() - t0
                self._broadcast(
                    json.dumps({"type": "chunk_end", "content": french_text})
                )
                self._ctx.add(french_text)
                _log_translation(arabic_text, french_text, latency)
                return

            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code in (429, 503) and attempt < config.TRANSLATION_MAX_RETRIES:
                    delay = config.TRANSLATION_RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "HTTP %d (attempt %d/%d), retrying in %.1fs",
                        code, attempt + 1, config.TRANSLATION_MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("OpenRouter HTTP error: %s", exc)
                # Toujours fermer le chunk actif côté frontend pour éviter le texte mélangé
                self._broadcast(json.dumps({"type": "chunk_end", "content": ""}))
                self._broadcast(
                    json.dumps({"type": "error", "content": f"Erreur API ({code})"})
                )
                return

            except Exception as exc:
                logger.error("Translation error: %s", exc, exc_info=True)
                self._broadcast(json.dumps({"type": "chunk_end", "content": ""}))
                self._broadcast(
                    json.dumps({"type": "error", "content": "Erreur de traduction"})
                )
                return

    # ------------------------------------------------------------------ #
    # Sync bridge (runs in its own thread, owns an asyncio event loop)    #
    # ------------------------------------------------------------------ #

    def run(self):
        logger.info("Translator started")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_loop())
        finally:
            loop.close()
        logger.info("Translator stopped")

    def _drain_queue(self) -> list[str]:
        """Récupère tous les chunks disponibles (max TRANSLATION_BATCH_SIZE)."""
        items = []
        while len(items) < config.TRANSLATION_BATCH_SIZE:
            try:
                items.append(self._transcript_q.get_nowait())
            except queue.Empty:
                break
        return items

    async def _run_loop(self):
        # Persistent client — connexion TCP/TLS réutilisée entre les traductions
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
        ) as client:
            while not self._stop.is_set():
                # Attendre le premier chunk
                try:
                    first = self._transcript_q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue

                # Drainer les chunks déjà en attente pour les regrouper
                batch = [first] + self._drain_queue()
                arabic_text = " ".join(batch)

                # Ignorer les fragments trop courts (bruit, hallucination résiduelle)
                if len(arabic_text) < config.TRANSLATION_MIN_ARABIC_CHARS:
                    logger.debug("Fragment trop court ignoré (%d chars): %s", len(arabic_text), arabic_text)
                    continue

                if len(batch) > 1:
                    logger.info("Batch de %d chunks envoyé au LLM", len(batch))

                await self._process_one(arabic_text, client)
