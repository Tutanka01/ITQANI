"""
Sliding window of the last N translated French sentences.

Used to inject context into the LLM prompt for semantic coherence
across chunk boundaries.
"""

import threading
from collections import deque
from typing import List

from itqani import config


class ContextManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._sentences: deque[str] = deque(maxlen=config.CONTEXT_WINDOW_SENTENCES)

    def add(self, french_text: str) -> None:
        """Record a newly translated French sentence."""
        with self._lock:
            self._sentences.append(french_text.strip())

    def get_context(self) -> List[str]:
        """Return current context window as an ordered list."""
        with self._lock:
            return list(self._sentences)

    def format_for_prompt(self) -> str:
        """
        Return a formatted block ready to inject into the user message.
        Empty string if no context yet.
        """
        sentences = self.get_context()
        if not sentences:
            return ""
        joined = "\n".join(f"- {s}" for s in sentences)
        return f"Contexte (traductions précédentes) :\n{joined}\n\n"
