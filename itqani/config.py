import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level above this package)
_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_FILE)

_logger = logging.getLogger(__name__)

# Audio
SAMPLE_RATE = 16000
CHUNK_FRAMES = 512  # required by silero-vad @ 16kHz

# VAD — chunking hybride temporel + dip-aligné
VAD_THRESHOLD = 0.35             # abaissé : meilleure détection de la parole
VAD_PRE_ROLL_FRAMES = 5          # ~160ms
VAD_MIN_RMS = 0.005              # abaissé : ne pas rejeter les micros faibles
CHUNK_TARGET_S = 4.0             # cible : flush ~4 secondes
CHUNK_MIN_S = 2.0                # jamais flush avant 2s
CHUNK_MAX_S = 7.0                # forcer la coupure à 7s max
CHUNK_DIP_WINDOW = 10            # fenêtre glissante pour détecter les dips
CHUNK_DIP_SOFT_THRESHOLD = 0.35  # prob en dessous = bon moment pour couper
CHUNK_SILENCE_S = 0.8            # vrai silence pour flush (800ms — les respirations d'imam sont plus courtes)


# ── Whisper — auto-détection du device ────────────────────────────
def _detect_device():
    """Détecte le meilleur device disponible pour faster-whisper (CTranslate2)."""
    try:
        import torch
        if torch.cuda.is_available():
            _logger.info("CUDA disponible — Whisper utilisera le GPU")
            return "cuda", "float16"
    except ImportError:
        pass
    _logger.info("Pas de CUDA — Whisper utilisera le CPU")
    return "cpu", "int8"


WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = _detect_device()

WHISPER_MODEL = "large-v3-turbo"
WHISPER_BEAM_SIZE = 1            # greedy = plus rapide, suffisant pour parole claire
WHISPER_CPU_THREADS = 4
WHISPER_INITIAL_PROMPT = "خطبة جمعة باللغة العربية الفصحى والدارجة المغربية."
WHISPER_CONTEXT_SENTENCES = 2    # réduit le risque de cascade

# OpenRouter / LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_TEMPERATURE = 0.1

# Translation resilience
TRANSLATION_MAX_RETRIES = 3
TRANSLATION_RETRY_BASE_DELAY = 0.5
TRANSLATION_BATCH_SIZE = 2         # max 2 chunks, on envoie plus vite
TRANSLATION_MIN_ARABIC_CHARS = 15  # ignorer les fragments trop courts

# Context
CONTEXT_WINDOW_SENTENCES = 4

# Persistence
LOGS_DIR = Path(__file__).parent.parent / "logs"

# Server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

SYSTEM_PROMPT = """\
Tu es interprète simultané d'une khutba du vendredi, arabe → français.

MISSION : Produire un français immédiatement compréhensible par un francophone \
qui ne comprend pas l'arabe. Chaque passage doit former une pensée complète et fluide.

RÈGLES :
1. Produis des phrases françaises naturelles et complètes. \
   Si le passage arabe est un fragment, reformule-le en phrase grammaticalement correcte.
2. Assure la continuité avec les traductions précédentes (contexte fourni). \
   Le résultat doit sembler être un discours continu, pas des phrases isolées.
3. NE TRADUIS JAMAIS ces termes islamiques — conserve-les tels quels : \
   tawakkul, akhira, deen, iman, taqwa, sabr, ikhlas, ihsan, \
   salat, zakat, sawm, hajj, shahada, ayah, hadith, sunnah, \
   halal, haram, Jannah, Jahannam, qiyama, Ummah, imam, sheikh, khatib, \
   barakah, rizq, nafs, ruh, fitrah.
4. [Coran] pour les versets, [Hadith] pour les hadiths.
5. Registre solennel et formel.
6. Dialecte marocain → traduis le sens, ne le signale pas.
7. Incompréhensible → [inaudible].

FORMAT : Français uniquement. Pas de guillemets, pas de commentaires.\
"""
