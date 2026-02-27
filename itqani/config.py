import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level above this package)
_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_FILE)

# Audio
SAMPLE_RATE = 16000
CHUNK_FRAMES = 512  # required by silero-vad @ 16kHz

# VAD
VAD_THRESHOLD = 0.5
SILENCE_DURATION_MS = 700    # attendre 700ms de silence avant de couper
CHUNK_MAX_DURATION_S = 15    # forcer la coupure à 15s max
CHUNK_MIN_DURATION_S = 2.0   # ignorer les chunks < 2s (bruit, hallucinations)

# Whisper
WHISPER_MODEL = "large-v3-turbo"
WHISPER_COMPUTE_TYPE = "int8_float16"
WHISPER_DEVICE = "cuda"
WHISPER_BEAM_SIZE = 3          # was 5 — sufficient pour l'arabe, ~40% plus rapide
WHISPER_CPU_THREADS = 4        # threads CPU pour les ops CTranslate2 hors GPU
WHISPER_INITIAL_PROMPT = "بسم الله الرحمن الرحيم. هذه خطبة جمعة."

# OpenRouter / LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-2.5-flash-lite"
OPENROUTER_TEMPERATURE = 0.1

# Translation resilience
TRANSLATION_MAX_RETRIES = 3       # tentatives max en cas d'erreur API
TRANSLATION_RETRY_BASE_DELAY = 0.5  # secondes, doublé à chaque tentative

# Context
CONTEXT_WINDOW_SENTENCES = 4

# Persistence
LOGS_DIR = Path(__file__).parent.parent / "logs"

# Server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

SYSTEM_PROMPT = """\
Tu es un traducteur expert en arabe islamique classique et dialecte marocain vers le français.

RÈGLES ABSOLUES :
1. Traduis fidèlement le sens du discours islamique (khutba du vendredi).
2. NE TRADUIS JAMAIS les termes islamiques suivants — conserve-les tels quels :
   tawakkul, akhira, deen, iman, taqwa, sabr, ikhlas, zuhd, wara', ihsan,
   salat, zakat, sawm, hajj, shahada, sura, ayah, hadith, sunnah, fiqh,
   halal, haram, makruh, wajib, Jannah, Jahannam, barzakh, qiyama, mizan,
   sirat, Ummah, khalifa, wali, alim, imam, sheikh, khatib,
   barakah, rizq, nafs, ruh, qalb, aql, fitrah.
3. Citations coraniques → préfixe [Coran]. Hadiths → préfixe [Hadith].
4. Registre formel et solennel.
5. Dialecte marocain → traduis le sens sans le signaler.
6. Traduis UNIQUEMENT le passage fourni, sans ajouts.
7. Passage incompréhensible → [inaudible].

FORMAT : Texte français uniquement, sans commentaires.\
"""
