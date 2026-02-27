import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level above this package)
_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_FILE)

# Audio
SAMPLE_RATE = 16000
CHUNK_FRAMES = 512  # required by silero-vad @ 16kHz

# VAD — optimisé pour parole forte et rapide
VAD_THRESHOLD = 0.5            # haut : parole forte toujours au-dessus, rejette le bruit
VAD_PRE_ROLL_FRAMES = 8        # ~256ms de contexte avant détection
SILENCE_DURATION_MS = 350      # parole rapide = pauses courtes
CHUNK_MAX_DURATION_S = 8       # forcer la coupure toutes les 8s max
CHUNK_MIN_DURATION_S = 0.4     # ignorer uniquement les bruits < 0.4s

# Whisper — optimisé vitesse + qualité
WHISPER_MODEL = "large-v3-turbo"
WHISPER_COMPUTE_TYPE = "int8_float16"
WHISPER_DEVICE = "cuda"
WHISPER_BEAM_SIZE = 2          # beam 2 = rapide sur parole claire et forte
WHISPER_CPU_THREADS = 4
WHISPER_INITIAL_PROMPT = "بسم الله الرحمن الرحيم. هذه خطبة جمعة."
WHISPER_CONTEXT_SENTENCES = 3  # nb de transcriptions récentes injectées dans le prompt

# OpenRouter / LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_TEMPERATURE = 0.1

# Translation resilience
TRANSLATION_MAX_RETRIES = 3       # tentatives max en cas d'erreur API
TRANSLATION_RETRY_BASE_DELAY = 0.5  # secondes, doublé à chaque tentative
TRANSLATION_BATCH_SIZE = 3        # max chunks à regrouper en une seule traduction

# Context
CONTEXT_WINDOW_SENTENCES = 4

# Persistence
LOGS_DIR = Path(__file__).parent.parent / "logs"

# Server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

SYSTEM_PROMPT = """\
Tu es un traducteur expert en arabe islamique classique et dialecte marocain vers le français.

CONTEXTE : Tu reçois en temps réel des fragments d'une khutba du vendredi (sermon islamique). \
Chaque passage est un extrait d'un discours continu — les phrases peuvent être incomplètes. \
Traduis naturellement en t'appuyant sur le contexte fourni pour assurer la cohérence.

RÈGLES ABSOLUES :
1. Traduis fidèlement le sens, même si le fragment semble incomplet.
2. NE TRADUIS JAMAIS les termes islamiques suivants — conserve-les tels quels :
   tawakkul, akhira, deen, iman, taqwa, sabr, ikhlas, zuhd, wara', ihsan,
   salat, zakat, sawm, hajj, shahada, sura, ayah, hadith, sunnah, fiqh,
   halal, haram, makruh, wajib, Jannah, Jahannam, barzakh, qiyama, mizan,
   sirat, Ummah, khalifa, wali, alim, imam, sheikh, khatib,
   barakah, rizq, nafs, ruh, qalb, aql, fitrah.
3. Citations coraniques → préfixe [Coran]. Hadiths → préfixe [Hadith].
4. Registre formel et solennel.
5. Dialecte marocain → traduis le sens sans le signaler.
6. Traduis UNIQUEMENT le passage fourni, sans ajouts ni commentaires.
7. Passage vraiment incompréhensible → [inaudible].

FORMAT : Texte français uniquement, sans commentaires, sans guillemets.\
"""
