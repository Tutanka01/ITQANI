import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level above this package)
_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_FILE)

# Audio
SAMPLE_RATE = 16000
CHUNK_FRAMES = 512  # required by silero-vad @ 16kHz

# VAD — optimisé pour obtenir des phrases complètes
VAD_THRESHOLD = 0.5            # haut : parole forte toujours au-dessus, rejette le bruit
VAD_PRE_ROLL_FRAMES = 8        # ~256ms de contexte avant détection
SILENCE_DURATION_MS = 800      # attendre 800ms de silence = fin de phrase naturelle
CHUNK_MAX_DURATION_S = 10      # forcer la coupure à 10s max
CHUNK_MIN_DURATION_S = 1.0     # ignorer les bruits et micro-fragments < 1s

# Whisper — optimisé vitesse + qualité
WHISPER_MODEL = "large-v3-turbo"
WHISPER_COMPUTE_TYPE = "int8_float16"
WHISPER_DEVICE = "cuda"
WHISPER_BEAM_SIZE = 2          # beam 2 = rapide sur parole claire et forte
WHISPER_CPU_THREADS = 4
WHISPER_INITIAL_PROMPT = "خطبة جمعة باللغة العربية الفصحى والدارجة المغربية."
WHISPER_CONTEXT_SENTENCES = 3  # nb de transcriptions récentes injectées dans le prompt

# OpenRouter / LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_TEMPERATURE = 0.1

# Translation resilience
TRANSLATION_MAX_RETRIES = 3
TRANSLATION_RETRY_BASE_DELAY = 0.5
TRANSLATION_BATCH_SIZE = 4         # regroupe jusqu'à 4 chunks avant traduction
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
