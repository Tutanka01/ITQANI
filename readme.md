# Itqani

Traduction en temps réel arabe → français pour la khutba du vendredi.

Le système capte le micro, détecte la parole (VAD), transcrit en arabe (Whisper) et traduit en français (LLM via OpenRouter), diffusé en streaming sur une interface web.

---

## Architecture

```
Micro → AudioCapture → VADChunker → Transcriber → Translator → WebSocket → Navigateur
          (sounddevice)  (silero-vad)  (faster-whisper)  (OpenRouter LLM)  (FastAPI)
```

5 threads en parallèle, communicant via des queues Python.

---

## Prérequis

- Python **3.10 – 3.12** (3.13 non supporté par PyTorch)
- GPU NVIDIA avec CUDA (recommandé) ou CPU
- Clé API [OpenRouter](https://openrouter.ai)

---

## Installation

### Windows (recommandé)

**1. Créer le fichier `.env`** à la racine du projet :

```
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
```

**2. Créer un environnement virtuel :**

```powershell
cd C:\Users\<nom>\Documents\ITQANI
python -m venv venv_win
venv_win\Scripts\activate
python -m pip install --upgrade pip
```

**3. Installer PyTorch**

Avec GPU NVIDIA (CUDA 12.4) :
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Sans GPU (CPU) :
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**4. Installer les dépendances :**

```powershell
pip install numpy --only-binary :all:
pip install requests
pip install fastapi==0.115.5 "uvicorn[standard]==0.32.1" httpx==0.28.1 python-dotenv==1.0.1 sounddevice==0.5.1 faster-whisper==1.1.0 --only-binary :all:
```

---

### Linux / WSL

> **Note WSL :** WSL n'a pas accès au micro Windows. Lancer l'application nativement sous Windows.

Sur Linux natif :

```bash
sudo apt-get install -y portaudio19-dev
pip install -r itqani/requirements.txt
```

---

## Premier lancement

Au premier démarrage, le modèle Whisper (`large-v3-turbo`, ~1.5 GB) est téléchargé automatiquement depuis HuggingFace et mis en cache. Les lancements suivants sont immédiats.

```powershell
# Depuis la racine du projet, venv activé
python main.py
```

Puis ouvrir le navigateur sur : **http://localhost:8000**

Arrêt : `Ctrl+C`

---

## Configuration

Tout est dans `itqani/config.py` :

| Paramètre | Défaut | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | Modèle Whisper |
| `WHISPER_DEVICE` | `cuda` | `cuda` ou `cpu` |
| `WHISPER_COMPUTE_TYPE` | `int8_float16` | `int8` pour CPU |
| `OPENROUTER_MODEL` | `google/gemini-2.5-flash-lite` | Modèle LLM |
| `SERVER_PORT` | `8000` | Port du serveur web |

### Utilisation sur CPU uniquement

Modifier `itqani/config.py` :

```python
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
```

---

## Structure du projet

```
ITQANI/
├── main.py                  # Point d'entrée
├── .env                     # Clé API (non versionné)
└── itqani/
    ├── config.py            # Configuration centralisée
    ├── main.py              # Initialisation et signaux
    ├── audio/
    │   ├── capture.py       # Capture micro (sounddevice)
    │   └── vad.py           # Détection de parole (silero-vad)
    ├── transcription/
    │   └── transcriber.py   # Transcription arabe (faster-whisper)
    ├── translation/
    │   └── translator.py    # Traduction LLM (OpenRouter)
    ├── context/
    │   └── manager.py       # Fenêtre de contexte
    ├── server/
    │   └── app.py           # Serveur web (FastAPI + WebSocket)
    ├── pipeline/
    │   └── coordinator.py   # Orchestration des threads
    └── requirements.txt
```
