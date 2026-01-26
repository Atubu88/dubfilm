import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =========================
# TELEGRAM
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TELEGRAM_VIDEO_UPLOAD_TIMEOUT = float(os.getenv("TELEGRAM_VIDEO_UPLOAD_TIMEOUT", "300"))

# =========================
# OPENAI (GPT + WHISPER)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
OPENAI_WHISPER_MODEL = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1").strip()

# =========================
# ASSEMBLYAI
# =========================
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()

# =========================
# TRANSCRIPTION PROVIDER SWITCH
# =========================
TRANSCRIBE_PROVIDER = os.getenv("TRANSCRIBE_PROVIDER", "whisper").strip().lower()

if TRANSCRIBE_PROVIDER not in ("whisper", "assemblyai"):
    print(f"⚠️ Unknown TRANSCRIBE_PROVIDER: {TRANSCRIBE_PROVIDER}. Fallback to whisper.")
    TRANSCRIBE_PROVIDER = "whisper"

# ✅ ЛОГ ПРИ СТАРТЕ
print("🔧 CONFIG LOADED:")
print(f"   🧠 GPT Model: {OPENAI_CHAT_MODEL}")
print(f"   🎙 Transcribe Provider: {TRANSCRIBE_PROVIDER}")

if TRANSCRIBE_PROVIDER == "whisper":
    print(f"   🎧 Whisper Model: {OPENAI_WHISPER_MODEL}")

if TRANSCRIBE_PROVIDER == "assemblyai":
    if ASSEMBLYAI_API_KEY:
        print("   ✅ AssemblyAI key loaded")
    else:
        print("   ❌ AssemblyAI key is MISSING!")

if not OPENAI_API_KEY:
    print("   ❌ OPENAI_API_KEY is MISSING! Translation & summary will NOT work.")
else:
    print("   ✅ OPENAI_API_KEY loaded")

# =========================
# DIRECTORIES
# =========================
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "tmp"
TEMP_DIR.mkdir(exist_ok=True)

# =========================
# TRANSLATION OPTIONS
# =========================
DEFAULT_TRANSLATION_CHOICES = [
    "English",
    "Arabic",
    "Uzbek",
    "Russian",
]
