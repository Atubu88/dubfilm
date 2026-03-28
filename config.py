import json
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
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip()
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy").strip()
OPENAI_TTS_FORMAT = os.getenv("OPENAI_TTS_FORMAT", "mp3").strip()

# Optional ElevenLabs TTS provider
DUB_TTS_PROVIDER = os.getenv("DUB_TTS_PROVIDER", "openai").strip().lower()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2").strip()
ELEVENLABS_DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_DEFAULT_VOICE_ID", "").strip()
_OPENAI_TTS_VOICE_ENV = OPENAI_TTS_VOICE
ENABLE_DUB_FLOW = os.getenv("ENABLE_DUB_FLOW", "1").strip() in {"1", "true", "yes", "on"}

DUB_PROFILE = os.getenv("DUB_PROFILE", "cartoon").strip().lower()


def _load_profile(profile_name: str) -> dict:
    base = Path(__file__).resolve().parent / "profiles"
    profile_path = base / f"{profile_name}.json"
    if not profile_path.exists():
        return {}
    try:
        return json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


_PROFILE = _load_profile(DUB_PROFILE)


def _profile_or_env(name: str, env_default: str) -> float:
    if name in _PROFILE:
        try:
            return float(_PROFILE[name])
        except Exception:
            pass
    return float(os.getenv(name, env_default))


def _profile_or_str(name: str, env_default: str) -> str:
    if name in _PROFILE:
        try:
            return str(_PROFILE[name]).strip()
        except Exception:
            pass
    return os.getenv(name, env_default).strip()


OPENAI_TTS_VOICE = _profile_or_str("OPENAI_TTS_VOICE", _OPENAI_TTS_VOICE_ENV)
DUB_TTS_STYLE = _profile_or_str("DUB_TTS_STYLE", "")
DUB_MULTI_VOICE = _profile_or_str("DUB_MULTI_VOICE", "0").lower() in {"1", "true", "yes", "on"}
DUB_MULTI_VOICE_LIST = [
    v.strip() for v in _profile_or_str("DUB_MULTI_VOICE_LIST", "").split(",") if v.strip()
]

# Optional explicit speaker mapping, e.g. "A:alloy,B:onyx,C:verse"
DUB_MULTI_VOICE_MAP_RAW = _profile_or_str("DUB_MULTI_VOICE_MAP", "")
DUB_MULTI_VOICE_MAP: dict[str, str] = {}
if DUB_MULTI_VOICE_MAP_RAW:
    for part in DUB_MULTI_VOICE_MAP_RAW.split(","):
        if ":" not in part:
            continue
        spk, v = part.split(":", 1)
        spk = spk.strip()
        v = v.strip()
        if spk and v:
            DUB_MULTI_VOICE_MAP[spk] = v

DUB_ORIGINAL_AUDIO_VOLUME = _profile_or_env("DUB_ORIGINAL_AUDIO_VOLUME", "0.20")
DUB_TTS_AUDIO_VOLUME = _profile_or_env("DUB_TTS_AUDIO_VOLUME", "1.00")
# TTS pacing controls (best-practice guardrails for natural voice)
DUB_TTS_MIN_SPEED = _profile_or_env("DUB_TTS_MIN_SPEED", "0.90")
DUB_TTS_MAX_SPEED = _profile_or_env("DUB_TTS_MAX_SPEED", "1.15")
DUB_TARGET_CHARS_PER_SEC = _profile_or_env("DUB_TARGET_CHARS_PER_SEC", "12.0")
DUB_MIN_SEGMENT_DURATION = _profile_or_env("DUB_MIN_SEGMENT_DURATION", "1.2")

# =========================
# ASSEMBLYAI
# =========================
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
ASSEMBLYAI_SPEECH_MODEL = os.getenv("ASSEMBLYAI_SPEECH_MODEL", "universal-2").strip()

# =========================
# PYANNOTE (optional diarization)
# =========================
PYANNOTE_AUTH_TOKEN = os.getenv("PYANNOTE_AUTH_TOKEN", "").strip()
PYANNOTE_MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1").strip()
PYANNOTE_MIN_SPEAKERS = os.getenv("PYANNOTE_MIN_SPEAKERS", "").strip()
PYANNOTE_MAX_SPEAKERS = os.getenv("PYANNOTE_MAX_SPEAKERS", "").strip()

# pyannoteAI cloud diarization (recommended fallback when local pyannote is unstable)
PYANNOTEAI_API_KEY = os.getenv("PYANNOTEAI_API_KEY", "").strip()
PYANNOTEAI_MODEL = os.getenv("PYANNOTEAI_MODEL", "precision-2").strip()

# =========================
# TRANSCRIPTION PROVIDER SWITCH
# =========================
TRANSCRIBE_PROVIDER = os.getenv("TRANSCRIBE_PROVIDER", "whisper").strip().lower()

if TRANSCRIBE_PROVIDER not in ("whisper", "assemblyai", "hybrid"):
    print(f"⚠️ Unknown TRANSCRIBE_PROVIDER: {TRANSCRIBE_PROVIDER}. Fallback to whisper.")
    TRANSCRIBE_PROVIDER = "whisper"

# =========================
# DIRECTORIES
# =========================
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "tmp"
TEMP_DIR.mkdir(exist_ok=True)

# =========================
# GLOSSARY / ISLAMIC RULES
# =========================
GLOSSARY_ENABLED = os.getenv("GLOSSARY_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
GLOSSARY_PATH = os.getenv("GLOSSARY_PATH", str(BASE_DIR / "glossary_islamic_ru.json")).strip()
GLOSSARY_SKIP_QURAN_AYAHS = os.getenv("GLOSSARY_SKIP_QURAN_AYAHS", "1").strip().lower() in {"1", "true", "yes", "on"}
QURAN_AYAH_TTS_MODE = os.getenv("QURAN_AYAH_TTS_MODE", "original_tts").strip().lower()  # original_tts|mute
ISLAMIC_TRANSLATION_MODE = os.getenv("ISLAMIC_TRANSLATION_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}

# ✅ ЛОГ ПРИ СТАРТЕ
print("🔧 CONFIG LOADED:")
print(f"   🧠 GPT Model: {OPENAI_CHAT_MODEL}")
print(f"   🎙 Transcribe Provider: {TRANSCRIBE_PROVIDER}")
print(f"   🗣 TTS Model: {OPENAI_TTS_MODEL} ({OPENAI_TTS_VOICE})")
print(f"   🔊 TTS Provider: {DUB_TTS_PROVIDER}")
print(f"   🎛 Dub profile: {DUB_PROFILE}")
if DUB_TTS_STYLE:
    print("   🎭 TTS style: custom profile instructions enabled")
if DUB_MULTI_VOICE:
    print(f"   🧩 Multi-voice: ON ({', '.join(DUB_MULTI_VOICE_LIST) if DUB_MULTI_VOICE_LIST else 'no voices configured'})")
    if DUB_MULTI_VOICE_MAP:
        print(f"   🗺 Speaker map: {DUB_MULTI_VOICE_MAP}")
if GLOSSARY_ENABLED:
    print(f"   📚 Glossary: ON ({GLOSSARY_PATH})")
print(f"   ☪️ Ayah TTS mode: {QURAN_AYAH_TTS_MODE}")
if ISLAMIC_TRANSLATION_MODE:
    print("   🕌 Islamic translation mode: ON")

if TRANSCRIBE_PROVIDER == "whisper":
    print(f"   🎧 Whisper Model: {OPENAI_WHISPER_MODEL}")

if TRANSCRIBE_PROVIDER == "assemblyai":
    if ASSEMBLYAI_API_KEY:
        print("   ✅ AssemblyAI key loaded")
        print(f"   🧾 AssemblyAI speech model: {ASSEMBLYAI_SPEECH_MODEL}")
    else:
        print("   ❌ AssemblyAI key is MISSING!")

if PYANNOTE_AUTH_TOKEN:
    print(f"   🧠 Pyannote diarization token: loaded ({PYANNOTE_MODEL})")
if PYANNOTEAI_API_KEY:
    print(f"   ☁️ pyannoteAI key: loaded ({PYANNOTEAI_MODEL})")
if ELEVENLABS_API_KEY:
    print(f"   ✅ ElevenLabs key loaded ({ELEVENLABS_MODEL_ID})")

if not OPENAI_API_KEY:
    print("   ❌ OPENAI_API_KEY is MISSING! Translation & summary will NOT work.")
else:
    print("   ✅ OPENAI_API_KEY loaded")

# =========================
# TRANSLATION OPTIONS
# =========================
DEFAULT_TRANSLATION_CHOICES = [
    "English",
    "Arabic",
    "Uzbek",
    "Russian",
]

# Subtitle sync mode: off | manual | auto
# - off: no extra sync correction
# - manual: apply SUBTITLE_EXTRA_DELAY_SECONDS
# - auto: detect first speech in audio and auto-shift subtitles
SUBTITLE_SYNC_MODE = os.getenv("SUBTITLE_SYNC_MODE", "auto").strip().lower()

# Manual global subtitle delay (seconds). Use positive value if subtitles appear too early.
SUBTITLE_EXTRA_DELAY_SECONDS = float(os.getenv("SUBTITLE_EXTRA_DELAY_SECONDS", "0"))

# Auto-sync safety clamp (seconds): max absolute correction applied in auto mode.
SUBTITLE_AUTO_MAX_SHIFT_SECONDS = float(os.getenv("SUBTITLE_AUTO_MAX_SHIFT_SECONDS", "8.0"))

# Optional post-sync with ffsubsync (recommended for strongest alignment across diverse clips)
SUBTITLE_ENABLE_FFSUBSYNC = os.getenv("SUBTITLE_ENABLE_FFSUBSYNC", "1").strip() in {"1", "true", "yes", "on"}
FFSUBSYNC_VAD = os.getenv("FFSUBSYNC_VAD", "auditok").strip().lower()  # auditok|webrtc
FFSUBSYNC_MAX_OFFSET_SECONDS = float(os.getenv("FFSUBSYNC_MAX_OFFSET_SECONDS", "20"))
FFSUBSYNC_USE_GSS = os.getenv("FFSUBSYNC_USE_GSS", "1").strip() in {"1", "true", "yes", "on"}
FFSUBSYNC_NO_FIX_FRAMERATE = os.getenv("FFSUBSYNC_NO_FIX_FRAMERATE", "0").strip() in {"1", "true", "yes", "on"}
FFSUBSYNC_MAX_ACCEPTED_OFFSET_SECONDS = float(os.getenv("FFSUBSYNC_MAX_ACCEPTED_OFFSET_SECONDS", "5"))
