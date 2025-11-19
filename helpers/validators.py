import json
import os

class WhisperValidationError(Exception):
    pass


def assert_valid_whisper(json_path: str, expected_language=None):
    """
    Проверяет корректность Whisper JSON, созданного через response.model_dump()
    """

    if not os.path.exists(json_path):
        raise WhisperValidationError(f"❌ JSON NOT FOUND → {json_path}")

    # ---- Load JSON ----
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise WhisperValidationError(f"❌ INVALID JSON → {e}")

    # ---- MUST contain final text ----
    if "text" not in data or len(data["text"].strip()) < 3:
        raise WhisperValidationError("❌ Whisper JSON contains EMPTY text")

    # ---- SEGMENTS ----
    segments = data.get("segments")
    if not segments:
        raise WhisperValidationError("❌ Whisper JSON has NO SEGMENTS")

    for i, seg in enumerate(segments):

        # MUST have required fields
        for key in ("start", "end", "text"):
            if key not in seg:
                raise WhisperValidationError(f"❌ Segment #{i} missing key '{key}'")

        # Timestamps must be valid
        if seg["end"] <= seg["start"]:
            raise WhisperValidationError(f"❌ Segment #{i} invalid timestamps")

        # Text must not be empty
        if not seg["text"].strip():
            raise WhisperValidationError(f"❌ Segment #{i} has EMPTY text")

    # ---- LANGUAGE CHECK ----
    detected_lang = data.get("language")

    # Accept both "ar" and "arabic"
    if expected_language:
        valid_forms = {expected_language}

        if expected_language == "ar":
            valid_forms.add("arabic")

        if detected_lang not in valid_forms:
            raise WhisperValidationError(
                f"❌ LANGUAGE MISMATCH → detected={detected_lang} expected={valid_forms}"
            )

    print(f"✅ Whisper JSON OK → {len(segments)} segments, language={detected_lang}")
    return True


class TranslationValidationError(Exception):
    pass


def assert_valid_translation(json_path: str, min_ratio=0.5):
    """
    Проверяет:
    - совпадение количества сегментов
    - наличие dst
    - длина перевода не слишком короткая
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise TranslationValidationError("❌ NOT A SEGMENT LIST")

    for i, seg in enumerate(data):

        for key in ("id", "start", "end", "src", "dst"):
            if key not in seg:
                raise TranslationValidationError(f"❌ Segment #{i} missing '{key}'")

        if not seg["dst"].strip():
            raise TranslationValidationError(f"❌ Segment #{i} has EMPTY TRANSLATION")

        ratio = len(seg["dst"]) / max(1, len(seg["src"]))

        if ratio < min_ratio:
            raise TranslationValidationError(
                f"❌ Segment #{i} TOO SHORT → ratio={ratio:.2f}"
            )

    print(f"✅ Translation VALID → {len(data)} segments OK")
    return True



class ChunkValidationError(Exception):
    pass


def assert_valid_chunks(chunks_dir: str):
    files = sorted(f for f in os.listdir(chunks_dir) if f.endswith(".json"))

    if not files:
        raise ChunkValidationError("❌ No chunks saved")

    for f in files:
        path = os.path.join(chunks_dir, f)
        data = json.load(open(path, encoding="utf-8"))

        text = data["text"].strip()
        duration = data["end"] - data["start"]

        if not text:
            raise ChunkValidationError(f"❌ EMPTY TEXT → {f}")

        if duration <= 0:
            raise ChunkValidationError(f"❌ INVALID TIME RANGE → {f}")

        # ❗ Только предупреждаем, не ломаем пайплайн
        if duration > 20:
            print(f"⚠️  WARNING: {f} = {duration:.1f}s (>20s)")

        if len(text) > 350:
            raise ChunkValidationError(f"❌ TOO MANY CHARACTERS ({len(text)}) → {f}")


import wave

class TTSValidationError(Exception):
    pass


def assert_valid_tts_chunk(wav_path: str, text: str):
    """
    Проверяет:
    - файл существует
    - не пустой
    - >= 0.3s
    - <= 20s (безопасный лимит OpenAI)
    """

    if not os.path.exists(wav_path):
        raise TTSValidationError(f"❌ TTS FILE MISSING → {wav_path}")

    if os.path.getsize(wav_path) < 2000:
        raise TTSValidationError(f"❌ TTS FILE TOO SMALL → {wav_path}")

    try:
        with wave.open(wav_path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            duration = frames / float(rate)
    except Exception as e:
        raise TTSValidationError(f"❌ INVALID WAV → {e}")

    if duration < 0.3:
        raise TTSValidationError(f"❌ TOO SHORT TTS ({duration:.2f}s) → {wav_path}")

    if duration > 20:
        raise TTSValidationError(f"❌ TOO LONG TTS ({duration:.2f}s) — text too big")

    print(f"   🔈 TTS duration: {duration:.2f}s (OK)")


class FinalAudioValidationError(Exception):
    pass


def assert_valid_final_audio(wav_path):
    if not os.path.exists(wav_path):
        raise FinalAudioValidationError(f"❌ final audio NOT FOUND → {wav_path}")

    try:
        import wave

        with wave.open(wav_path, "rb") as w:
            nframes = w.getnframes()
            framerate = w.getframerate()
            duration = nframes / float(framerate)

            if duration < 0.5:
                raise FinalAudioValidationError("❌ Final audio too short")

            if duration > 10 * 60:
                print("⚠️ WARNING: Very long audio (>10 min)")

        print(f"🔎 Final WAV OK → duration {duration:.2f}s")

    except Exception as e:
        raise FinalAudioValidationError(f"❌ Invalid final WAV → {e}")
