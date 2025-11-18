import json
import os
from openai import OpenAI
from config import WHISPER_DIR, AUDIO_DIR, OPENAI_API_KEY
from helpers.validators import assert_valid_whisper   # ← ДОБАВИЛИ

client = OpenAI(api_key=OPENAI_API_KEY)

def whisper_transcribe(audio_file="input.wav", expected_language=None):
    audio_path = os.path.join(AUDIO_DIR, audio_file)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ ERROR: Audio file not found → {audio_path}")

    print(f"🎧 Transcribing: {audio_path}")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )

    os.makedirs(WHISPER_DIR, exist_ok=True)

    json_path = os.path.join(WHISPER_DIR, "transcript.json")
    txt_path = os.path.join(WHISPER_DIR, "transcript.txt")

    # 📄 Сохраняем JSON — теперь правильно
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(response.model_dump(), jf, ensure_ascii=False, indent=2)

    # 📝 Сохраняем raw-текст
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(response.text)

    print(f"📄 JSON saved → {json_path}")
    print(f"📝 TXT saved  → {txt_path}")

    # 🛡 Whisper validation
    assert_valid_whisper(json_path, expected_language)

    print("🟢 Whisper validation PASSED")
    return json_path


if __name__ == "__main__":
    whisper_transcribe(expected_language="ar")
