import json
import os
from openai import OpenAI
from config import OPENAI_API_KEY
from pipeline.constants import WHISPER_DIR, AUDIO_DIR
from helpers.validators import assert_valid_whisper
from helpers.gpt_cleaner import clean_segments_with_gpt
from helpers.cleaning_utils import is_garbage_arabic   # ← ДОБАВИЛИ
from helpers.vad_filter import filter_segments_by_vad

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

    whisper_json = response.model_dump()

    # ---------------------------------------------------------
    # 🧹 1) VAD-фильтр + удаление арабского мусора БЕЗ GPT
    # ---------------------------------------------------------
    segments = whisper_json.get("segments", [])

    # 1a) VAD: убираем сегменты без реальной речи
    segments = filter_segments_by_vad(segments, audio_path)

    # 1b) Доп. чистка арабского мусора по символам
    cleaned_segments = []
    for seg in segments:
        text = seg.get("text", "")

        if is_garbage_arabic(text):
            seg["text"] = ""   # 🔥 заменяем мусор на пустую строку

        cleaned_segments.append(seg)

    whisper_json["segments"] = cleaned_segments

    # пересобираем общий text
    whisper_json["text"] = " ".join(
        seg["text"].strip() for seg in cleaned_segments if seg.get("text")
    )

    # ---------------------------------------------------------
    # 2) Сохраняем whisper raw JSON + TXT
    # ---------------------------------------------------------
    os.makedirs(WHISPER_DIR, exist_ok=True)

    json_path = os.path.join(WHISPER_DIR, "transcript.json")
    txt_path = os.path.join(WHISPER_DIR, "transcript.txt")

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(whisper_json, jf, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(whisper_json.get("text", ""))

    print(f"📄 JSON saved → {json_path}")
    print(f"📝 TXT saved  → {txt_path}")

    # ---------------------------------------------------------
    # 3) Проверяем Whisper JSON
    # ---------------------------------------------------------
    assert_valid_whisper(json_path, expected_language)

    # ---------------------------------------------------------
    # 4) GPT-cleaner (теперь получает только более чистый текст)
    # ---------------------------------------------------------
    whisper_json = clean_segments_with_gpt(whisper_json)

    # пересохраняем после GPT-чистки
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(whisper_json, jf, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(whisper_json.get("text", ""))

    print("🟢 Whisper validation PASSED")
    return json_path


if __name__ == "__main__":
    whisper_transcribe(expected_language="ar")
