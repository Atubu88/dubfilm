import json
import os
from openai import OpenAI
from config import WHISPER_DIR, TRANSLATION_DIR, OPENAI_API_KEY
from helpers.validators import assert_valid_translation

client = OpenAI(api_key=OPENAI_API_KEY)


def translate_segments(
        whisper_json="transcript.json",
        target_lang="en"  # например: "ru", "en", "fr"
):
    """
    🔹 Загружает сегменты Whisper
    🔹 Отправляет GPT запрос на перевод
    🔹 Сохраняет новый JSON с 'src' + 'dst'
    🔹 Проверяет корректность
    """

    whisper_path = os.path.join(WHISPER_DIR, whisper_json)

    with open(whisper_path, "r", encoding="utf-8") as f:
        whisper_data = json.load(f)

    segments = whisper_data["segments"]

    print(f"📖 Loaded {len(segments)} segments for translation")

    # 🧠 Формируем список строк с номерами
    numbered_list = "\n".join(
        f"{i+1}. {seg['text']}"
        for i, seg in enumerate(segments)
    )

    system_prompt = f"""
You are a professional translator.
Translate Arabic speech into {target_lang}.
⚠️ RULES:
- KEEP SEGMENT ORDER
- DO NOT MERGE segments
- DO NOT ADD segments
- The output MUST be a numbered list 1:N
Example:
1. text...
2. text...
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": numbered_list}
        ]
    )

    translated_text = response.choices[0].message.content.strip()

    # 🧩 ПАРСИМ НАЗАД В МАССИВ
    translated_lines = [
        line.split(". ", 1)[1]   # Удаляем "1. "
        for line in translated_text.split("\n")
        if ". " in line
    ]

    if len(translated_lines) != len(segments):
        raise RuntimeError(f"❌ GPT LOST SEGMENTS ({len(translated_lines)} vs {len(segments)})")

    # 🏗 СТРОИМ НОВЫЙ JSON
    translated_segments = []

    for seg, dst in zip(segments, translated_lines):
        translated_segments.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "src": seg["text"],
            "dst": dst.strip()
        })

    os.makedirs(TRANSLATION_DIR, exist_ok=True)

    json_out = os.path.join(TRANSLATION_DIR, "translated.json")
    txt_out = os.path.join(TRANSLATION_DIR, "translated.txt")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(t["dst"] for t in translated_segments))

    print(f"💾 SAVED → {json_out}")

    # 🛡 ПРОВЕРЯЕМ
    assert_valid_translation(json_out)

    print("🟢 Translation OK")

    return json_out


if __name__ == "__main__":
    out = translate_segments(
        whisper_json="transcript.json",
        target_lang="ru"   # ⚠️ ТУТ ставь язык перевода
    )
    print("✅ Translation saved to:", out)
