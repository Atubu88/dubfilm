import json
import os
from openai import OpenAI
from pipeline.constants import WHISPER_DIR, TRANSLATION_DIR
from config import OPENAI_API_KEY
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

    ⚠️ Сегментация строго сохраняется — модель получает JSON и обязана
       вернуть JSON того же размера. Так мы исключаем потери сегментов,
       которые раньше возникали при парсинге пронумерованных строк.
    """

    whisper_path = os.path.join(WHISPER_DIR, whisper_json)

    with open(whisper_path, "r", encoding="utf-8") as f:
        whisper_data = json.load(f)

    segments = whisper_data["segments"]

    print(f"📖 Loaded {len(segments)} segments for translation")

    # 🧠 Отправляем JSON, чтобы исключить двусмысленности при парсинге
    payload = {
        "target_lang": target_lang,
        "segments": [
            {
                "id": seg["id"],
                "text": seg["text"]
            }
            for seg in segments
        ]
    }

    system_prompt = (
        "You are a professional translator. Translate the provided segments "
        f"into {target_lang} and keep the order EXACTLY the same. "
        "Respond ONLY with JSON that matches the schema: "
        '{"segments": [{"id": <int>, "dst": "translated"}]}'
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]
    )

    try:
        translated_payload = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"❌ GPT returned invalid JSON: {exc}") from exc

    translated_lines = translated_payload.get("segments")

    if not isinstance(translated_lines, list):
        raise RuntimeError("❌ GPT JSON has no 'segments' list")

    if len(translated_lines) != len(segments):
        raise RuntimeError(
            f"❌ GPT LOST SEGMENTS ({len(translated_lines)} vs {len(segments)})"
        )

    # 🏗 СТРОИМ НОВЫЙ JSON
    translated_segments = []

    for seg, translated in zip(segments, translated_lines):
        if seg["id"] != translated.get("id"):
            raise RuntimeError(
                f"❌ GPT misaligned IDs: expected {seg['id']} got {translated.get('id')}"
            )

        dst_text = translated.get("dst", "").strip()
        if not dst_text:
            raise RuntimeError(f"❌ Empty translation for segment {seg['id']}")

        translated_segments.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "src": seg["text"],
            "dst": dst_text
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