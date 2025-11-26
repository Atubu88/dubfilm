import json
import os
import sys
from typing import List, Dict

import srt
from openai import OpenAI

from config import OPENAI_API_KEY
from helpers.validators import assert_valid_translation
from pipeline.constants import TRANSLATION_DIR, WHISPER_DIR

client = OpenAI(api_key=OPENAI_API_KEY)


def load_srt_segments(srt_filename: str = "subtitles.srt") -> List[Dict]:
    srt_path = os.path.join(WHISPER_DIR, srt_filename)
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"❌ SRT file not found → {srt_path}")

    with open(srt_path, "r", encoding="utf-8") as f:
        parsed = list(srt.parse(f.read()))

    segments = []
    for idx, entry in enumerate(parsed):
        text = entry.content.replace("\n", " ").strip()
        if not text:
            continue

        segments.append({
            "id": idx,
            "start": entry.start.total_seconds(),
            "end": entry.end.total_seconds(),
            "text": text,
        })

    if not segments:
        raise RuntimeError("❌ Parsed SRT contains no text segments")

    print(f"📖 Loaded {len(segments)} SRT segments for translation")
    return segments


def translate_segments(
        srt_filename="subtitles.srt",
        target_lang="en"  # например: "ru", "en", "fr"
):
    """
    🔹 Загружает сегменты из SRT (AssemblyAI)
    🔹 Отправляет GPT запрос на перевод
    🔹 Сохраняет новый JSON с 'src' + 'dst'
    🔹 Проверяет корректность
    """

    segments = load_srt_segments(srt_filename)

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

        dst_text = translated.get("dst", "")

        # Если GPT вернул пустую строку, подставляем исходный текст,
        # а если и он пустой — ставим заглушку
        if not dst_text or not dst_text.strip():
            print(f"⚠️  Empty translation for segment {seg['id']} — using source text")
            fallback = seg["text"].strip() if seg.get("text", "").strip() else "[UNTRANSLATED]"
            dst_text = fallback
        else:
            dst_text = dst_text.strip()

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
    lang = sys.argv[1] if len(sys.argv) > 1 else "ru"
    out = translate_segments(
        srt_filename="subtitles.srt",
        target_lang=lang
    )
    print("✅ Translation saved to:", out)
