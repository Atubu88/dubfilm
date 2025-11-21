import json
import os
from openai import OpenAI
from pipeline.constants import WHISPER_DIR, TRANSLATION_DIR
from config import OPENAI_API_KEY
from helpers.validators import assert_valid_translation

client = OpenAI(api_key=OPENAI_API_KEY)


def translate_segments(
        whisper_json="transcript.json",
        target_lang="en"
):
    """
    🔹 Загружает сегменты Whisper
    🔹 Отправляет GPT JSON, состоящий ТОЛЬКО из непустых сегментов
    🔹 Вставляет пустые сегменты обратно
    🔹 Сохраняет перевод, проверяет структуру
    """

    whisper_path = os.path.join(WHISPER_DIR, whisper_json)

    with open(whisper_path, "r", encoding="utf-8") as f:
        whisper_data = json.load(f)

    segments = whisper_data["segments"]
    leading_silence = float(whisper_data.get("leading_silence", 0.0) or 0.0)

    print(f"📖 Loaded {len(segments)} segments for translation")

    # -----------------------------------
    # 1. Разделяем сегменты
    # -----------------------------------
    empty_segments = [s for s in segments if not s["text"].strip()]
    non_empty_segments = [s for s in segments if s["text"].strip()]

    print(f"🌑 Empty segments: {len(empty_segments)}")
    print(f"🟩 To translate: {len(non_empty_segments)}")

    # -----------------------------------
    # 2. GPT получает только НЕпустые сегменты
    # -----------------------------------
    payload = {
        "target_lang": target_lang,
        "segments": [
            {"id": seg["id"], "text": seg["text"]}
            for seg in non_empty_segments
        ]
    }

    system_prompt = (
        "You are a professional translator. Translate EVERY provided segment into "
        f"{target_lang} WITHOUT adding, merging, or inventing any text.\n"
        "RULES:\n"
        "1) Keep ids and order EXACTLY as provided.\n"
        "2) One source segment → one translated segment.\n"
        "3) Never introduce greetings, fillers, or any new sentences.\n"
        "4) If the source text looks like noise or non-speech → return an empty string for dst.\n"
        "5) Respond ONLY with JSON: {\"segments\":[{\"id\":<int>,\"dst\":\"translated text\"}]}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]
    )

    # -----------------------------------
    # 3. Разбираем ответ GPT
    # -----------------------------------
    try:
        translated_payload = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"❌ GPT returned invalid JSON: {exc}") from exc

    translated_non_empty = translated_payload.get("segments")

    if not isinstance(translated_non_empty, list):
        raise RuntimeError("❌ GPT JSON missing 'segments' list")

    if len(translated_non_empty) != len(non_empty_segments):
        raise RuntimeError(
            f"❌ GPT LOST SEGMENTS ({len(translated_non_empty)} vs {len(non_empty_segments)})"
        )

    # словарь {id → dst}
    translated_dict = {t["id"]: t["dst"] for t in translated_non_empty}

    # -----------------------------------
    # 4. Собираем финальный список сегментов
    # -----------------------------------
    translated_segments = []

    for seg in segments:

        # пустой сегмент — перевод НЕ требуется
        if not seg["text"].strip():
            translated_segments.append({
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "src": seg["text"],
                "dst": ""
            })
            continue

        # непустой сегмент — перевод обязателен
        dst = translated_dict.get(seg["id"])
        if dst is None:
            raise RuntimeError(f"❌ Missing translation for id {seg['id']}")

        translated_segments.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "src": seg["text"],
            "dst": dst.strip()
        })

    # -----------------------------------
    # 5. Сохраняем файлы
    # -----------------------------------
    os.makedirs(TRANSLATION_DIR, exist_ok=True)

    json_out = os.path.join(TRANSLATION_DIR, "translated.json")
    txt_out = os.path.join(TRANSLATION_DIR, "translated.txt")

    payload = {
        "leading_silence": leading_silence,
        "segments": translated_segments,
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(t["dst"] for t in translated_segments))

    print(f"💾 SAVED → {json_out}")

    # -----------------------------------
    # 6. Валидация
    # -----------------------------------
    assert_valid_translation(json_out, max_ratio=3.0)

    print("🟢 Translation OK")

    return json_out


if __name__ == "__main__":
    out = translate_segments(
        whisper_json="transcript.json",
        target_lang="ru"
    )
    print("✅ Translation saved to:", out)
