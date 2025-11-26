import json
import os
from typing import List, Dict

from helpers.validators import assert_valid_chunks
from pipeline.constants import TRANSLATION_DIR, CHUNKS_DIR

MAX_CHARS = 260          # безопасный лимит для TTS
TRANSLATED_JSON = f"{TRANSLATION_DIR}/translated.json"


def load_segments() -> List[Dict]:
    with open(TRANSLATED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chunk(idx, start, end, text, ids=None):
    chunk = {
        "start": round(start, 3),
        "end": round(end, 3),
        "text": text.strip(),
    }
    if ids:
        chunk["segment_ids"] = ids

    path_json = os.path.join(CHUNKS_DIR, f"chunk_{idx:03d}.json")
    path_txt = os.path.join(CHUNKS_DIR, f"chunk_{idx:03d}.txt")

    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)

    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(chunk["text"])


def split_into_chunks():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    # очищаем старые чанки
    for f in os.listdir(CHUNKS_DIR):
        os.remove(os.path.join(CHUNKS_DIR, f))

    segments = load_segments()
    if not segments:
        raise RuntimeError("❌ No translated segments to split")

    chunks = []
    current_text = ""
    current_start = None
    current_end = None
    current_ids: List[int] = []

    for seg in segments:
        seg_text = (seg.get("dst") or "").strip()
        if not seg_text:
            continue

        proposed_text = (current_text + " " + seg_text).strip() if current_text else seg_text

        if current_text and len(proposed_text) > MAX_CHARS:
            chunks.append((current_start, current_end, current_text, current_ids.copy()))
            current_text = seg_text
            current_start = seg["start"]
            current_end = seg["end"]
            current_ids = [seg.get("id")]
            continue

        if not current_text:
            current_start = seg["start"]
            current_ids = [seg.get("id")]
        else:
            current_ids.append(seg.get("id"))

        current_text = proposed_text
        current_end = seg["end"]

    if current_text:
        chunks.append((current_start, current_end, current_text, current_ids))

    for i, (start, end, text, ids) in enumerate(chunks, 1):
        save_chunk(i, start, end, text, ids)

    print(f"📦 Created {len(chunks)} chunks")

    assert_valid_chunks(CHUNKS_DIR)
    print("🟢 All chunks valid")


if __name__ == "__main__":
    split_into_chunks()
