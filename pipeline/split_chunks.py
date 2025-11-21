import json
import os

from pipeline.constants import TRANSLATION_DIR, CHUNKS_DIR
from helpers.validators import assert_valid_chunks

MAX_CHARS = 260          # безопасный лимит для TTS
MAX_DURATION = 15        # сек. защитный порог по длительности (≈ 2 предложения)
TRANSLATED_JSON = f"{TRANSLATION_DIR}/translated.json"


def compute_global_offset(segments, leading_silence: float = 0.0):
    if not segments:
        return 0.0

    first_segment_start = float(segments[0].get("start", 0.0))
    offset = max(leading_silence - first_segment_start, 0.0)

    if offset != 0:
        print(
            "⏱️ Restoring preserved leading silence → "
            f"baseline={leading_silence:.3f}s, offset={offset:+.3f}s"
        )

    return offset


def apply_offset(value: float, offset: float) -> float:
    return max(round(value + offset, 3), 0.0)

def load_segments():
    with open(TRANSLATED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    leading_silence = 0.0
    if isinstance(data, dict):
        leading_silence = float(data.get("leading_silence", 0.0) or 0.0)
        data = data.get("segments", [])

    return data, leading_silence


def save_chunk(idx, start, end, text, offset):
    chunk = {
        "start": apply_offset(start, offset),
        "end": apply_offset(end, offset),
        "text": text.strip()
    }

    path_json = os.path.join(CHUNKS_DIR, f"chunk_{idx:03d}.json")
    path_txt  = os.path.join(CHUNKS_DIR, f"chunk_{idx:03d}.txt")

    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)

    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(chunk["text"])


def split_into_chunks():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    # очищаем старые чанки
    for f in os.listdir(CHUNKS_DIR):
        os.remove(os.path.join(CHUNKS_DIR, f))

    segments, leading_silence = load_segments()
    if not segments:
        print("❌ translated.json is empty — no chunks to create")
        return

    offset = compute_global_offset(segments, leading_silence)

    chunks = []
    current_text = ""
    current_start = segments[0]["start"]
    current_end = segments[0]["end"]

    for seg in segments:
        text = seg["dst"]
        duration = seg["end"] - current_start

        if (
            len(current_text) + len(text) > MAX_CHARS
            or duration > MAX_DURATION
        ):
            chunks.append((current_start, current_end, current_text))
            current_text = ""
            current_start = seg["start"]

        current_text += text + " "
        current_end = seg["end"]

    chunks.append((current_start, current_end, current_text))

    for i, (start, end, text) in enumerate(chunks, 1):
        save_chunk(i, start, end, text, offset)

    print(f"📦 Created {len(chunks)} chunks")

    assert_valid_chunks(CHUNKS_DIR)
    print("🟢 All chunks valid")


if __name__ == "__main__":
    split_into_chunks()
