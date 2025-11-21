import json
from typing import Any, Dict, Iterable, List

from openai import OpenAI

from config import OPENAI_API_KEY

MODEL_NAME = "gpt-4o-mini"
CHUNK_CHAR_LIMIT = 6000
MAX_RETRIES = 2

SYSTEM_PROMPT = (
    "You are a cleaner for Whisper Arabic transcripts. Follow strictly: \n"
    "1) Keep JSON structure exactly the same: same ids, start, end, order, and segment count.\n"
    "2) Only modify segment['text']; do not change start/end/id or add/remove fields.\n"
    "3) Remove filler/noise like 'مممم', 'اييي', 'مم مم', 'اوووووو', 'اييي..', 'мммм', 'оооо', 'эййй', sighs, grunts, or non-speech sounds by setting text to an empty string ''.\n"
    "4) Fix misrecognized Arabic words to proper, clear Arabic without inventing new phrases.\n"
    "5) If a segment has no real speech or only noise, set text to ''.\n"
    "Respond ONLY with JSON in the format: {\"segments\": [{\"id\": <int>, \"start\": <float>, \"end\": <float>, \"text\": \"...\"}]}"
)

client = OpenAI(api_key=OPENAI_API_KEY)


def _chunk_segments(segments: List[Dict[str, Any]], limit: int) -> Iterable[List[Dict[str, Any]]]:
    current: List[Dict[str, Any]] = []
    current_len = 0

    for seg in segments:
        seg_len = len(str(seg.get("text", "")))
        if current and current_len + seg_len > limit:
            yield current
            current = []
            current_len = 0
        current.append(seg)
        current_len += seg_len

    if current:
        yield current


def _call_gpt(payload: Dict[str, Any]) -> Dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"⚠️  GPT attempt {attempt} failed: {exc}")

    raise RuntimeError(f"❌ GPT failed after {MAX_RETRIES} attempts: {last_error}")


def _process_chunk(chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = {
        "segments": [
            {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg.get("text", ""),
            }
            for seg in chunk
        ]
    }

    result = _call_gpt(payload)

    cleaned_segments = result.get("segments")
    if not isinstance(cleaned_segments, list):
        raise RuntimeError("❌ GPT JSON missing 'segments' list")

    if len(cleaned_segments) != len(chunk):
        raise RuntimeError(
            f"❌ GPT lost segments ({len(cleaned_segments)} vs {len(chunk)})"
        )

    for original, cleaned in zip(chunk, cleaned_segments):
        if original.get("id") != cleaned.get("id"):
            raise RuntimeError(
                f"❌ GPT misaligned IDs: expected {original.get('id')} got {cleaned.get('id')}"
            )
        if original.get("start") != cleaned.get("start") or original.get("end") != cleaned.get("end"):
            raise RuntimeError(
                f"❌ GPT changed timestamps for segment {original.get('id')}"
            )
        if "text" not in cleaned:
            raise RuntimeError(
                f"❌ GPT segment missing text for id {original.get('id')}"
            )
        cleaned_text = cleaned["text"]
        if not isinstance(cleaned_text, str):
            raise RuntimeError(
                f"❌ GPT returned non-string text for id {original.get('id')}"
            )
        cleaned["text"] = cleaned_text.strip()

    return cleaned_segments


def _rebuild_text(segments: List[Dict[str, Any]]) -> str:
    return " ".join(seg["text"].strip() for seg in segments if seg.get("text"))


def clean_segments_with_gpt(json_data: Dict[str, Any]) -> Dict[str, Any]:
    segments = json_data.get("segments") or []

    if not segments:
        print("⚠️  No segments to clean")
        return json_data

    total = len(segments)
    cleaned_count = 0
    empty_count = 0
    corrected_count = 0
    cleaned_segments: List[Dict[str, Any]] = []

    for chunk in _chunk_segments(segments, CHUNK_CHAR_LIMIT):
        processed = _process_chunk(chunk)

        for original, cleaned in zip(chunk, processed):
            if cleaned.get("text", "") != original.get("text", ""):
                cleaned_count += 1
                if cleaned.get("text") == "":
                    empty_count += 1
                else:
                    corrected_count += 1
        cleaned_segments.extend(processed)

    json_data["segments"] = cleaned_segments
    json_data["text"] = _rebuild_text(cleaned_segments)

    print(
        "🧹 GPT clean stats → "
        f"total={total}, changed={cleaned_count}, empty={empty_count}, corrected={corrected_count}"
    )

    return json_data
