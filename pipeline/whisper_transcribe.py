import json
import os
import time
from typing import List, Dict, Any

import requests
from openai import OpenAI

from config import ASSEMBLYAI_API_KEY, OPENAI_API_KEY, TRANSCRIBE_PROVIDER
from pipeline.constants import WHISPER_DIR, AUDIO_DIR

from helpers.validators import assert_valid_whisper
from helpers.gpt_cleaner import clean_segments_with_gpt
from helpers.cleaning_utils import is_garbage_arabic
from helpers.vad_filter import filter_segments_by_vad

ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"
DEFAULT_PROVIDER = TRANSCRIBE_PROVIDER or "assemblyai"


def _get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for Whisper transcription")
    return OpenAI(api_key=OPENAI_API_KEY)


def _build_segments_from_words(words: List[Dict[str, Any]], text: str, duration: float | None) -> List[Dict[str, Any]]:
    if not words:
        if duration is None:
            duration = 0.0
        if duration <= 0:
            duration = max(0.0, len(text.split()) * 0.35)
        return [
            {
                "id": 0,
                "start": 0.0,
                "end": round(float(duration), 3),
                "text": text.strip(),
            }
        ]

    segments: List[Dict[str, Any]] = []
    current_words: List[str] = []
    segment_start = float(words[0].get("start", 0.0)) / 1000.0

    for idx, word in enumerate(words):
        start = float(word.get("start", 0.0)) / 1000.0
        end = float(word.get("end", start)) / 1000.0
        token = str(word.get("text", "")).strip()

        if not current_words:
            segment_start = start

        current_words.append(token)
        is_last = idx == len(words) - 1

        gap = 0.0
        if not is_last:
            next_start = float(words[idx + 1].get("start", end)) / 1000.0
            gap = max(0.0, next_start - end)

        should_close = (
            token.endswith((".", "!", "?"))
            or len(current_words) >= 30
            or gap >= 1.5
            or is_last
        )

        if should_close:
            segment_text = " ".join(current_words).strip()
            segments.append(
                {
                    "id": len(segments),
                    "start": round(segment_start, 3),
                    "end": round(max(end, segment_start + 0.01), 3),
                    "text": segment_text,
                }
            )
            current_words = []

    return segments


def _transcribe_with_assemblyai(audio_path: str) -> Dict[str, Any]:
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("ASSEMBLYAI_API_KEY is missing")

    headers = {"authorization": ASSEMBLYAI_API_KEY}

    def _stream_file():
        with open(audio_path, "rb") as f:
            while chunk := f.read(5_242_880):
                yield chunk

    upload_resp = requests.post(
        f"{ASSEMBLYAI_BASE_URL}/upload",
        headers=headers,
        data=_stream_file(),
        timeout=60,
    )
    upload_resp.raise_for_status()
    audio_url = upload_resp.json().get("upload_url")
    if not audio_url:
        raise RuntimeError("Failed to upload audio to AssemblyAI")

    payload = {
        "audio_url": audio_url,
        "speech_model": "universal",
        "language_detection": True,
        "punctuate": True,
        "format_text": True,
    }

    transcript_resp = requests.post(
        f"{ASSEMBLYAI_BASE_URL}/transcript",
        headers=headers,
        json=payload,
        timeout=30,
    )
    transcript_resp.raise_for_status()
    transcript_id = transcript_resp.json().get("id")
    if not transcript_id:
        raise RuntimeError("AssemblyAI did not return transcript id")

    polling_endpoint = f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}"
    print("⏳ Waiting for AssemblyAI transcript...")
    while True:
        poll_resp = requests.get(polling_endpoint, headers=headers, timeout=30)
        poll_resp.raise_for_status()
        data = poll_resp.json()

        status = data.get("status")
        if status == "completed":
            break
        if status == "error":
            raise RuntimeError(f"Transcription failed: {data.get('error')}")
        time.sleep(3)

    words = data.get("words") or []
    segments = _build_segments_from_words(
        words,
        data.get("text", ""),
        data.get("audio_duration"),
    )

    return {
        "text": data.get("text", ""),
        "segments": segments,
        "language": data.get("language_code") or data.get("language"),
    }


def _transcribe_with_openai(audio_path: str) -> Dict[str, Any]:
    client = _get_openai_client()
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )
    return response.model_dump()


def _align_segments_to_first_voice(segments):
    """Drop leading empty segments while tracking measured silence."""
    if not segments:
        return [], 0.0

    first_voiced_idx = None
    for idx, seg in enumerate(segments):
        if seg.get("text", "").strip():
            first_voiced_idx = idx
            break

    # no voiced content left
    if first_voiced_idx is None:
        return segments, 0.0

    # Don't drop any segment that still carries text after VAD refinements.
    if first_voiced_idx > 0:
        leading_has_text = any(seg.get("text", "").strip() for seg in segments[:first_voiced_idx])
        if leading_has_text:
            first_voiced_idx = 0

    # drop leading empty segments
    trimmed = segments[first_voiced_idx:]
    leading_silence = float(trimmed[0].get("start", 0.0))

    if leading_silence > 0:
        print(
            "⏩ Leading silence preserved → "
            f"dropped={first_voiced_idx}, leading={leading_silence:.3f}s"
        )

    return trimmed, leading_silence


def whisper_transcribe(audio_file="input.wav", expected_language=None):
    audio_path = os.path.join(AUDIO_DIR, audio_file)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ ERROR: Audio file not found → {audio_path}")

    print(f"🎧 Transcribing ({DEFAULT_PROVIDER}): {audio_path}")

    if DEFAULT_PROVIDER == "whisper":
        raw_transcript = _transcribe_with_openai(audio_path)
    else:
        raw_transcript = _transcribe_with_assemblyai(audio_path)

    os.makedirs(WHISPER_DIR, exist_ok=True)

    json_path = os.path.join(WHISPER_DIR, "transcript.json")
    txt_path = os.path.join(WHISPER_DIR, "transcript.txt")

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(raw_transcript, jf, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(raw_transcript.get("text", ""))

    assert_valid_whisper(json_path, expected_language)

    segments = raw_transcript.get("segments", [])

    segments = filter_segments_by_vad(segments, audio_path)

    cleaned_segments = []
    for seg in segments:
        text = seg.get("text", "")

        if is_garbage_arabic(text):
            seg["text"] = ""

        cleaned_segments.append(seg)

    aligned_segments, leading_silence = _align_segments_to_first_voice(cleaned_segments)

    raw_transcript["segments"] = aligned_segments
    raw_transcript["text"] = " ".join(
        seg["text"].strip() for seg in aligned_segments if seg.get("text")
    )
    raw_transcript["leading_silence"] = leading_silence

    raw_transcript = clean_segments_with_gpt(raw_transcript)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(raw_transcript, jf, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(raw_transcript.get("text", ""))

    print("🟢 Transcription validation PASSED (after cleaning)")
    return json_path


if __name__ == "__main__":
    whisper_transcribe(expected_language="ar")
