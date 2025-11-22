import json
import os
import time
import requests
from openai import OpenAI
from config import OPENAI_API_KEY, ASSEMBLYAI_API_KEY, TRANSCRIBE_PROVIDER
from pipeline.constants import WHISPER_DIR, AUDIO_DIR
from helpers.validators import assert_valid_whisper
from helpers.gpt_cleaner import clean_segments_with_gpt

from pydub import AudioSegment
from pydub.silence import split_on_silence

client = OpenAI(api_key=OPENAI_API_KEY)
ASSEMBLYAI_API_URL = "https://api.assemblyai.com/v2"


# ============================================================
# ⭐ 1. ПРОФЕССИОНАЛЬНАЯ СЕГМЕНТАЦИЯ — ПО ПАУЗАМ В АУДИО ⭐
# ============================================================

def segment_by_silence(audio_path: str, full_text: str):
    """
    Делит аудио по паузам и распределяет текст по сегментам.
    Возвращает сегменты полностью в стиле Whisper CLI.
    """

    audio = AudioSegment.from_wav(audio_path)

    chunks = split_on_silence(
        audio,
        min_silence_len=350,       # пауза ≥ 0.35 сек = граница фразы
        silence_thresh=-40,        # порог тишины
        keep_silence=120           # сохраняем небольшой хвост тишины
    )

    if not chunks:
        # fallback: один сегмент
        return [{
            "id": 0,
            "start": 0.0,
            "end": len(audio) / 1000,
            "text": full_text.strip()
        }]

    # Определяем start/end каждого сегмента
    segments = []
    cursor = 0
    for chunk in chunks:
        start = cursor
        end = cursor + len(chunk)
        segments.append((start, end))
        cursor = end

    # Текст делим по количеству сегментов
    words = full_text.split()
    chunk_size = max(1, len(words) // len(segments))

    whisper_segments = []
    word_idx = 0

    for i, (start, end) in enumerate(segments):
        chunk_words = words[word_idx: word_idx + chunk_size]
        word_idx += chunk_size

        text_part = " ".join(chunk_words).strip()
        if not text_part:
            continue

        whisper_segments.append({
            "id": len(whisper_segments),
            "start": start / 1000,
            "end": end / 1000,
            "text": text_part
        })

    # Добавляем остаток текста в последний сегмент
    if word_idx < len(words):
        whisper_segments[-1]["text"] += " " + " ".join(words[word_idx:])

    return whisper_segments


# ============================================================
# ⭐ 2. AssemblyAI — получаем текст, но сегменты уже НЕ используем ⭐
# ============================================================

def _transcribe_with_assemblyai(audio_path, expected_language=None):
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("❌ ASSEMBLYAI_API_KEY is missing")

    headers = {"authorization": ASSEMBLYAI_API_KEY}

    def _read_file(path):
        with open(path, "rb") as f:
            while chunk := f.read(5_242_880):
                yield chunk

    print("⬆️  Uploading audio to AssemblyAI...")
    upload_resp = requests.post(
        f"{ASSEMBLYAI_API_URL}/upload",
        headers=headers,
        data=_read_file(audio_path)
    )
    upload_resp.raise_for_status()
    audio_url = upload_resp.json()["upload_url"]

    payload = {
        "audio_url": audio_url,
        "speech_model": "universal",
        "punctuate": True,
    }
    if expected_language:
        payload["language_code"] = expected_language

    print("🛰  Requesting AssemblyAI transcription...")
    resp = requests.post(
        f"{ASSEMBLYAI_API_URL}/transcript",
        json=payload,
        headers=headers
    )
    resp.raise_for_status()
    transcript_id = resp.json()["id"]

    # Ждем результата
    poll_url = f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}"
    while True:
        poll = requests.get(poll_url, headers=headers).json()
        if poll["status"] == "completed":
            print("✅ AssemblyAI transcription completed")
            break
        if poll["status"] == "error":
            raise RuntimeError(poll.get("error"))
        time.sleep(3)

    full_text = poll.get("text", "").strip()

    # ⭐ Подставляем НАШИ сегменты — по тишине ⭐
    segments = segment_by_silence(audio_path, full_text)

    return {
        "text": full_text,
        "language": poll.get("language_code", expected_language),
        "segments": segments,
        "duration": poll.get("audio_duration")
    }


# ============================================================
# ⭐ 3. Whisper API (без сегментов в API) ⭐
# ============================================================

def _transcribe_with_whisper(audio_path, expected_language=None):
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )
    result = response.model_dump()

    # Whisper API не возвращает сегменты — делаем сами
    full_text = result.get("text", "")
    segments = segment_by_silence(audio_path, full_text)

    result["segments"] = segments
    return result


# ============================================================
# ⭐ 4. Основная функция whisper_transcribe() ⭐
# ============================================================

def whisper_transcribe(audio_file="input.wav", expected_language=None):
    audio_path = os.path.join(AUDIO_DIR, audio_file)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Audio file not found → {audio_path}")

    provider = TRANSCRIBE_PROVIDER.lower()
    print(f"🎧 Transcribing using {provider}: {audio_path}")

    if provider == "whisper":
        whisper_json = _transcribe_with_whisper(audio_path, expected_language)
    else:
        whisper_json = _transcribe_with_assemblyai(audio_path, expected_language)

    os.makedirs(WHISPER_DIR, exist_ok=True)

    json_path = os.path.join(WHISPER_DIR, "transcript.json")
    txt_path = os.path.join(WHISPER_DIR, "transcript.txt")

    # Сохраняем
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(whisper_json, jf, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(whisper_json.get("text", ""))

    print("📄 JSON saved")
    print("📝 TXT saved")

    # Валидация
    assert_valid_whisper(json_path, expected_language)

    # Очистка GPT
    whisper_json = clean_segments_with_gpt(whisper_json)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(whisper_json, jf, indent=2, ensure_ascii=False)

    print("🟢 Whisper validation PASSED")
    return json_path


if __name__ == "__main__":
    whisper_transcribe(expected_language="ar")
