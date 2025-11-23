import json
import os
import re
import time

import requests
from openai import OpenAI
from pydub import AudioSegment
from pydub.silence import split_on_silence

from config import OPENAI_API_KEY, ASSEMBLYAI_API_KEY, TRANSCRIBE_PROVIDER
from helpers.gpt_cleaner import clean_segments_with_gpt
from helpers.validators import assert_valid_whisper
from pipeline.constants import AUDIO_DIR, WHISPER_DIR

client = OpenAI(api_key=OPENAI_API_KEY)
ASSEMBLYAI_API_URL = "https://api.assemblyai.com/v2"


# ============================================================
# ⭐ 1. ПРОФЕССИОНАЛЬНАЯ СЕГМЕНТАЦИЯ — ПО ТИШИНЕ ⭐
# ============================================================

def segment_by_silence(audio_path: str, full_text: str):
    """
    Делим аудио по паузам → затем равномерно распределяем текст по сегментам.
    """

    audio = AudioSegment.from_file(audio_path)

    # 🔥 Адаптивный порог тишины
    silence_threshold = audio.dBFS - 15

    chunks = split_on_silence(
        audio,
        min_silence_len=220,             # 0.22 сек → подходит для арабской речи
        silence_thresh=silence_threshold,
        keep_silence=80
    )

    if not chunks:
        # fallback: один сегмент
        return [{
            "id": 0,
            "start": 0.0,
            "end": len(audio) / 1000,
            "text": full_text.strip()
        }]

    # ⭐ Разбиваем текст на предложения с учетом арабского языка
    pattern = re.compile(r"[^.!?؟…]+(?:[.!?؟…]+|$)")
    sentences = [s.strip() for s in pattern.findall(full_text) if s.strip()]

    def count_words(s):
        return len(s.split())

    # Считаем длину каждого аудио-сегмента
    segments = []
    cursor = 0
    for chunk in chunks:
        start = cursor
        end = cursor + len(chunk)
        segments.append((start, end))
        cursor = end

    durations = [end - start for start, end in segments]
    total_duration = sum(durations) or 1

    # Готовим предложения (sentence + word_count)
    sentence_data = [(s, count_words(s)) for s in sentences]
    total_words = sum(cnt for _, cnt in sentence_data)

    whisper_segments = []

    for idx, (start, end) in enumerate(segments):
        remaining_segments = len(segments) - idx

        if not sentence_data:
            break

        # Последний сегмент — кладём всё оставшееся
        if remaining_segments == 1:
            picked = sentence_data
            sentence_data = []
        else:
            remaining_dur = sum(durations[idx:])
            target = max(1, round(total_words * durations[idx] / remaining_dur))

            picked = []
            picked_words = 0

            while sentence_data:
                # Чтобы впереди остался хотя бы один sentence
                if len(sentence_data) <= (remaining_segments - 1) and picked:
                    break

                sent, count = sentence_data[0]

                # Не переполняем сегмент
                if picked and picked_words + count > target:
                    break

                picked.append(sentence_data.pop(0))
                picked_words += count

            # safety fallback
            if not picked:
                picked.append(sentence_data.pop(0))

        text = " ".join(s for s, _ in picked).strip()
        total_words -= sum(cnt for _, cnt in picked)

        whisper_segments.append({
            "id": len(whisper_segments),
            "start": start / 1000,
            "end": end / 1000,
            "text": text
        })

    return whisper_segments


# ============================================================
# ⭐ 2. AssemblyAI без их сегментов — мы делаем свои ⭐
# ============================================================

def _transcribe_with_assemblyai(audio_path, expected_language=None):
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("❌ ASSEMBLYAI_API_KEY missing")

    headers = {"authorization": ASSEMBLYAI_API_KEY}

    def _read_file(path):
        with open(path, "rb") as f:
            while chunk := f.read(5_242_880):
                yield chunk

    print("⬆️  Uploading audio...")
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
        "punctuate": True
    }
    if expected_language:
        payload["language_code"] = expected_language

    print("🛰  Requesting transcription...")
    resp = requests.post(
        f"{ASSEMBLYAI_API_URL}/transcript",
        json=payload,
        headers=headers
    )
    resp.raise_for_status()
    transcript_id = resp.json()["id"]

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
    segments = segment_by_silence(audio_path, full_text)

    return {
        "text": full_text,
        "language": poll.get("language_code", expected_language),
        "segments": segments,
        "duration": poll.get("audio_duration")
    }


# ============================================================
# ⭐ 3. Whisper API → сегменты тоже по паузам ⭐
# ============================================================

def _transcribe_with_whisper(audio_path, expected_language=None):
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )
    result = resp.model_dump()

    full_text = result.get("text", "")
    segments = segment_by_silence(audio_path, full_text)

    result["segments"] = segments
    return result


# ============================================================
# ⭐ 4. Основная функция ⭐
# ============================================================

def whisper_transcribe(audio_file="input.wav", expected_language=None):
    audio_path = os.path.join(AUDIO_DIR, audio_file)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Audio file not found → {audio_path}")

    provider = TRANSCRIBE_PROVIDER.lower()
    print(f"🎧 Transcribing using {provider}: {audio_path}")

    if provider == "whisper":
        data = _transcribe_with_whisper(audio_path, expected_language)
    else:
        data = _transcribe_with_assemblyai(audio_path, expected_language)

    os.makedirs(WHISPER_DIR, exist_ok=True)

    json_path = os.path.join(WHISPER_DIR, "transcript.json")
    txt_path = os.path.join(WHISPER_DIR, "transcript.txt")

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(data.get("text", ""))

    print("📄 JSON saved")
    print("📝 TXT saved")

    # Валидация структуры
    assert_valid_whisper(json_path, expected_language)

    # GPT очистка сегментов
    cleaned = clean_segments_with_gpt(data)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(cleaned, jf, ensure_ascii=False, indent=2)

    print("🟢 Whisper validation PASSED")
    return json_path


if __name__ == "__main__":
    whisper_transcribe(expected_language="ar")
