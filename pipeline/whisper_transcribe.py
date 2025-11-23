import json
import os
import re
import time

import requests
from openai import OpenAI
from pydub import AudioSegment
from pydub.silence import detect_nonsilent, detect_silence

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

    # Используем реальные паузы: учитываем позиции тишины, а не только длину куска
    min_pause_ms = 950
    keep_silence_ms = 350
    min_segment_ms = 5000
    max_segment_ms = 22000

    # Точные позиции тишины (>= ~1 сек). Если их нет — не режем аудио.
    long_pauses = detect_silence(
        audio,
        min_silence_len=min_pause_ms,
        silence_thresh=silence_threshold
    )

    if not long_pauses:
        return [{
            "id": 0,
            "start": 0.0,
            "end": len(audio) / 1000,
            "text": full_text.strip()
        }]

    # Формируем сегменты по реальным таймкодам тишины, сохраняя края паузы
    segments_ms = []
    cursor = 0
    pad = keep_silence_ms // 2

    for pause_start, pause_end in long_pauses:
        seg_end = max(cursor, pause_start + pad)
        if seg_end - cursor > 0:
            segments_ms.append([cursor, seg_end])
        cursor = max(seg_end, pause_end - pad)

    if cursor < len(audio):
        segments_ms.append([cursor, len(audio)])

    segments_ms = [seg for seg in segments_ms if seg[1] - seg[0] > 0]

    # Если пауз мало → не дробим аудио на искусственные куски
    if len(segments_ms) < 3:
        return [{
            "id": 0,
            "start": 0.0,
            "end": len(audio) / 1000,
            "text": full_text.strip()
        }]

    # Сливаем короткие сегменты, чтобы минимальная длина была 4–6 секунд
    merged_segments = []
    acc_start, acc_end = None, None
    for start, end in segments_ms:
        if acc_start is None:
            acc_start, acc_end = start, end
            continue

        if acc_end - acc_start < min_segment_ms:
            acc_end = end
        elif end - start < min_segment_ms:
            acc_end = end
        else:
            merged_segments.append([acc_start, acc_end])
            acc_start, acc_end = start, end

    if acc_start is not None:
        merged_segments.append([acc_start, acc_end])

    if len(merged_segments) > 1 and merged_segments[-1][1] - merged_segments[-1][0] < min_segment_ms:
        merged_segments[-2][1] = merged_segments[-1][1]
        merged_segments.pop()

    # Контроль огромных сегментов: делим слишком длинные, но только крупные (>> 20 c)
    bounded_segments = []
    for start, end in merged_segments:
        duration = end - start
        if duration <= max_segment_ms or duration <= min_segment_ms * 2:
            bounded_segments.append([start, end])
            continue

        parts = max(1, round(duration / max_segment_ms))
        step = duration / parts
        for i in range(parts):
            seg_start = int(start + i * step)
            seg_end = int(start + (i + 1) * step)
            bounded_segments.append([seg_start, seg_end])

    segments_ms = bounded_segments

    if len(segments_ms) < 3:
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

    # Реальный объём речи в каждом сегменте (а не длина паузы)
    speech_ranges = detect_nonsilent(
        audio,
        min_silence_len=200,
        silence_thresh=silence_threshold
    )

    def speech_duration_ms(seg_start, seg_end):
        total = 0
        for speech_start, speech_end in speech_ranges:
            overlap = max(0, min(seg_end, speech_end) - max(seg_start, speech_start))
            total += overlap
        return total

    def estimate_tts_ms(text: str) -> int:
        words = len(text.split())
        if not words:
            return 0
        # ~0.43 c/слово + небольшой запас на паузы
        return max(1200, int(words * 430))

    def distribute_text(segment_windows):
        sentence_data = [(s, count_words(s)) for s in sentences]
        total_words = sum(cnt for _, cnt in sentence_data)

        # веса по реальной речи; если не нашли — используем 60% длительности
        weights = []
        for start, end in segment_windows:
            speech_ms = speech_duration_ms(start, end)
            weights.append(max(speech_ms, int((end - start) * 0.6)))

        whisper_segments = []

        for idx, (start, end) in enumerate(segment_windows):
            remaining_segments = len(segment_windows) - idx

            if not sentence_data:
                break

            if remaining_segments == 1:
                picked = sentence_data
                sentence_data = []
            else:
                remaining_weight = sum(weights[idx:]) or 1
                target_words = max(1, round(total_words * weights[idx] / remaining_weight))

                picked = []
                picked_words = 0

                while sentence_data:
                    if len(sentence_data) <= (remaining_segments - 1) and picked:
                        break

                    sent, count = sentence_data[0]
                    if picked and picked_words + count > target_words:
                        break

                    picked.append(sentence_data.pop(0))
                    picked_words += count

                if not picked:
                    picked.append(sentence_data.pop(0))

            text = " ".join(s for s, _ in picked).strip()
            total_words -= sum(cnt for _, cnt in picked)

            whisper_segments.append({
                "id": len(whisper_segments),
                "start": start / 1000,
                "end": end / 1000,
                "text": text,
                "_duration_ms": end - start
            })

        return whisper_segments

    # Итеративно объединяем, если текст не помещается в окно
    safety_counter = 0
    while True:
        safety_counter += 1
        whisper_segments = distribute_text(segments_ms)

        violation_idx = None
        for idx, seg in enumerate(whisper_segments[:-1]):
            if estimate_tts_ms(seg["text"]) > seg["_duration_ms"]:
                violation_idx = idx
                break

        if violation_idx is None or len(segments_ms) == 1 or safety_counter > 6:
            break

        # Расширяем окно, объединяя с соседним сегментом
        segments_ms[violation_idx][1] = segments_ms[violation_idx + 1][1]
        segments_ms.pop(violation_idx + 1)

    # Убираем техническое поле
    for seg in whisper_segments:
        seg.pop("_duration_ms", None)

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
