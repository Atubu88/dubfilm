import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from typing import List

from openai import OpenAI
from pydub import AudioSegment

from config import OPENAI_API_KEY, OUTPUT_DIR, TRANSLATION_DIR, WHISPER_DIR

client = OpenAI(api_key=OPENAI_API_KEY)

TRANSLATED_JSON = os.path.join(TRANSLATION_DIR, "translated.json")
TRANSCRIPT_JSON = os.path.join(WHISPER_DIR, "transcript.json")
MP3_OUTPUT = os.path.join(OUTPUT_DIR, "voice_over_tts.mp3")
WAV_OUTPUT = os.path.join(OUTPUT_DIR, "voice_over_tts.wav")
FINAL_AUDIO = os.path.join(OUTPUT_DIR, "final_audio.wav")
VOICE_CACHE_DIR = os.path.join(OUTPUT_DIR, "voice_over_segments")

TARGET_SAMPLE_RATE = 16000
TOLERANCE_MS = 10
SILENCE_TAIL_MS = 500


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str

    @property
    def duration_ms(self) -> int:
        return max(int(round((self.end - self.start) * 1000)), 0)


class VoiceOverError(RuntimeError):
    pass


def load_translated_segments() -> List[Segment]:
    if not os.path.exists(TRANSLATED_JSON):
        raise VoiceOverError("❌ translated.json not found — cannot build voice-over track")

    with open(TRANSLATED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise VoiceOverError("❌ translated.json has unexpected format (expected a list)")

    segments: List[Segment] = []
    for raw in data:
        text = (raw.get("dst") or "").strip()
        if not text:
            continue

        start = float(raw.get("start", 0.0))
        end = float(raw.get("end", start))
        if end <= start:
            continue

        segment_id = int(raw.get("id", len(segments)))
        segments.append(Segment(id=segment_id, start=start, end=end, text=text))

    if not segments:
        raise VoiceOverError("❌ No translated segments with text found")

    print(f"📄 Loaded {len(segments)} translated segments for voice-over")
    return segments


def load_original_duration(segments: List[Segment]) -> float:
    if os.path.exists(TRANSCRIPT_JSON):
        with open(TRANSCRIPT_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)
        duration = meta.get("duration")
        if isinstance(duration, (int, float)) and duration > 0:
            return float(duration)

    return max((seg.end for seg in segments), default=0.0)


def synthesize_text_to_audio(text: str) -> AudioSegment:
    print(f"🔊 Synthesizing TTS for: {text[:60]}{'…' if len(text) > 60 else ''}")
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="nova",
        input=text,
    )

    audio_bytes = response.read()
    buffer = io.BytesIO(audio_bytes)
    audio = AudioSegment.from_file(buffer, format="mp3")
    audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
    return audio


def build_atempo_chain(ratio: float) -> str:
    if ratio <= 0:
        raise VoiceOverError("❌ Invalid tempo ratio computed")

    filters = []
    temp_ratio = ratio
    while temp_ratio < 0.5 or temp_ratio > 2.0:
        if temp_ratio < 0.5:
            filters.append("atempo=0.5")
            temp_ratio /= 0.5
        else:
            filters.append("atempo=2.0")
            temp_ratio /= 2.0
    filters.append(f"atempo={temp_ratio:.4f}")
    return ",".join(filters)


def stretch_with_ffmpeg(audio: AudioSegment, ratio: float) -> AudioSegment:
    os.makedirs(VOICE_CACHE_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        audio.export(tmp_in.name, format="wav")
        input_path = tmp_in.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        filter_chain = build_atempo_chain(ratio)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-filter:a",
            filter_chain,
            output_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise VoiceOverError("❌ Failed to stretch audio via ffmpeg")
        stretched = AudioSegment.from_file(output_path, format="wav")
        stretched = stretched.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
    finally:
        for path in (input_path, output_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    return stretched


def match_duration(audio: AudioSegment, target_ms: int) -> AudioSegment:
    if target_ms <= 0:
        return AudioSegment.silent(duration=0)

    current_ms = len(audio)
    if current_ms == 0:
        return AudioSegment.silent(duration=target_ms)

    if abs(current_ms - target_ms) <= TOLERANCE_MS:
        trimmed = audio[:target_ms]
        if len(trimmed) < target_ms:
            trimmed += AudioSegment.silent(duration=target_ms - len(trimmed))
        return trimmed

    ratio = current_ms / target_ms
    stretched = stretch_with_ffmpeg(audio, ratio)

    if len(stretched) > target_ms:
        stretched = stretched[:target_ms]
    elif len(stretched) < target_ms:
        stretched += AudioSegment.silent(duration=target_ms - len(stretched))

    return stretched


def place_segments_on_timeline(segments: List[Segment], total_duration: float) -> AudioSegment:
    timeline_duration_ms = int(math.ceil(total_duration * 1000)) + SILENCE_TAIL_MS
    print(f"🧱 Building voice-over timeline of {timeline_duration_ms / 1000:.2f}s")
    final_audio = AudioSegment.silent(duration=timeline_duration_ms, frame_rate=TARGET_SAMPLE_RATE)

    for seg in segments:
        target_ms = seg.duration_ms
        if target_ms <= 0:
            continue

        synthesized = synthesize_text_to_audio(seg.text)
        stretched = match_duration(synthesized, target_ms)

        position_ms = int(seg.start * 1000)
        print(
            f"  • Segment {seg.id}: start={seg.start:.2f}s end={seg.end:.2f}s duration={target_ms / 1000:.2f}s"
        )
        final_audio = final_audio.overlay(stretched, position=position_ms)

    return final_audio


def sanity_check_wav(path: str, min_duration: float) -> None:
    if not os.path.exists(path):
        raise VoiceOverError(f"❌ WAV not found → {path}")

    if os.path.getsize(path) == 0:
        raise VoiceOverError("❌ WAV file is empty")

    with wave.open(path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)

    print(f"🔎 Voice-over WAV duration: {duration:.2f}s")
    if duration < min_duration:
        raise VoiceOverError(
            f"❌ Voice-over WAV too short ({duration:.2f}s). Expected at least {min_duration:.2f}s"
        )


def export_audio_track(audio: AudioSegment) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
    audio.export(WAV_OUTPUT, format="wav")
    audio.export(MP3_OUTPUT, format="mp3")
    shutil.copyfile(WAV_OUTPUT, FINAL_AUDIO)
    print(f"💾 Saved WAV → {WAV_OUTPUT}")
    print(f"💾 Saved MP3 → {MP3_OUTPUT}")
    print(f"📦 Copied voice-over track to FINAL_AUDIO → {FINAL_AUDIO}")


def generate_voice_over_track():
    segments = load_translated_segments()
    total_duration = load_original_duration(segments)
    if total_duration <= 0:
        raise VoiceOverError("❌ Unable to determine original duration")

    final_audio = place_segments_on_timeline(segments, total_duration)
    export_audio_track(final_audio)
    sanity_check_wav(WAV_OUTPUT, min_duration=max(0.5, total_duration - 0.5))
    print("🟢 Voice-over track ready!")


if __name__ == "__main__":
    generate_voice_over_track()