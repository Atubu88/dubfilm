"""Lightweight voice-activity detection for Whisper segments."""
from __future__ import annotations

import math
from typing import Iterable, List, MutableMapping

import webrtcvad
from pydub import AudioSegment


class VoiceActivityDetector:
    """Simple wrapper around WebRTC VAD for segment-level speech checks."""

    def __init__(
        self,
        aggressiveness: int = 2,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        min_voiced_ratio: float = 0.2,
        min_voiced_frames: int = 1,
    ) -> None:
        if aggressiveness < 0 or aggressiveness > 3:
            raise ValueError("aggressiveness must be in [0, 3]")
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("sample_rate must be one of 8000, 16000, 32000, 48000")
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30 milliseconds")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.min_voiced_ratio = min_voiced_ratio
        self.min_voiced_frames = min_voiced_frames
        self._vad = webrtcvad.Vad(aggressiveness)

    def _prepare_audio(self, audio_path: str) -> AudioSegment:
        """Load audio, force mono 16-bit PCM at target sample rate."""
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(self.sample_rate)
        audio = audio.set_sample_width(2)
        return audio

    def _frame_generator(self, audio_slice: AudioSegment) -> Iterable[bytes]:
        bytes_per_frame = int(self.sample_rate * (self.frame_ms / 1000.0)) * audio_slice.sample_width
        raw = audio_slice.raw_data

        for offset in range(0, len(raw) - bytes_per_frame + 1, bytes_per_frame):
            yield raw[offset : offset + bytes_per_frame]

    def _has_voice_in_slice(self, audio_slice: AudioSegment) -> bool:
        frames = list(self._frame_generator(audio_slice))
        if not frames:
            return False

        voiced_frames = sum(1 for frame in frames if self._vad.is_speech(frame, self.sample_rate))
        ratio = voiced_frames / len(frames)
        return voiced_frames >= self.min_voiced_frames and ratio >= self.min_voiced_ratio

    def filter_segments(self, segments: List[MutableMapping], audio_path: str) -> List[MutableMapping]:
        """Mark segments without speech as empty text."""
        if not segments:
            return []

        audio = self._prepare_audio(audio_path)
        filtered_segments: List[MutableMapping] = []
        voiced_count = 0
        emptied_count = 0

        for seg in segments:
            start_ms = int(math.floor(float(seg.get("start", 0)) * 1000))
            end_ms = int(math.ceil(float(seg.get("end", 0)) * 1000))

            if end_ms <= start_ms:
                seg["text"] = ""
                emptied_count += 1
                filtered_segments.append(seg)
                continue

            audio_slice = audio[start_ms:end_ms]
            if self._has_voice_in_slice(audio_slice):
                voiced_count += 1
            else:
                seg["text"] = ""
                emptied_count += 1

            filtered_segments.append(seg)

        print(
            "🔎 VAD filter → "
            f"total={len(segments)}, voiced={voiced_count}, emptied={emptied_count}"
        )
        return filtered_segments


def filter_segments_by_vad(
    segments: List[MutableMapping],
    audio_path: str,
    aggressiveness: int = 2,
    sample_rate: int = 16000,
    frame_ms: int = 30,
    min_voiced_ratio: float = 0.2,
    min_voiced_frames: int = 1,
) -> List[MutableMapping]:
    """Convenience helper to run VAD filtering over Whisper segments."""
    vad = VoiceActivityDetector(
        aggressiveness=aggressiveness,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        min_voiced_ratio=min_voiced_ratio,
        min_voiced_frames=min_voiced_frames,
    )
    return vad.filter_segments(segments, audio_path)


__all__ = ["VoiceActivityDetector", "filter_segments_by_vad"]
