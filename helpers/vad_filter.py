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
        localized_window_frames: int = 5,
        localized_voiced_ratio: float = 0.6,
        trailing_voiced_ratio: float = 0.7,
        trailing_portion_start: float = 0.5,
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
        self.localized_window_frames = max(localized_window_frames, 1)
        self.localized_voiced_ratio = localized_voiced_ratio
        self.trailing_voiced_ratio = trailing_voiced_ratio
        self.trailing_portion_start = min(max(trailing_portion_start, 0.0), 1.0)
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

        speech_flags = [self._vad.is_speech(frame, self.sample_rate) for frame in frames]
        voiced_frames = sum(1 for flag in speech_flags if flag)
        ratio = voiced_frames / len(frames)

        if voiced_frames >= self.min_voiced_frames and ratio >= self.min_voiced_ratio:
            return True

        # Localized sliding-window check: keep the segment if any subwindow has sustained speech.
        window = min(len(frames), self.localized_window_frames)
        for start in range(0, len(frames) - window + 1):
            window_flags = speech_flags[start : start + window]
            local_voiced = sum(1 for flag in window_flags if flag)
            local_ratio = local_voiced / window
            if local_voiced >= self.min_voiced_frames and local_ratio >= self.localized_voiced_ratio:
                return True

        # Trailing portion sanity check with a slightly higher threshold.
        trailing_start = int(len(frames) * self.trailing_portion_start)
        trailing_flags = speech_flags[trailing_start:]
        if trailing_flags:
            trailing_voiced = sum(1 for flag in trailing_flags if flag)
            trailing_ratio = trailing_voiced / len(trailing_flags)
            if trailing_voiced >= self.min_voiced_frames and trailing_ratio >= self.trailing_voiced_ratio:
                return True

        return False

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
    localized_window_frames: int = 5,
    localized_voiced_ratio: float = 0.6,
    trailing_voiced_ratio: float = 0.7,
    trailing_portion_start: float = 0.5,
) -> List[MutableMapping]:
    """Convenience helper to run VAD filtering over Whisper segments."""
    vad = VoiceActivityDetector(
        aggressiveness=aggressiveness,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        min_voiced_ratio=min_voiced_ratio,
        min_voiced_frames=min_voiced_frames,
        localized_window_frames=localized_window_frames,
        localized_voiced_ratio=localized_voiced_ratio,
        trailing_voiced_ratio=trailing_voiced_ratio,
        trailing_portion_start=trailing_portion_start,
    )
    return vad.filter_segments(segments, audio_path)


__all__ = ["VoiceActivityDetector", "filter_segments_by_vad"]
