import pathlib
import sys
import os

from pydub import AudioSegment
from pydub.generators import Sine
from pytest import approx

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ASSEMBLYAI_API_KEY", "test-key")

from pipeline.whisper_transcribe import segment_by_silence


def test_segment_by_silence_uses_pauses(tmp_path):
    tone = Sine(440).to_audio_segment(duration=500).apply_gain(-3)
    pause = AudioSegment.silent(duration=300)

    # три чётких блока речи, разделённых паузами
    audio = tone + pause + tone + pause + tone

    audio_path = os.path.join(tmp_path, "sample.wav")
    audio.export(audio_path, format="wav")

    text = "Первый. Второй. Третий."

    segments = segment_by_silence(audio_path, text)

    assert len(segments) == 3

    starts = [seg["start"] for seg in segments]
    ends = [seg["end"] for seg in segments]

    # каждая граница сегмента соответствует своей полосе речи между паузами
    assert starts == approx([0.0, 0.72, 1.52], rel=0, abs=0.03)
    assert ends == approx([0.58, 1.38, 2.10], rel=0, abs=0.03)

    # текст равномерно распределяется по непрерывным отрезкам речи
    assert segments[0]["text"].startswith("Первый")
    assert segments[1]["text"].startswith("Второй")
    assert segments[2]["text"].startswith("Третий")

    for previous, current in zip(segments, segments[1:]):
        assert current["start"] >= previous["end"]
