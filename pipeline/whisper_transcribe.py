import os
import warnings

from pipeline.assemblyai_transcribe import assemblyai_transcribe
from pipeline.constants import AUDIO_DIR


def whisper_transcribe(audio_file: str = "input.wav", expected_language=None):
    """
    Deprecated shim retained for backward compatibility.

    All previous Whisper/VAD segmentation is disabled — we now rely solely on
    AssemblyAI SRT subtitles as the timing source for downstream steps.
    """

    warnings.warn(
        "whisper_transcribe is deprecated. Use assemblyai_transcribe instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    audio_path = os.path.join(AUDIO_DIR, audio_file)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Audio not found: {audio_path}")

    return assemblyai_transcribe(audio_file)


if __name__ == "__main__":
    whisper_transcribe()
