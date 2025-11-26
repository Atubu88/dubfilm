import os

import assemblyai as aai

from config import ASSEMBLYAI_API_KEY
from pipeline.constants import AUDIO_DIR, WHISPER_DIR


def assemblyai_transcribe(audio_file: str = "input.wav") -> str:
    """
    🔊 Transcribe pipeline audio with AssemblyAI and export SRT subtitles.

    Args:
        audio_file: filename inside 2_audio/ to transcribe.

    Returns:
        Path to the saved SRT file.
    """

    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("❌ Missing ASSEMBLYAI_API_KEY in environment")

    audio_path = os.path.join(AUDIO_DIR, audio_file)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Audio not found: {audio_path}")

    print("🔑 Initializing AssemblyAI client…")
    aai.settings.api_key = ASSEMBLYAI_API_KEY

    print(f"🛰 Sending '{audio_file}' to AssemblyAI for transcription…")
    transcript = aai.Transcriber().transcribe(audio_path)

    subtitles = transcript.export_subtitles_srt()

    os.makedirs(WHISPER_DIR, exist_ok=True)
    srt_path = os.path.join(WHISPER_DIR, "subtitles.srt")

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(subtitles)

    print(f"✅ SRT subtitles saved → {srt_path}")
    return srt_path


if __name__ == "__main__":
    assemblyai_transcribe()
