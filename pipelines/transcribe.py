from pathlib import Path

from ai.service import AIService


async def run_transcription(audio_path: Path, ai_service: AIService) -> dict:
    return await ai_service.transcribe_audio(audio_path)
