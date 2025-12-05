from pathlib import Path

from ai.provider import AIProvider


class AIService:
    def __init__(self, provider: AIProvider) -> None:
        self.provider = provider

    async def transcribe_audio(self, file_path: Path) -> dict:
        return await self.provider.transcribe(file_path)

    async def translate_text(self, text: str, source_language: str, target_language: str) -> str:
        return await self.provider.translate(text=text, source_language=source_language, target_language=target_language)

    async def summarize_text(self, original_text: str, translated_text: str, target_language: str) -> str:
        return await self.provider.summarize(
            original_text=original_text,
            translated_text=translated_text,
            target_language=target_language,
        )
