from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from ai.base import BaseAIProvider
from config import OPENAI_API_KEY


class AIProvider(BaseAIProvider):
    def __init__(self, chat_model: str, whisper_model: str) -> None:
        self.chat_model = chat_model
        self.whisper_model = whisper_model
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def transcribe(self, file_path: Path) -> dict[str, Any]:
        with open(file_path, "rb") as audio_file:
            response = await self.client.audio.transcriptions.create(
                model=self.whisper_model,
                file=audio_file,
                response_format="verbose_json",
            )
        return {"text": response.text, "language": response.language}

    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        prompt = (
            "You are a translator. Translate the user message from {src} to {tgt}. "
            "Return translation only without quotes or commentary."
        ).format(src=source_language, tgt=target_language)

        response = await self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    async def summarize(self, original_text: str, translated_text: str, target_language: str) -> str:
        prompt = (
            "You are a helpful assistant. Summarize the original text concisely in {lang}. "
            "Use up to 3 sentences. Use the translation for context if needed."
        ).format(lang=target_language)

        response = await self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "assistant",
                    "content": "Original text:\n" + original_text + "\n\nTranslation:\n" + translated_text,
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()
