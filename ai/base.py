from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseAIProvider(ABC):
    @abstractmethod
    async def transcribe(self, file_path: Path) -> dict[str, Any]:
        """Transcribe audio file to text."""

    @abstractmethod
    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        """Translate text from source_language to target_language."""

    @abstractmethod
    async def summarize(self, original_text: str, translated_text: str, target_language: str) -> str:
        """Summarize original_text using translated_text context in target_language."""
