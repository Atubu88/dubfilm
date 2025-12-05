from typing import Any

from openai import AsyncOpenAI

from config import OPENAI_API_KEY

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def ask_ai(kind: str, **kwargs: Any) -> Any:
    if kind == "transcription":
        return await client.audio.transcriptions.create(**kwargs)
    if kind == "chat":
        return await client.chat.completions.create(**kwargs)
    raise ValueError(f"Unsupported AI operation: {kind}")
