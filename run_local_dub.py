import asyncio
from pathlib import Path

from ai.service import AIService
from ai.provider import AIProvider
from config import OPENAI_CHAT_MODEL, OPENAI_WHISPER_MODEL
from pipelines.dub import run_dub_pipeline

import os

VIDEO = Path(os.getenv('DUB_LOCAL_INPUT', '/home/fanfan/projects/dubfilm/in/e62512f8-a22e-4c1f-adaa-7bfe305e4e3f.mp4'))
OUT_DIR = Path('/home/fanfan/projects/dubfilm/out')


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ai = AIService(provider=AIProvider(chat_model=OPENAI_CHAT_MODEL, whisper_model=OPENAI_WHISPER_MODEL))
    out = await run_dub_pipeline(VIDEO, 'Russian', ai)
    final = OUT_DIR / f'{VIDEO.stem}_ru_dub.mp4'
    out.replace(final)
    print(final)


if __name__ == '__main__':
    asyncio.run(main())
