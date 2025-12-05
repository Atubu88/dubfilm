import asyncio
from pathlib import Path
from typing import Optional
from uuid import uuid4

from aiogram import Bot
from aiogram.types import Message

from config import TEMP_DIR


def _extract_file_data(message: Message) -> tuple[str, Optional[str]]:
    if message.audio:
        return message.audio.file_id, Path(message.audio.file_name or "audio").suffix or ".mp3"
    if message.voice:
        return message.voice.file_id, ".ogg"
    if message.video:
        return message.video.file_id, ".mp4"
    if message.video_note:
        return message.video_note.file_id, ".mp4"
    if message.document:
        suffix = Path(message.document.file_name or "file").suffix
        return message.document.file_id, suffix or ".dat"
    raise ValueError("Unsupported media type")


async def _download_file(bot: Bot, file_id: str, destination: Path) -> Path:
    file = await bot.get_file(file_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    await bot.download_file(file.file_path, destination)
    return destination


async def _convert_to_wav(source_path: Path) -> Path:
    target_path = source_path.with_suffix(".wav")
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(target_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")
    return target_path


async def prepare_audio_file(bot: Bot, media: Message) -> Path:
    file_id, suffix = _extract_file_data(media)
    raw_path = TEMP_DIR / f"{uuid4()}{suffix}"
    downloaded_path = await _download_file(bot, file_id, raw_path)
    return await _convert_to_wav(downloaded_path)
