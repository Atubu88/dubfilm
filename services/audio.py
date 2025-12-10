import asyncio
import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from aiogram import Bot
from aiogram.types import Message

from config import TEMP_DIR
from services.video_duration import validate_media_duration  # <-- ВАЖНО

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024      # лимит входного файла
MAX_WHISPER_SIZE_BYTES = 25 * 1024 * 1024   # лимит OpenAI


def _extract_file_data(message: Message) -> tuple[str, Optional[str], Optional[int]]:
    if message.audio:
        suffix = Path(message.audio.file_name or "audio").suffix or ".mp3"
        return message.audio.file_id, suffix, message.audio.file_size

    if message.voice:
        return message.voice.file_id, ".ogg", message.voice.file_size

    if message.video:
        return message.video.file_id, ".mp4", message.video.file_size

    if message.video_note:
        return message.video_note.file_id, ".mp4", message.video_note.file_size

    if message.document:
        suffix = Path(message.document.file_name or "file").suffix
        return message.document.file_id, suffix or ".dat", message.document.file_size

    raise ValueError("Unsupported media type")


async def _download_file(bot: Bot, file_id: str, destination: Path) -> Path:
    file = await bot.get_file(file_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    await bot.download_file(file.file_path, destination)
    return destination


async def convert_to_wav(source_path: Path) -> Path:
    target_path = source_path.with_name(f"{source_path.stem}_conv.wav")

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i", str(source_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(target_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode() or stdout.decode())

    return target_path


async def prepare_audio_file(bot: Bot, media: Message) -> Path:
    file_id, suffix, file_size = _extract_file_data(media)

    # определяем тип
    is_video_media = bool(media.video or media.video_note)
    is_audio_media = bool(media.audio or media.voice)

    if media.document:
        mime = message.document.mime_type or ""
        is_video_media = is_video_media or mime.startswith("video/")
        is_audio_media = is_audio_media or mime.startswith("audio/")

    # проверка размера входного файла
    if file_size and file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError("File exceeds maximum allowed size")

    # скачиваем
    raw_path = TEMP_DIR / f"{uuid4()}{suffix}"
    downloaded_path = await _download_file(bot, file_id, raw_path)

    try:
        # проверяем длительность и АУДИО, и ВИДЕО
        if is_audio_media or is_video_media:
            await validate_media_duration(downloaded_path)

        # конвертация
        wav_path = await convert_to_wav(downloaded_path)

        # проверяем размер WAV перед отправкой в Whisper
        if wav_path.stat().st_size > MAX_WHISPER_SIZE_BYTES:
            raise ValueError("Audio too large")

        return wav_path

    finally:
        downloaded_path.unlink(missing_ok=True)


def get_media_size(media: Message) -> Optional[int]:
    _, _, file_size = _extract_file_data(media)
    return file_size
