import asyncio
import json
import logging
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from config import TEMP_DIR
from services.audio import convert_to_wav

logger = logging.getLogger(__name__)

SUPPORTED_DOMAINS = (
    "youtube.com",
    "youtu.be",
    "tiktok.com",
    "instagram.com",
    "facebook.com",
    "fb.watch",
)

DOWNLOAD_TIMEOUT = 120
MAX_DURATION_SEC = 300  # 5 минут


def is_supported_media_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return any(domain in hostname for domain in SUPPORTED_DOMAINS)


async def _run_subprocess(*cmd: str, timeout: float | None = None) -> tuple[str, str, int]:
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        raise TimeoutError(f"{cmd[0]} timed out after {timeout} seconds")

    return stdout.decode(), stderr.decode(), process.returncode


# 🔥 NEW: проверка длительности до скачивания
async def _get_duration_from_url(url: str) -> float:
    stdout, stderr, returncode = await _run_subprocess(
        "yt-dlp",
        "--dump-json",
        url,
        timeout=20,
    )

    if returncode != 0:
        raise RuntimeError(stderr or stdout)

    try:
        info = json.loads(stdout)
        duration = float(info.get("duration") or 0)
        return duration
    except Exception:
        raise RuntimeError("Unable to read duration from yt-dlp")


async def _download_media(url: str) -> Path:
    download_dir = TEMP_DIR / f"download_{uuid4()}"
    download_dir.mkdir(parents=True, exist_ok=True)
    output_template = download_dir / "%(title)s.%(ext)s"

    stdout, stderr, returncode = await _run_subprocess(
        "yt-dlp",
        "-f",
        "bv*+ba/b",
        "-o",
        str(output_template),
        url,
        timeout=DOWNLOAD_TIMEOUT,
    )

    if returncode != 0:
        raise RuntimeError(stderr or stdout)

    files = sorted(download_dir.glob("*"))
    if not files:
        raise FileNotFoundError("yt-dlp produced no files")

    return max(files, key=lambda p: p.stat().st_size)


async def download_audio_from_url(url: str) -> Path:
    # 🔥 1. Проверяем длительность ДО скачивания
    duration = await _get_duration_from_url(url)

    if duration > MAX_DURATION_SEC:
        raise ValueError("Video too long")

    # 🔥 2. Скачиваем, если видео прошло проверку
    media_path = await _download_media(url)

    try:
        wav_path = await convert_to_wav(media_path)
        return wav_path
    finally:
        try:
            media_path.unlink(missing_ok=True)
        except OSError:
            pass
