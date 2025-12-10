import asyncio
import logging
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from config import TEMP_DIR
from services.audio import convert_to_wav
from services.video_duration import validate_video_duration


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
        logger.error("Command %s timed out after %.0f seconds", cmd[0], timeout or 0)
        process.kill()
        stdout, stderr = await process.communicate()
        raise TimeoutError(f"{cmd[0]} timed out after {timeout} seconds")
    return stdout.decode(), stderr.decode(), process.returncode


async def _download_media(url: str) -> Path:
    download_dir = TEMP_DIR / f"download_{uuid4()}"
    download_dir.mkdir(parents=True, exist_ok=True)
    output_template = download_dir / "%(title)s.%(ext)s"

    stdout, stderr, returncode = await _run_subprocess(
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "-o",
        str(output_template),
        url,
        timeout=DOWNLOAD_TIMEOUT,
    )

    if returncode != 0:
        logger.error("yt-dlp failed for %s: %s", url, stderr or stdout)
        raise RuntimeError(f"yt-dlp failed: {stderr or stdout}")

    files = sorted(download_dir.glob("*"))
    if not files:
        logger.error("yt-dlp produced no files for %s", url)
        raise FileNotFoundError("yt-dlp did not produce any files")

    return max(files, key=lambda p: p.stat().st_size)


async def download_audio_from_url(url: str) -> Path:
    media_path = await _download_media(url)
    try:
        await validate_video_duration(media_path)
        wav_path = await convert_to_wav(media_path)
    finally:
        try:
            media_path.unlink(missing_ok=True)
        except OSError:
            # Directory may not be empty or already removed
            pass

    return wav_path
