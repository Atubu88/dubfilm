import asyncio
from pathlib import Path


FFPROBE_STREAM_CMD = (
    "ffprobe",
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=duration",
    "-of",
    "default=noprint_wrappers=1:nokey=1",
)

FFPROBE_FORMAT_CMD = (
    "ffprobe",
    "-v",
    "error",
    "-show_entries",
    "format=duration",
    "-of",
    "default=noprint_wrappers=1:nokey=1",
)


async def _run_ffprobe_duration(cmd: tuple[str, ...], path: Path) -> float | None:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        return None

    output = (stdout.decode() or "").strip()
    try:
        value = float(output)
    except (TypeError, ValueError):
        return None

    if value <= 0:
        return None

    return value


async def get_video_duration(path: Path) -> float:
    # 1) Try stream duration (often works for mp4)
    duration = await _run_ffprobe_duration(FFPROBE_STREAM_CMD, path)
    if duration is not None:
        return duration

    # 2) Fallback to container format duration (more reliable for webm)
    duration = await _run_ffprobe_duration(FFPROBE_FORMAT_CMD, path)
    if duration is not None:
        return duration

    raise RuntimeError("Unable to determine video duration")


async def validate_video_duration(path: Path, max_seconds: int = 0) -> None:
    try:
        duration = await get_video_duration(path)
    except Exception as exc:
        raise ValueError(f"Cannot read video duration: {exc}")

    # max_seconds <= 0 means "no limit"
    if max_seconds and max_seconds > 0 and duration > max_seconds:
        raise ValueError("Video too long")


async def validate_media_duration(path: Path, max_seconds: int = 0) -> None:
    """
    Универсальная проверка длительности и для аудио, и для видео.
    max_seconds <= 0 => без лимита.
    """
    duration = await get_video_duration(path)
    if max_seconds and max_seconds > 0 and duration > max_seconds:
        raise ValueError("Video too long")
