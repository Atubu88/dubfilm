import asyncio
from pathlib import Path


FFPROBE_CMD = (
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


async def get_video_duration(path: Path) -> float:
    process = await asyncio.create_subprocess_exec(
        *FFPROBE_CMD,
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_output = stderr.decode() or stdout.decode()
        raise RuntimeError(f"ffprobe failed to read duration: {error_output}")

    output = stdout.decode().strip()
    try:
        duration = float(output)
    except (TypeError, ValueError):
        raise RuntimeError("Unable to determine video duration") from None

    if duration <= 0:
        raise RuntimeError("Unable to determine video duration")

    return duration


async def validate_video_duration(path: Path, max_seconds: int = 300) -> None:
    duration = await get_video_duration(path)
    if duration > max_seconds:
        raise ValueError("Video too long")

