import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ai.service import AIService
from config import TEMP_DIR
from services.downloader import is_supported_media_url

logger = logging.getLogger(__name__)


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str


async def _run_subprocess(*cmd: str, timeout: float | None = None) -> tuple[str, str, int]:
    logger.debug("Running subprocess: %s", " ".join(cmd))
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


async def extract_audio_from_video(video_path: Path) -> Path:
    audio_path = TEMP_DIR / f"{video_path.stem}_audio_{uuid4().hex}.wav"
    logger.info("Extracting audio from %s to %s", video_path, audio_path)

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        error_output = stderr.decode() or stdout.decode()
        logger.error("ffmpeg failed to extract audio from %s: %s", video_path, error_output)
        raise RuntimeError(f"ffmpeg failed to extract audio: {error_output}")

    return audio_path


async def transcribe_segments(audio_path: Path, ai_service: AIService) -> tuple[list[SubtitleSegment], str]:
    logger.info("Transcribing audio for subtitles: %s", audio_path)
    result = await ai_service.transcribe_audio(audio_path)
    language = result.get("language", "unknown")

    segments: list[SubtitleSegment] = []
    for seg in result.get("segments", []) or []:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = seg.get("text", "").strip()
            if text:
                segments.append(SubtitleSegment(start=start, end=end, text=text))
        except Exception:
            logger.exception("Failed to parse segment %s", seg)

    if not segments and result.get("text"):
        segments.append(SubtitleSegment(start=0.0, end=0.0, text=result["text"]))

    if not segments:
        raise RuntimeError("No transcription segments produced")

    return segments, language


async def translate_segments(
    segments: list[SubtitleSegment],
    source_language: str,
    target_language: str,
    ai_service: AIService,
) -> list[SubtitleSegment]:
    translated_segments: list[SubtitleSegment] = []
    for segment in segments:
        try:
            translated_text = await ai_service.translate_text(
                text=segment.text,
                source_language=source_language,
                target_language=target_language,
            )
        except Exception:
            logger.exception(
                "Failed to translate subtitle segment from %s to %s",
                source_language,
                target_language,
            )
            raise

        translated_segments.append(
            SubtitleSegment(start=segment.start, end=segment.end, text=translated_text)
        )

    return translated_segments


def _format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def build_srt_content(segments: list[SubtitleSegment]) -> str:
    logger.info("Building SRT content for %d segments", len(segments))
    lines: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        start_ts = _format_timestamp(segment.start)
        end_ts = _format_timestamp(segment.end if segment.end > segment.start else segment.start + 2)
        lines.extend([str(idx), f"{start_ts} --> {end_ts}", segment.text.strip(), ""])
    return "\n".join(lines).strip() + "\n"


async def burn_subtitles(video_path: Path, srt_content: str) -> Path:
    subtitles_path = TEMP_DIR / f"{video_path.stem}_subs_{uuid4().hex}.srt"
    output_path = TEMP_DIR / f"{video_path.stem}_subtitled_{uuid4().hex}.mp4"

    logger.info("Writing subtitles to %s", subtitles_path)
    subtitles_path.write_text(srt_content, encoding="utf-8")

    cmd = (
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"subtitles={subtitles_path.as_posix()}",
        "-c:a",
        "copy",
        str(output_path),
    )

    try:
        stdout, stderr, returncode = await _run_subprocess(*cmd)
        if returncode != 0:
            logger.error(
                "ffmpeg failed to burn subtitles into %s: %s", video_path, stderr or stdout
            )
            raise RuntimeError(f"ffmpeg failed to burn subtitles: {stderr or stdout}")
    finally:
        try:
            subtitles_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove temporary subtitles file %s", subtitles_path)

    return output_path


async def download_video_from_url(url: str) -> Path:
    if not is_supported_media_url(url):
        raise ValueError("Unsupported media URL")

    download_dir = TEMP_DIR / f"video_{uuid4()}"
    download_dir.mkdir(parents=True, exist_ok=True)
    output_template = download_dir / "%(title)s.%(ext)s"

    stdout, stderr, returncode = await _run_subprocess(
        "yt-dlp",
        "-f",
        "bv*+ba/b",  # best video + audio fallback to best
        "-o",
        str(output_template),
        url,
    )

    if returncode != 0:
        logger.error("yt-dlp failed for %s: %s", url, stderr or stdout)
        raise RuntimeError(f"yt-dlp failed: {stderr or stdout}")

    files = sorted(download_dir.glob("*"))
    if not files:
        raise FileNotFoundError("yt-dlp did not produce any files")

    return max(files, key=lambda p: p.stat().st_size)
