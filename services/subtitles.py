import asyncio
import logging
import re
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
    return await batch_translate_segments(
        segments=segments,
        source_language=source_language,
        target_language=target_language,
        ai_service=ai_service,
    )


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


async def batch_translate_segments(
    segments: list[SubtitleSegment],
    source_language: str,
    target_language: str,
    ai_service: AIService,
) -> list[SubtitleSegment]:
    if not segments:
        return []

    numbered_texts: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        segment_text = (segment.text or "").strip()
        numbered_texts.append(f"[{idx}] {segment_text}")

    prompt = (
        "Переведи каждый пункт списка на {target_language}.\n"
        "Сохрани нумерацию и порядок.\n"
        "Не добавляй комментариев.\n"
        "Не объединяй строки.\n"
        "Формат ответа строго:\n\n"
        "[1] перевод\n"
        "[2] перевод\n"
        "[3] перевод\n\n"
        "{texts}"
    ).format(target_language=target_language, texts="\n".join(numbered_texts))

    try:
        translated_response = await ai_service.translate_text(
            text=prompt,
            source_language=source_language,
            target_language=target_language,
        )
    except Exception:
        logger.exception(
            "Failed to translate subtitle batch from %s to %s",
            source_language,
            target_language,
        )
        raise

    translations: dict[int, str] = {}
    for match in re.finditer(
        r"\[(\d+)\]\s*(.*?)(?=(?:\n\[\d+\]\s)|\Z)",
        translated_response.strip(),
        flags=re.DOTALL,
    ):
        index = int(match.group(1))
        text = match.group(2).strip()
        translations[index] = text

    translated_segments: list[SubtitleSegment] = []
    for idx, segment in enumerate(segments, start=1):
        translated_text = translations.get(idx, segment.text)
        translated_segments.append(
            SubtitleSegment(start=segment.start, end=segment.end, text=translated_text)
        )

    return translated_segments

from uuid import uuid4
import asyncio

async def burn_subtitles(video_path: Path, srt_content: str) -> Path:
    subtitles_path = TEMP_DIR / f"subs_{uuid4().hex}.srt"
    output_path = TEMP_DIR / f"out_{uuid4().hex}.mp4"

    logger.info("Writing subtitles to %s", subtitles_path)
    subtitles_path.write_text(srt_content, encoding="utf-8")

    cmd = (
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"subtitles={subtitles_path.as_posix()}",
        "-c:v",
        "libx264",          # ✅ ЖЁСТКО фиксируем кодек
        "-preset",
        "veryfast",        # ✅ Ускоряем
        "-pix_fmt",
        "yuv420p",         # ✅ Совместимость с Telegram
        "-c:a",
        "aac",             # ✅ Аудио перекодируем (НЕ copy!)
        str(output_path),
    )

    logger.info("Starting ffmpeg burn: %s", output_path)

    try:
        stdout, stderr, returncode = await _run_subprocess(
            *cmd,
            timeout=180,     # ✅ ФАТАЛЬНО ВАЖНО: защита от зависания
        )

        if returncode != 0:
            logger.error("ffmpeg failed: %s", stderr or stdout)
            raise RuntimeError(stderr or stdout)

    except asyncio.TimeoutError:
        logger.error("ffmpeg timeout exceeded, killing process")
        raise RuntimeError("ffmpeg timed out while burning subtitles")

    finally:
        try:
            subtitles_path.unlink(missing_ok=True)
        except OSError:
            pass

    logger.info("Subtitled video successfully created: %s", output_path)
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
