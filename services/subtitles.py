import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ai.service import AIService
from config import TEMP_DIR
from services.downloader import is_supported_media_url
from services.video_duration import validate_video_duration
import json

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

    if len(segments) > 200:
        raise RuntimeError("Too many subtitle segments for batch translation")

    numbered_texts: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        segment_text = (segment.text or "").strip()
        # ✅ защита от квадратных скобок
        segment_text = segment_text.replace("[", "(").replace("]", ")")
        numbered_texts.append(f"[{idx}] {segment_text}")

    prompt = (
        f"Переведи каждый пункт списка с {source_language} на {target_language}.\n"
        "Сохрани нумерацию и порядок.\n"
        "Не добавляй комментариев.\n"
        "Не объединяй строки.\n"
        "Формат ответа строго:\n\n"
        "[1] перевод\n"
        "[2] перевод\n"
        "[3] перевод\n\n"
        f"{chr(10).join(numbered_texts)}"
    )

    try:
        translated_response = await ai_service.translate_text(
            text=prompt,
            source_language="auto",   # ✅ ВАЖНО
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

    if not translations:
        raise RuntimeError("Batch translation returned empty result")

    translated_segments: list[SubtitleSegment] = []
    for idx, segment in enumerate(segments, start=1):
        translated_text = translations.get(idx, segment.text)
        translated_segments.append(
            SubtitleSegment(
                start=segment.start,
                end=segment.end,
                text=translated_text,
            )
        )

    return translated_segments

async def get_video_resolution(video_path: Path) -> tuple[int, int]:
    stdout, stderr, returncode = await _run_subprocess(
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path),
    )

    if returncode != 0:
        raise RuntimeError(stderr or stdout)

    width, height = map(int, stdout.strip().split(","))
    return width, height



async def burn_subtitles(video_path: Path, srt_content: str) -> Path:
    subtitles_path = TEMP_DIR / f"subs_{uuid4().hex}.srt"
    output_path = TEMP_DIR / f"out_{uuid4().hex}.mp4"

    logger.info("Writing subtitles to %s", subtitles_path)
    subtitles_path.write_text(srt_content, encoding="utf-8")

    # 🔥 1. Узнаём разрешение видео
    width, height = await get_video_resolution(video_path)

    # 🔥 2. Динамический размер шрифта
    if height > width:  # вертикальное
        fontsize = int(height * 0.025)
    else:
        fontsize = int(height * 0.035)

    # 🔥 Ограничения, чтобы не было слишком большого текста
    fontsize = max(18, min(fontsize, 60))  # от 18 до 60 пикселей

    # 🔥 3. Динамическая обводка
    outline = max(1, fontsize // 12)

    logger.info(f"Dynamic subtitle style: fontsize={fontsize}, outline={outline}")

    # 🔥 4. FFmpeg команда
    cmd = (
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"subtitles={subtitles_path.as_posix()}:"
        f"force_style='Fontsize={fontsize},Outline={outline},Shadow=1,"
        "PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&'",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(output_path),
    )

    logger.info("Starting ffmpeg burn")

    try:
        stdout, stderr, returncode = await _run_subprocess(*cmd, timeout=180)

        if returncode != 0:
            raise RuntimeError(stderr or stdout)

    except asyncio.TimeoutError:
        raise RuntimeError("ffmpeg timed out")
    finally:
        subtitles_path.unlink(missing_ok=True)

    logger.info("Video created: %s", output_path)
    return output_path


async def download_video_from_url(url: str) -> Path:
    if not is_supported_media_url(url):
        raise ValueError("Unsupported media URL")

    # 1️⃣ СНАЧАЛА получаем метаданные, чтобы быстро узнать длительность
    stdout, stderr, returncode = await _run_subprocess(
        "yt-dlp",
        "--dump-json",
        url,
        timeout=20,
    )

    if returncode != 0:
        raise RuntimeError(stderr or stdout)

    info = json.loads(stdout)
    duration = float(info.get("duration") or 0)

    # 2️⃣ Если более 5 минут → НЕ СКАЧИВАЕМ видео
    if duration > 300:
        raise ValueError("Video too long")

    # 3️⃣ Теперь скачиваем, раз знаем, что не длинное
    download_dir = TEMP_DIR / f"video_{uuid4()}"
    download_dir.mkdir(parents=True, exist_ok=True)
    output_template = download_dir / "%(title)s.%(ext)s"

    stdout, stderr, returncode = await _run_subprocess(
        "yt-dlp",
        "-f",
        "bv*+ba/b",
        "-o",
        str(output_template),
        url,
    )

    if returncode != 0:
        raise RuntimeError(stderr or stdout)

    files = sorted(download_dir.glob("*"))
    if not files:
        raise FileNotFoundError("yt-dlp produced no files")

    return max(files, key=lambda p: p.stat().st_size)