import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ai.service import AIService
from config import (
    TEMP_DIR,
    FFSUBSYNC_VAD,
    FFSUBSYNC_MAX_OFFSET_SECONDS,
    FFSUBSYNC_USE_GSS,
    FFSUBSYNC_NO_FIX_FRAMERATE,
    FFSUBSYNC_MAX_ACCEPTED_OFFSET_SECONDS,
    GLOSSARY_ENABLED,
    GLOSSARY_PATH,
    GLOSSARY_SKIP_QURAN_AYAHS,
    ISLAMIC_TRANSLATION_MODE,
)
from services.downloader import is_supported_media_url
from services.video_duration import validate_video_duration
import json

logger = logging.getLogger(__name__)


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


_MEANINGFUL_TEXT_RE = re.compile(r"[A-Za-zА-Яа-я0-9]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def _load_glossary_map() -> tuple[dict[str, dict], dict[str, str]]:
    if not GLOSSARY_ENABLED:
        return {}, {}
    try:
        data = json.loads(Path(GLOSSARY_PATH).read_text(encoding="utf-8"))
        entries = data.get("entries", []) if isinstance(data, dict) else []
        by_term: dict[str, dict] = {}
        by_term_lc: dict[str, str] = {}
        for e in entries:
            term = (e.get("term") or "").strip()
            if not term:
                continue
            by_term[term] = e
            by_term_lc[term.lower()] = term
        return by_term, by_term_lc
    except Exception as exc:
        logger.warning("Glossary load failed (%s): %s", GLOSSARY_PATH, exc)
        return {}, {}


def _apply_glossary_to_text(text: str, glossary_by_term: dict[str, dict], glossary_by_term_lc: dict[str, str]) -> str:
    out = text
    for term, entry in glossary_by_term.items():
        preferred = (entry.get("preferred") or "").strip()
        if not preferred:
            continue
        # Replace case-insensitively with word boundaries.
        out = re.sub(rf"\b{re.escape(term)}\b", preferred, out, flags=re.IGNORECASE)
        for bad in entry.get("forbidden", []) or []:
            bad_s = (bad or "").strip()
            if bad_s:
                out = re.sub(rf"\b{re.escape(bad_s)}\b", preferred, out, flags=re.IGNORECASE)
    return out


def _is_likely_quran_ayah(segment_text: str) -> bool:
    t = (segment_text or "").strip()
    if not t:
        return False
    if not _ARABIC_RE.search(t):
        return False
    # Conservative heuristic: mark only explicit Quran indicators.
    # Do NOT classify plain Arabic speech as ayah.
    hints = ("﷽", "بسم الله", "قال الله", "صدق الله", "سورة", "آية", "القرآن")
    return any(h in t for h in hints)


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
                segments.append(
                    SubtitleSegment(
                        start=start,
                        end=end,
                        text=text,
                        speaker=(seg.get("speaker") if isinstance(seg, dict) else None),
                    )
                )
        except Exception:
            logger.exception("Failed to parse segment %s", seg)

    if not segments and result.get("text"):
        segments.append(SubtitleSegment(start=0.0, end=0.0, text=result["text"]))

    if not segments:
        raise RuntimeError("No transcription segments produced")

    return segments, language


def apply_time_offset(segments: list[SubtitleSegment], offset: float) -> list[SubtitleSegment]:
    if offset <= 0:
        return segments

    return shift_segments(segments, offset)


def shift_segments(segments: list[SubtitleSegment], offset: float) -> list[SubtitleSegment]:
    if offset == 0:
        return segments

    adjusted_segments: list[SubtitleSegment] = []
    for segment in segments:
        start = max(0.0, segment.start + offset)
        end = max(start, segment.end + offset)
        adjusted_segments.append(SubtitleSegment(start=start, end=end, text=segment.text, speaker=segment.speaker))
    return adjusted_segments


def normalize_segments_by_speech_start(
    segments: list[SubtitleSegment],
    *,
    min_start_seconds: float = 1.0,
    min_meaningful_chars: int = 3,
) -> list[SubtitleSegment]:
    speech_start = find_first_meaningful_segment_start(
        segments,
        min_meaningful_chars=min_meaningful_chars,
    )

    if speech_start is None or speech_start < min_start_seconds:
        return segments

    logger.info(
        "Normalizing subtitles by speech start at %.2fs (threshold %.2fs)",
        speech_start,
        min_start_seconds,
    )
    return shift_segments(segments, -speech_start)


def find_first_meaningful_segment_start(
    segments: list[SubtitleSegment],
    *,
    min_meaningful_chars: int = 3,
) -> float | None:
    for segment in segments:
        if _is_meaningful_text(segment.text, min_meaningful_chars=min_meaningful_chars):
            return max(0.0, segment.start)
    return None


def _is_meaningful_text(text: str, *, min_meaningful_chars: int) -> bool:
    if not text:
        return False
    meaningful_chars = _MEANINGFUL_TEXT_RE.findall(text)
    return len(meaningful_chars) >= min_meaningful_chars


async def detect_first_speech_start(
    audio_path: Path,
    noise_db: float = -40.0,
    min_silence_duration: float = 0.20,
) -> float:
    """Detect first non-silent moment in extracted audio using ffmpeg silencedetect.
    Returns seconds from start.
    """
    stdout, stderr, returncode = await _run_subprocess(
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_duration}",
        "-f",
        "null",
        "-",
    )

    if returncode != 0:
        # ffmpeg silencedetect often writes to stderr but may still return 0.
        # If non-zero, do a safe fallback.
        logger.warning("silencedetect failed, fallback to 0.0: %s", stderr or stdout)
        return 0.0

    output = (stderr or "") + "\n" + (stdout or "")
    # We care about initial silence_end if present.
    m = re.search(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)", output)
    if m:
        try:
            return max(0.0, float(m.group(1)))
        except ValueError:
            return 0.0

    # If no silence markers found, assume speech starts near 0.
    return 0.0


async def get_audio_start_offset(video_path: Path) -> float:
    stdout, stderr, returncode = await _run_subprocess(
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=start_time",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    )

    if returncode != 0:
        raise RuntimeError(stderr or stdout)

    raw_value = stdout.strip()
    if not raw_value:
        return 0.0

    try:
        offset = float(raw_value)
    except ValueError:
        logger.warning("Unexpected audio start_time value: %s", raw_value)
        return 0.0

    if offset < 0:
        logger.warning("Audio start_time is negative (%.3f); ignoring.", offset)
        return 0.0

    return offset

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


def _wrap_subtitle_text(text: str, max_chars_per_line: int = 24, max_lines: int = 3) -> str:
    """Wrap text without dropping words.
    If lines exceed max_lines, caller should split segment in time.
    """
    words = (text or "").strip().split()
    if not words:
        return ""

    lines: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join(current + [word]).strip()
        if len(candidate) <= max_chars_per_line:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return "\n".join(lines)


def _split_text_for_subtitles(text: str, max_chars_per_line: int = 24, max_lines: int = 3) -> list[str]:
    """Split long text into multiple subtitle chunks so nothing is lost.
    Each chunk fits into max_lines * max_chars_per_line envelope.
    """
    words = (text or "").strip().split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []

    for word in words:
        test_words = current_words + [word]
        wrapped = _wrap_subtitle_text(" ".join(test_words), max_chars_per_line, max_lines)
        if wrapped and len(wrapped.splitlines()) <= max_lines:
            current_words.append(word)
        else:
            if current_words:
                chunks.append(" ".join(current_words))
            current_words = [word]

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _split_segment_to_fit(segment: SubtitleSegment, max_chars_per_line: int = 24, max_lines: int = 3) -> list[SubtitleSegment]:
    chunks = _split_text_for_subtitles(segment.text, max_chars_per_line, max_lines)
    if not chunks:
        return []
    if len(chunks) == 1:
        return [SubtitleSegment(start=segment.start, end=segment.end, text=chunks[0], speaker=segment.speaker)]

    duration = max(1.0, segment.end - segment.start)
    total_chars = sum(max(1, len(c)) for c in chunks)

    result: list[SubtitleSegment] = []
    cursor = segment.start
    for i, chunk in enumerate(chunks):
        ratio = max(1, len(chunk)) / total_chars
        chunk_duration = duration * ratio
        # keep tiny minimum readability window
        chunk_duration = max(0.8, chunk_duration)

        if i == len(chunks) - 1:
            end = max(cursor + 0.8, segment.end)
        else:
            end = min(segment.end, cursor + chunk_duration)

        result.append(SubtitleSegment(start=cursor, end=end, text=chunk, speaker=segment.speaker))
        cursor = end

    return result


def build_srt_content(segments: list[SubtitleSegment]) -> str:
    logger.info("Building SRT content for %d segments", len(segments))
    fitted_segments: list[SubtitleSegment] = []
    for segment in segments:
        fitted_segments.extend(_split_segment_to_fit(segment))

    lines: list[str] = []
    for idx, segment in enumerate(fitted_segments, start=1):
        start_ts = _format_timestamp(segment.start)
        end_ts = _format_timestamp(segment.end if segment.end > segment.start else segment.start + 1.2)
        wrapped_text = _wrap_subtitle_text(segment.text)
        lines.extend([str(idx), f"{start_ts} --> {end_ts}", wrapped_text, ""])
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

    glossary_by_term, glossary_by_term_lc = _load_glossary_map()

    numbered_texts: list[str] = []
    skip_translate_idx: set[int] = set()
    for idx, segment in enumerate(segments, start=1):
        segment_text = (segment.text or "").strip()

        # Skip Quran ayahs from translation if enabled.
        if GLOSSARY_SKIP_QURAN_AYAHS and _is_likely_quran_ayah(segment_text):
            skip_translate_idx.add(idx)

        # ✅ защита от квадратных скобок
        segment_text = segment_text.replace("[", "(").replace("]", ")")
        numbered_texts.append(f"[{idx}] {segment_text}")

    skip_line = ""
    if skip_translate_idx:
        skip_line = (
            "Строки с номерами "
            + ", ".join(str(i) for i in sorted(skip_translate_idx))
            + " НЕ переводи, верни их как есть (оригинал).\n"
        )

    glossary_line = ""
    if glossary_by_term:
        sample = []
        for term, entry in list(glossary_by_term.items())[:40]:
            pref = (entry.get("preferred") or "").strip()
            if pref:
                sample.append(f"{term} -> {pref}")
        if sample:
            glossary_line = "Используй глоссарий:\n" + "\n".join(sample) + "\n"

    islamic_rules = ""
    if ISLAMIC_TRANSLATION_MODE:
        islamic_rules = (
            "Ты переводчик исламского контента для мусульманской аудитории.\n"
            "Передавай смысл точно и уважительно, привычной исламской терминологией.\n"
            "Не добавляй толкования, комментарии и секулярные перефразы.\n"
            "Не используй сленг и иронию в религиозном контексте.\n"
        )

    prompt = (
        islamic_rules
        + f"Переведи каждый пункт списка с {source_language} на {target_language}.\n"
        "Сохрани нумерацию и порядок.\n"
        "Не добавляй комментариев.\n"
        "Не объединяй строки.\n"
        + skip_line
        + glossary_line
        + "Формат ответа строго:\n\n"
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
        if idx in skip_translate_idx:
            translated_text = segment.text
        else:
            translated_text = translations.get(idx, segment.text)

        if glossary_by_term:
            translated_text = _apply_glossary_to_text(translated_text, glossary_by_term, glossary_by_term_lc)

        translated_segments.append(
            SubtitleSegment(
                start=segment.start,
                end=segment.end,
                text=translated_text,
                speaker=segment.speaker,
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



async def sync_srt_with_ffsubsync(video_path: Path, srt_content: str) -> str:
    """Run ffsubsync to align subtitles to audio track.
    Returns synced SRT text; if ffsubsync unavailable/fails, returns original content.
    """
    src_srt = TEMP_DIR / f"presync_{uuid4().hex}.srt"
    out_srt = TEMP_DIR / f"synced_{uuid4().hex}.srt"
    src_srt.write_text(srt_content, encoding="utf-8")

    try:
        cmd = [
            "ffsubsync",
            str(video_path),
            "-i", str(src_srt),
            "-o", str(out_srt),
            "--max-offset-seconds", str(int(FFSUBSYNC_MAX_OFFSET_SECONDS)),
            "--vad", FFSUBSYNC_VAD,
        ]
        if FFSUBSYNC_USE_GSS:
            cmd.append("--gss")
        if FFSUBSYNC_NO_FIX_FRAMERATE:
            cmd.append("--no-fix-framerate")

        logger.info("SyncDiag: ffsubsync cmd=%s", " ".join(cmd))
        stdout, stderr, returncode = await _run_subprocess(*cmd, timeout=180)
        logger.info("SyncDiag: ffsubsync returncode=%s", returncode)
        # ffsubsync writes useful details to stderr; keep last lines for diagnostics
        tail = "\n".join((stderr or "").splitlines()[-8:])
        if tail:
            logger.info("SyncDiag: ffsubsync stderr tail:\n%s", tail)

        if returncode != 0 or not out_srt.exists():
            logger.warning("ffsubsync failed, using original subtitles: %s", stderr or stdout)
            return srt_content

        # Guardrail: reject suspiciously large offsets that usually indicate bad alignment match.
        offset_match = re.search(r"offset seconds:\s*([-+]?\d+(?:\.\d+)?)", stderr or "")
        if offset_match:
            offset_value = abs(float(offset_match.group(1)))
            if offset_value > FFSUBSYNC_MAX_ACCEPTED_OFFSET_SECONDS:
                logger.warning(
                    "ffsubsync offset %.3fs exceeds accepted threshold %.3fs; using original subtitles",
                    offset_value,
                    FFSUBSYNC_MAX_ACCEPTED_OFFSET_SECONDS,
                )
                return srt_content

        return out_srt.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        logger.warning("ffsubsync is not installed; using original subtitles")
        return srt_content
    except Exception as exc:
        logger.warning("ffsubsync error, using original subtitles: %s", exc)
        return srt_content
    finally:
        src_srt.unlink(missing_ok=True)
        out_srt.unlink(missing_ok=True)


async def burn_subtitles(video_path: Path, srt_content: str) -> Path:
    subtitles_path = TEMP_DIR / f"subs_{uuid4().hex}.srt"
    output_path = TEMP_DIR / f"out_{uuid4().hex}.mp4"

    logger.info("Writing subtitles to %s", subtitles_path)
    subtitles_path.write_text(srt_content, encoding="utf-8")

    # 🔥 1. Узнаём разрешение видео
    width, height = await get_video_resolution(video_path)

    # 🔥 2. Динамический размер шрифта
    # Используем меньшую сторону кадра, чтобы на вертикальных видео
    # текст не становился чрезмерно большим и не выходил за экран.
    base_size = min(width, height)
    # Robust size for social vertical formats: anchor to width, not min side.
    fontsize = int(width * 0.015)

    # Keep subtitle text compact and safe across 720p/1080p vertical videos.
    fontsize = max(11, min(fontsize, 22))

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
        f"force_style='Fontsize={fontsize},Outline={outline},Shadow=0,"
        "PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BackColour=&H80000000&,"
        "WrapStyle=2,Alignment=2,MarginV=96,MarginL=42,MarginR=42,BorderStyle=3'", 
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

    # 2️⃣ Лимит длительности отключен: скачиваем видео любой длины

    # 3️⃣ Теперь скачиваем
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
