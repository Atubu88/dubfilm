import asyncio
import logging
import re
from pathlib import Path
from uuid import uuid4

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, Message

from ai.service import AIService
from config import DEFAULT_TRANSLATION_CHOICES, ENABLE_DUB_FLOW, TEMP_DIR, TELEGRAM_VIDEO_UPLOAD_TIMEOUT
from pipelines.dub import run_dub_pipeline
from services.audio import MAX_FILE_SIZE_BYTES
from services.downloader import is_supported_media_url
from services.subtitles import _run_subprocess, download_video_from_url
from services.video_duration import validate_video_duration

router = Router()
logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"(https?://\S+)", re.IGNORECASE)


class DubState(StatesGroup):
    waiting_for_video = State()
    choosing_language = State()
    generating = State()


LANG_MAP = {choice: choice.lower() for choice in DEFAULT_TRANSLATION_CHOICES}


def _extract_supported_url(text: str) -> str | None:
    for match in URL_PATTERN.finditer(text):
        url = match.group(1)
        if is_supported_media_url(url):
            return url
    return None


def _is_video_document(message: Message) -> bool:
    if message.video:
        return True
    if not message.document:
        return False
    mime = message.document.mime_type or ""
    return mime.startswith("video/")


async def _download_video_file(bot, file_id: str, suffix: str) -> Path:
    destination = TEMP_DIR / f"dub_{uuid4()}{suffix}"
    file = await bot.get_file(file_id)
    destination.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            await bot.download_file(file.file_path, destination)
            return destination
        except Exception as exc:
            last_error = exc
            logger.warning("Dub download attempt %d/3 failed for %s: %s", attempt, file_id, exc)
            await asyncio.sleep(1.2 * attempt)

    raise RuntimeError(f"Failed to download video after retries: {last_error}")


async def _prepare_video_from_message(message: Message) -> Path:
    if message.video:
        suffix = Path(message.video.file_name or "video").suffix or ".mp4"
        file_id = message.video.file_id
        file_size = message.video.file_size
    elif message.document and _is_video_document(message):
        suffix = Path(message.document.file_name or "video").suffix or ".mp4"
        file_id = message.document.file_id
        file_size = message.document.file_size
    else:
        raise ValueError("Поддерживается только видео файл или ссылка на видео")

    if file_size and file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError("Файл слишком большой. Максимум 20 МБ.")

    video_path = await _download_video_file(message.bot, file_id, suffix)
    await validate_video_duration(video_path)
    return video_path


async def _compress_for_telegram(input_path: Path) -> Path | None:
    """Try to compress oversized video under Telegram bot-safe limit."""
    size_mb = input_path.stat().st_size / (1024 * 1024)
    if size_mb <= 49:
        return input_path

    attempts = [
        ("720", "28", "128k"),
        ("540", "30", "96k"),
    ]

    for idx, (width, crf, audio_bitrate) in enumerate(attempts, start=1):
        out_path = TEMP_DIR / f"dub_send_{idx}_{uuid4().hex}.mp4"
        stdout, stderr, code = await _run_subprocess(
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            f"scale='min({width},iw)':-2:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            crf,
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-movflags",
            "+faststart",
            str(out_path),
            timeout=600,
        )
        if code != 0:
            logger.warning("Dub compress attempt %d failed: %s", idx, stderr or stdout)
            out_path.unlink(missing_ok=True)
            continue

        out_size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info("Dub compress attempt %d result size: %.2f MB", idx, out_size_mb)
        if out_size_mb <= 49:
            return out_path

    return None


async def _ask_language_choice(message: Message, state: FSMContext, video_path: Path) -> None:
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=choice, callback_data=f"dub_lang:{LANG_MAP[choice]}")]
            for choice in DEFAULT_TRANSLATION_CHOICES
        ]
    )
    await state.update_data(video_path=str(video_path))
    await state.set_state(DubState.choosing_language)
    await message.answer("Выбери язык дубляжа:", reply_markup=keyboard)


async def start_dub(message: Message, state: FSMContext) -> None:
    if not ENABLE_DUB_FLOW:
        await message.answer("Флоу дубляжа временно выключен")
        return

    await state.clear()
    await state.set_state(DubState.waiting_for_video)
    await message.answer("Пришли видео или ссылку. Я верну видео с переводом поверх оригинала.")


@router.message(Command("dub"))
async def cmd_dub(message: Message, state: FSMContext) -> None:
    await start_dub(message, state)


@router.message(DubState.waiting_for_video, F.text)
async def handle_dub_link(message: Message, state: FSMContext) -> None:
    url = _extract_supported_url(message.text or "")
    if not url:
        return

    await message.answer("Скачиваю видео по ссылке...")
    try:
        video_path = await download_video_from_url(url)
    except Exception:
        logger.exception("Failed to download video for dub from %s", url)
        await message.answer("Не удалось скачать видео по ссылке")
        return

    await _ask_language_choice(message, state, video_path)


@router.message(DubState.waiting_for_video, F.video | F.document)
async def handle_dub_upload(message: Message, state: FSMContext) -> None:
    try:
        video_path = await _prepare_video_from_message(message)
    except Exception as exc:
        await message.answer(str(exc))
        return

    await _ask_language_choice(message, state, video_path)


@router.callback_query(DubState.choosing_language, F.data.startswith("dub_lang:"))
async def handle_dub_lang(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    target_language = callback.data.split(":", 1)[1].title()
    data = await state.get_data()
    video_path_str = data.get("video_path")

    if not video_path_str:
        await state.clear()
        if callback.message:
            await callback.message.answer("Не нашёл видео. Пришли заново.")
        return

    if callback.message:
        await callback.message.answer("Генерирую дубляж, подожди немного...")

    await state.set_state(DubState.generating)
    ai_service: AIService = callback.message.bot.ai_service

    try:
        result_path = await run_dub_pipeline(Path(video_path_str), target_language, ai_service)
    except Exception:
        logger.exception("Dub pipeline failed for %s", video_path_str)
        if callback.message:
            await callback.message.answer("Не удалось сделать дубляж. Попробуй позже.")
        await state.clear()
        return

    compressed_path: Path | None = None
    try:
        if callback.message:
            # Telegram Bot API often rejects oversized files with 413 Request Entity Too Large.
            # Try auto-compress before giving up.
            file_size_mb = result_path.stat().st_size / (1024 * 1024)
            logger.info("Dub result size: %.2f MB (%s)", file_size_mb, result_path)

            send_path = result_path
            if file_size_mb > 49:
                compressed_path = await _compress_for_telegram(result_path)
                if compressed_path is None:
                    await callback.message.answer(
                        f"Итоговое видео слишком большое для отправки ботом ({file_size_mb:.1f} MB). "
                        "Даже после сжатия не уложилось в лимит."
                    )
                    return
                send_path = compressed_path
                send_size_mb = send_path.stat().st_size / (1024 * 1024)
                logger.info("Dub compressed result size: %.2f MB (%s)", send_size_mb, send_path)

            last_error: Exception | None = None
            too_large_error = False
            for attempt in range(1, 4):
                try:
                    video_file = FSInputFile(send_path)
                    await callback.message.answer_video(
                        video_file,
                        caption="Готово! Видео с переводом поверх оригинала.",
                        request_timeout=max(900, int(TELEGRAM_VIDEO_UPLOAD_TIMEOUT * attempt * 2)),
                    )
                    logger.info("VIDEO_DONE mode=dub send=video path=%s", send_path)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if "Request Entity Too Large" in str(exc):
                        too_large_error = True
                    logger.warning(
                        "Dub send video attempt %d/3 failed for %s: %s",
                        attempt,
                        send_path,
                        exc,
                    )
                    if too_large_error:
                        break
                    await asyncio.sleep(1.5 * attempt)

            if last_error is not None and not too_large_error:
                # Fallback: document send is often more stable on large files / slow links
                try:
                    doc_file = FSInputFile(send_path)
                    await callback.message.answer_document(
                        doc_file,
                        caption="Готово! Видео с переводом (документ).",
                        request_timeout=max(1200, int(TELEGRAM_VIDEO_UPLOAD_TIMEOUT * 4)),
                    )
                    logger.info("VIDEO_DONE mode=dub send=document path=%s", send_path)
                    last_error = None
                except Exception as doc_exc:
                    logger.warning("Dub document fallback failed for %s: %s", send_path, doc_exc)

            if last_error is not None:
                if too_large_error:
                    await callback.message.answer(
                        "Не удалось отправить: файл слишком большой для Telegram Bot API."
                    )
                raise last_error
    except Exception:
        logger.exception("Failed to send dubbed video %s", result_path)
        if callback.message:
            await callback.message.answer("Не удалось отправить итоговое видео.")
    finally:
        await state.clear()
        try:
            Path(video_path_str).unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
            if compressed_path and compressed_path != result_path:
                compressed_path.unlink(missing_ok=True)
        except OSError:
            pass
