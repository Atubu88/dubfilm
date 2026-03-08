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
from config import DEFAULT_TRANSLATION_CHOICES, TEMP_DIR, TELEGRAM_VIDEO_UPLOAD_TIMEOUT
from pipelines.subtitles import run_subtitles_pipeline
from services.audio import MAX_FILE_SIZE_BYTES
from services.downloader import is_supported_media_url
from services.subtitles import _run_subprocess, download_video_from_url
from services.video_duration import validate_video_duration

router = Router()
logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"(https?://\S+)", re.IGNORECASE)


class SubtitleState(StatesGroup):
    waiting_for_video = State()
    choosing_subtitle_language = State()
    generating = State()
    sending_result = State()


LANG_MAP = {choice: choice.lower() for choice in DEFAULT_TRANSLATION_CHOICES}


async def _get_ai_service(message: Message) -> AIService:
    return message.bot.ai_service


def _is_video_document(message: Message) -> bool:
    if message.video:
        return True
    if not message.document:
        return False
    mime = message.document.mime_type or ""
    return mime.startswith("video/")


def _extract_supported_url(text: str) -> str | None:
    for match in URL_PATTERN.finditer(text):
        url = match.group(1)
        if is_supported_media_url(url):
            return url
    return None


async def _download_video_file(bot, file_id: str, suffix: str) -> Path:
    destination = TEMP_DIR / f"subtitle_{uuid4()}{suffix}"
    file = await bot.get_file(file_id)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # 1) Primary path: aiogram helper with retries
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            await bot.download_file(
                file.file_path,
                destination,
            )
            return destination
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Video download attempt %d/3 failed for %s: %s",
                attempt,
                file_id,
                exc,
            )
            await asyncio.sleep(1.2 * attempt)

    # 2) Fallback path: direct Telegram file URL streaming
    # This helps when aiogram stream gets cancelled/timeouts on unstable links.
    import aiohttp

    file_url = f"https://api.telegram.org/file/bot{bot.token}/{file.file_path}"
    timeout = aiohttp.ClientTimeout(total=max(600, int(TELEGRAM_VIDEO_UPLOAD_TIMEOUT * 2)))

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(file_url) as resp:
                resp.raise_for_status()
                with destination.open("wb") as f:
                    async for chunk in resp.content.iter_chunked(1024 * 256):
                        if chunk:
                            f.write(chunk)
        return destination
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download video after retries (aiogram + fallback): {exc}; last_aiogram_error={last_error}"
        )


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
        raise ValueError("Unsupported media type for subtitles")

    if file_size and file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError("File exceeds maximum size")

    video_path = await _download_video_file(message.bot, file_id, suffix)

    try:
        await validate_video_duration(video_path)
    except Exception:
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to remove oversized video %s", video_path)
        raise

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
        out_path = TEMP_DIR / f"subs_send_{idx}_{uuid4().hex}.mp4"
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
            logger.warning("Subtitles compress attempt %d failed: %s", idx, stderr or stdout)
            out_path.unlink(missing_ok=True)
            continue

        out_size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info("Subtitles compress attempt %d result size: %.2f MB", idx, out_size_mb)
        if out_size_mb <= 49:
            return out_path

    return None


async def _ask_language_choice(message: Message, state: FSMContext, video_path: Path) -> None:
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=choice,
                    callback_data=f"subtitle_lang:{LANG_MAP[choice]}",
                )
            ]
            for choice in DEFAULT_TRANSLATION_CHOICES
        ]
    )
    await state.update_data(video_path=str(video_path))
    await state.set_state(SubtitleState.choosing_subtitle_language)
    await message.answer(
        "Видео получено! Выбери язык, на котором сделать субтитры:",
        reply_markup=keyboard,
    )


@router.message(Command("subtitles"))
async def start_subtitles(message: Message, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(SubtitleState.waiting_for_video)
    await message.answer(
        "Пришли видеофайл или ссылку на ролик, я извлеку аудио, переведу его и добавлю субтитры в видео."
    )


@router.message(SubtitleState.waiting_for_video, F.text)
async def handle_video_link(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    if data.get("processing"):
        await message.answer("Я уже обрабатываю предыдущий запрос, подожди чуть-чуть.")
        return

    url = _extract_supported_url(message.text or "")
    if not url:
        return

    await state.update_data(processing=True)
    await message.answer("Скачиваю видео по ссылке, подожди немного...")

    try:
        video_path = await download_video_from_url(url)

    except ValueError as exc:
        # ⛔️ Это НЕ ошибка — просто длинное видео
        if str(exc) == "Video too long":
            await message.answer("Видео слишком длинное. Максимальная длительность — 5 минут.")
        else:
            await message.answer("Не удалось скачать видео по ссылке. Попробуй другое или позже.")

        # ⚠️ НЕ ЛОГИРУЕМ exception — потому что это НЕ ошибка
        await state.update_data(processing=False)
        return

    except Exception:
        # ❌ Это настоящая ошибка (yt-dlp, ffmpeg, сеть)
        logger.exception("Failed to download video for subtitles from %s", url)
        await message.answer("Не удалось скачать видео по ссылке. Попробуй другое или позже.")
        await state.update_data(processing=False)
        return

    # ✔ всё хорошо → продолжаем
    await state.update_data(processing=False)
    await _ask_language_choice(message, state, video_path)



@router.message(SubtitleState.waiting_for_video, F.video | F.document)
async def handle_video_upload(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    if data.get("processing"):
        await message.answer("Я уже обрабатываю предыдущий запрос, подожди чуть-чуть.")
        return

    await state.update_data(processing=True)
    try:
        video_path = await _prepare_video_from_message(message)
    except ValueError as exc:
        if str(exc) == "Video too long":
            await message.answer("Видео слишком длинное. Максимальная длительность — 5 минут.")
        else:
            await message.answer(str(exc))
        await state.update_data(processing=False)
        return
    except Exception:
        logger.exception("Failed to download uploaded video for subtitles")
        await message.answer("Не удалось скачать видео. Попробуй позже или пришли другой файл.")
        await state.update_data(processing=False)
        return

    await state.update_data(processing=False)
    await _ask_language_choice(message, state, video_path)


@router.callback_query(
    SubtitleState.choosing_subtitle_language, F.data.startswith("subtitle_lang:")
)
async def handle_language_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    target_language = callback.data.split(":", 1)[1].title()
    data = await state.get_data()
    video_path_str = data.get("video_path")
    if not video_path_str:
        await state.clear()
        if callback.message:
            await callback.message.answer("Не нашёл видео. Пришли его заново.")
        return

    if callback.message:
        await callback.message.answer(
            "Генерирую субтитры — это может занять пару минут. Держись!"
        )

    await state.update_data(processing=True)
    await state.set_state(SubtitleState.generating)
    ai_service = await _get_ai_service(callback.message)

    try:
        result_path = await run_subtitles_pipeline(
            video_path=Path(video_path_str),
            target_language=target_language,
            ai_service=ai_service,
        )
    except Exception:
        logger.exception(
            "Failed to build subtitles for video %s", video_path_str
        )
        try:
            Path(video_path_str).unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to remove source video %s after error", video_path_str)
        await state.clear()
        if callback.message:
            await callback.message.answer(
                "Не получилось сгенерировать субтитры. Попробуй другой ролик или язык."
            )
        return

    await state.set_state(SubtitleState.sending_result)

    compressed_path: Path | None = None
    try:
        if callback.message:
            file_size_mb = result_path.stat().st_size / (1024 * 1024)
            logger.info("Subtitles result size: %.2f MB (%s)", file_size_mb, result_path)

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
                logger.info("Subtitles compressed result size: %.2f MB (%s)", send_size_mb, send_path)

            # Sending large videos can fail on unstable network (Broken pipe / ClientOSError).
            # Retry a few times with increasing timeout.
            last_error: Exception | None = None
            too_large_error = False
            for attempt in range(1, 4):
                try:
                    video_file = FSInputFile(send_path)
                    await callback.message.answer_video(
                        video_file,
                        caption="Готово! Держи видео с субтитрами.",
                        request_timeout=max(600, int(TELEGRAM_VIDEO_UPLOAD_TIMEOUT * attempt)),
                    )
                    logger.info("VIDEO_DONE mode=subtitles send=video path=%s", send_path)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if "Request Entity Too Large" in str(exc):
                        too_large_error = True
                    logger.warning(
                        "Send video attempt %d/3 failed for %s: %s",
                        attempt,
                        send_path,
                        exc,
                    )
                    if too_large_error:
                        break
                    await asyncio.sleep(1.5 * attempt)

            if last_error is not None and not too_large_error:
                # Fallback: send as document (often more stable than streamed video)
                try:
                    doc_file = FSInputFile(send_path)
                    await callback.message.answer_document(
                        doc_file,
                        caption="Готово! Видео с субтитрами (документ).",
                        request_timeout=max(900, int(TELEGRAM_VIDEO_UPLOAD_TIMEOUT * 3)),
                    )
                    logger.info("VIDEO_DONE mode=subtitles send=document path=%s", send_path)
                    last_error = None
                except Exception as doc_exc:
                    logger.warning("Document fallback failed for %s: %s", send_path, doc_exc)

            if last_error is not None:
                if too_large_error:
                    await callback.message.answer(
                        "Не удалось отправить: файл слишком большой для Telegram Bot API."
                    )
                raise last_error
    except Exception:
        logger.exception("Failed to send subtitled video %s", result_path)
        if callback.message:
            await callback.message.answer("Не удалось отправить итоговое видео. Попробуй ещё раз позже.")
    finally:
        try:
            Path(video_path_str).unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to remove source video %s", video_path_str)
        try:
            result_path.unlink(missing_ok=True)
            if compressed_path and compressed_path != result_path:
                compressed_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to remove subtitled video %s", result_path)
        await state.clear()
