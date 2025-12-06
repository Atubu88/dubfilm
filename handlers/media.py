import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from ai.service import AIService
from config import DEFAULT_TRANSLATION_CHOICES
from pipelines.summary import run_summary
from pipelines.transcribe import run_transcription
from pipelines.translate import run_translation
from handlers.subtitles import SubtitleState
from services.audio import MAX_FILE_SIZE_BYTES, get_media_size, prepare_audio_file
from services.downloader import download_audio_from_url, is_supported_media_url

router = Router()
logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"(https?://\S+)", re.IGNORECASE)


class TranslationState(StatesGroup):
    waiting_for_language = State()


@dataclass
class TranscriptionResult:
    text: str
    language: str


LANG_MAP = {
    "English": "english",
    "Arabic": "arabic",
    "Uzbek": "uzbek",
    "Russian": "russian",
}


async def _get_ai_service(message: Message) -> AIService:
    ai_service: AIService = message.bot.ai_service
    return ai_service


async def _send_long_message(message: Message, text: str, chunk_size: int = 3900) -> None:
    if len(text) <= chunk_size:
        await message.answer(text)
        return

    for start in range(0, len(text), chunk_size):
        await message.answer(text[start:start + chunk_size])


async def _request_translation_language(message: Message, transcription: TranscriptionResult, state: FSMContext) -> None:
    options = ", ".join(DEFAULT_TRANSLATION_CHOICES)
    await state.update_data(text=transcription.text, language=transcription.language)
    await state.set_state(TranslationState.waiting_for_language)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=choice,
                    callback_data=f"translation:{LANG_MAP[choice]}",
                )
            ]
            for choice in DEFAULT_TRANSLATION_CHOICES
        ]
    )
    await message.answer(
        (
            "Готово! Я определил язык: {lang}. На какой язык перевести?\n"
            "Варианты: {options}\n"
            "Выбирай язык перевода кнопкой ниже."
        ).format(lang=transcription.language.title(), options=options),
        reply_markup=keyboard,
    )


def _is_supported_document(message: Message) -> bool:
    if not message.document:
        return False
    mime = message.document.mime_type or ""
    return mime.startswith("audio/") or mime.startswith("video/")


def _extract_supported_url(text: str) -> str | None:
    for match in URL_PATTERN.finditer(text):
        url = match.group(1)
        if is_supported_media_url(url):
            return url
    return None


async def _process_audio(
    message: Message, state: FSMContext, ai_service: AIService, audio_path: Path
) -> None:
    try:
        transcription_data = await run_transcription(audio_path=audio_path, ai_service=ai_service)
    except Exception:
        logger.exception("Failed to transcribe audio %s", audio_path)
        await message.answer("Не удалось обработать аудио. Попробуй ещё раз или позже.")
        return
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass

        parent = audio_path.parent
        if parent.name.startswith("download_"):
            try:
                parent.rmdir()
            except OSError:
                pass

    transcription = TranscriptionResult(
        text=transcription_data["text"],
        language=transcription_data["language"],
    )

    await _request_translation_language(message, transcription, state)


async def _translate_and_summarize(
    message: Message, state: FSMContext, target_language: str
) -> None:
    ai_service = await _get_ai_service(message)
    data: dict[str, Any] = await state.get_data()

    original_text = data.get("text", "")
    detected_language = data.get("language", "unknown")

    if not original_text:
        await state.clear()
        await message.answer("Не нашёл текст для перевода, пришли аудио/видео заново.")
        return

    await message.answer("Перевожу и готовлю краткое резюме...")

    translation = await run_translation(
        text=original_text,
        source_language=detected_language,
        target_language=target_language,
        ai_service=ai_service,
    )

    summary_text = await run_summary(
        original_text=original_text,
        translated_text=translation,
        target_language=target_language,
        ai_service=ai_service,
    )

    # ✅ ИЗМЕНЁН ТОЛЬКО ФОРМАТ СООБЩЕНИЯ
    response = (
        "📝 Суть видео:\n\n"
        "{summary}\n\n"
        "━━━━━━━━━━━━━━\n"
        "🗣 Оригинал ({src}):\n{orig}\n\n"
        "🌍 Перевод ({target}):\n{translated}"
    ).format(
        summary=summary_text,
        src=detected_language.title(),
        orig=original_text,
        target=target_language.title(),
        translated=translation,
    )

    await _send_long_message(message, response)
    await state.clear()


@router.message(F.audio | F.voice | F.video | F.video_note | F.document)
async def handle_media(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state in {
        SubtitleState.waiting_for_video.state,
        SubtitleState.choosing_subtitle_language.state,
        SubtitleState.generating.state,
        SubtitleState.sending_result.state,
    }:
        return

    data = await state.get_data()
    if data.get("processing"):
        await message.answer("Я уже обрабатываю предыдущий запрос, подожди, пожалуйста.")
        return

    if message.document and not _is_supported_document(message):
        await message.answer("Поддерживаются только аудио или видео документы.")
        return

    file_size = get_media_size(message)
    if file_size is not None and file_size > MAX_FILE_SIZE_BYTES:
        await message.answer("Файл слишком большой. Максимальный размер — 20 МБ.")
        return

    ai_service = await _get_ai_service(message)

    await state.update_data(processing=True)
    try:
        await message.answer("Скачиваю и обрабатываю файл, секунду...")
        audio_path = await prepare_audio_file(bot=message.bot, media=message)
        await _process_audio(message, state, ai_service, audio_path)
    except Exception:
        logger.exception(
            "Failed to process uploaded media from user %s",
            message.from_user.id if message.from_user else "unknown"
        )
        await message.answer("Не удалось скачать или обработать файл. Проверь его и попробуй снова чуть позже.")
    finally:
        await state.update_data(processing=False)


@router.message(F.text.regexp(URL_PATTERN))
async def handle_media_links(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state in {
        SubtitleState.waiting_for_video.state,
        SubtitleState.choosing_subtitle_language.state,
        SubtitleState.generating.state,
        SubtitleState.sending_result.state,
    }:
        return

    data = await state.get_data()
    if data.get("processing"):
        return

    url = _extract_supported_url(message.text or "")
    if not url:
        return

    ai_service = await _get_ai_service(message)

    await state.update_data(processing=True)
    try:
        await message.answer("Скачиваю медиа по ссылке, секунду...")
        audio_path = await download_audio_from_url(url)
        await _process_audio(message, state, ai_service, audio_path)
    except Exception:
        logger.exception("Failed to download media from %s", url)
        await message.answer("Не удалось скачать или обработать ссылку. Проверь её и попробуй снова.")
    finally:
        await state.update_data(processing=False)


@router.message(TranslationState.waiting_for_language)
async def handle_translation_request(message: Message, state: FSMContext) -> None:
    await message.answer("Пожалуйста, выбери язык кнопкой ниже ⬇️")


@router.callback_query(TranslationState.waiting_for_language, F.data.startswith("translation:"))
async def handle_translation_button(callback: CallbackQuery, state: FSMContext) -> None:
    target_language = callback.data.split(":", 1)[1].title()
    await callback.answer()
    if callback.message:
        await _translate_and_summarize(callback.message, state, target_language)
