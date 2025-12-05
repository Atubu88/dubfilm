from dataclasses import dataclass
from typing import Any

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message

from ai.service import AIService
from config import DEFAULT_TRANSLATION_CHOICES
from pipelines.summary import run_summary
from pipelines.transcribe import run_transcription
from pipelines.translate import run_translation
from services.audio import prepare_audio_file

router = Router()


class TranslationState(StatesGroup):
    waiting_for_language = State()


@dataclass
class TranscriptionResult:
    text: str
    language: str


# ✅ ИСПРАВЛЕНО: доступ через атрибут, а не через []
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
    await message.answer(
        "Готово! Я определил язык: {lang}. На какой язык перевести?\nВарианты: {options}\n"
        "Можно выбрать любой другой — просто напиши название языка."
        .format(lang=transcription.language.title(), options=options)
    )


def _is_supported_document(message: Message) -> bool:
    if not message.document:
        return False
    mime = message.document.mime_type or ""
    return mime.startswith("audio/") or mime.startswith("video/")


@router.message(F.audio | F.voice | F.video | F.video_note | F.document)
async def handle_media(message: Message, state: FSMContext) -> None:
    if message.document and not _is_supported_document(message):
        await message.answer("Поддерживаются только аудио или видео документы.")
        return

    ai_service = await _get_ai_service(message)

    await message.answer("Скачиваю и обрабатываю файл, секунду...")
    audio_path = await prepare_audio_file(bot=message.bot, media=message)

    try:
        transcription_data = await run_transcription(audio_path=audio_path, ai_service=ai_service)
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass

    transcription = TranscriptionResult(
        text=transcription_data["text"],
        language=transcription_data["language"],
    )

    await _request_translation_language(message, transcription, state)


@router.message(TranslationState.waiting_for_language)
async def handle_translation_request(message: Message, state: FSMContext) -> None:
    ai_service = await _get_ai_service(message)
    data: dict[str, Any] = await state.get_data()

    target_language = message.text.strip()
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

    response = (
        "🗣 Оригинал ({src}):\n{orig}\n\n"
        "🌍 Перевод ({target}):\n{translated}\n\n"
        "✍️ Кратко: {summary}"
    ).format(
        src=detected_language.title(),
        orig=original_text,
        target=target_language.title(),
        translated=translation,
        summary=summary_text,
    )

    await _send_long_message(message, response)
    await state.clear()
