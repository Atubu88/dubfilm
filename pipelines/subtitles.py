import logging
from pathlib import Path

from ai.service import AIService
from services.subtitles import (
    build_srt_content,
    burn_subtitles,
    extract_audio_from_video,
    transcribe_segments,
    translate_segments,
)
from services.video_duration import validate_video_duration

logger = logging.getLogger(__name__)


async def run_subtitles_pipeline(
    video_path: Path,
    target_language: str,
    ai_service: AIService,
) -> Path:
    logger.info("Starting subtitles pipeline for %s to %s", video_path, target_language)
    await validate_video_duration(video_path)
    audio_path: Path | None = None
    try:
        audio_path = await extract_audio_from_video(video_path)
        segments, detected_language = await transcribe_segments(audio_path, ai_service)
        translated_segments = await translate_segments(
            segments=segments,
            source_language=detected_language,
            target_language=target_language,
            ai_service=ai_service,
        )
        srt_content = build_srt_content(translated_segments)
        output_video = await burn_subtitles(video_path, srt_content)
        logger.info("Subtitled video created: %s", output_video)
        return output_video
    finally:
        if audio_path:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to delete temporary audio %s", audio_path)
        parent = video_path.parent
        if parent.name.startswith("video_"):
            try:
                video_path.unlink(missing_ok=True)
                parent.rmdir()
            except OSError:
                logger.debug("Temporary video directory %s not removed", parent)
