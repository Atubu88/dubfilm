import logging
from pathlib import Path

from ai.service import AIService
from services.subtitles import extract_audio_from_video, transcribe_segments, translate_segments
from services.dub import (
    compose_dubbed_video_from_segments,
    constrain_translated_segments,
    normalize_segments_for_dub,
    stabilize_speakers,
    synthesize_segment_audios,
)
from services.video_duration import validate_video_duration

logger = logging.getLogger(__name__)


async def run_dub_pipeline(video_path: Path, target_language: str, ai_service: AIService) -> Path:
    logger.info("Starting dub pipeline for %s to %s", video_path, target_language)
    await validate_video_duration(video_path)

    audio_path: Path | None = None
    tts_audio_items: list[tuple] = []

    try:
        audio_path = await extract_audio_from_video(video_path)
        segments, detected_language = await transcribe_segments(audio_path, ai_service)
        logger.info(
            "DubDiag: provider=whisper detected_language=%s segments=%d first_seg_start=%.3f first_seg_end=%.3f",
            detected_language,
            len(segments),
            (segments[0].start if segments else 0.0),
            (segments[0].end if segments else 0.0),
        )

        normalized_segments = normalize_segments_for_dub(segments)
        logger.info("DubDiag: normalize segments before translate: %d -> %d", len(segments), len(normalized_segments))

        stabilized_segments = stabilize_speakers(normalized_segments)

        translated_segments = await translate_segments(
            segments=stabilized_segments,
            source_language=detected_language,
            target_language=target_language,
            ai_service=ai_service,
        )

        translated_segments = await constrain_translated_segments(
            translated_segments,
            target_language=target_language,
            ai_service=ai_service,
        )

        logger.info("DubDiag: translated_segments=%d", len(translated_segments))
        tts_audio_items = await synthesize_segment_audios(
            ai_service,
            translated_segments,
            target_language=target_language,
        )
        out_video = await compose_dubbed_video_from_segments(video_path, tts_audio_items)
        logger.info("Dubbed video created: %s", out_video)
        return out_video

    finally:
        if audio_path:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
        if tts_audio_items:
            for _seg, tts_path, _dur in tts_audio_items:
                try:
                    Path(tts_path).unlink(missing_ok=True)
                except OSError:
                    pass
