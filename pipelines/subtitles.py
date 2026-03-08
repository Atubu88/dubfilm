import logging
from pathlib import Path

from ai.service import AIService
from config import (
    SUBTITLE_SYNC_MODE,
    SUBTITLE_EXTRA_DELAY_SECONDS,
    SUBTITLE_AUTO_MAX_SHIFT_SECONDS,
    SUBTITLE_ENABLE_FFSUBSYNC,
    TRANSCRIBE_PROVIDER,
)
from services.subtitles import (
    apply_time_offset,
    build_srt_content,
    burn_subtitles,
    extract_audio_from_video,
    detect_first_speech_start,
    get_audio_start_offset,
    shift_segments,
    sync_srt_with_ffsubsync,
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
        audio_offset = await get_audio_start_offset(video_path)
        audio_path = await extract_audio_from_video(video_path)
        segments, detected_language = await transcribe_segments(audio_path, ai_service)
        logger.info(
            "SyncDiag: provider=%s detected_language=%s segments=%d first_seg_start=%.3f first_seg_end=%.3f",
            TRANSCRIBE_PROVIDER,
            detected_language,
            len(segments),
            (segments[0].start if segments else 0.0),
            (segments[0].end if segments else 0.0),
        )

        if audio_offset:
            logger.info("SyncDiag: audio_stream_start_offset=%.3f", audio_offset)
            segments = apply_time_offset(segments, audio_offset)
            logger.info(
                "SyncDiag: first_seg_after_audio_offset=%.3f",
                (segments[0].start if segments else 0.0),
            )

        # Sync correction modes: off | manual | auto
        # IMPORTANT: if ffsubsync is enabled, we skip pre-shift to avoid double-shifting.
        sync_mode = SUBTITLE_SYNC_MODE if SUBTITLE_SYNC_MODE in {"off", "manual", "auto"} else "auto"

        if not SUBTITLE_ENABLE_FFSUBSYNC:
            if sync_mode == "manual" and SUBTITLE_EXTRA_DELAY_SECONDS:
                segments = shift_segments(segments, SUBTITLE_EXTRA_DELAY_SECONDS)

            elif sync_mode == "auto":
                # Robust mode: delay-only correction.
                # We only shift subtitles FORWARD when they appear earlier than real speech.
                speech_start = await detect_first_speech_start(audio_path)
                first_seg_start = segments[0].start if segments else 0.0
                delta = speech_start - first_seg_start
                max_shift = max(0.0, SUBTITLE_AUTO_MAX_SHIFT_SECONDS)
                logger.info(
                    "SyncDiag: auto_mode speech_start=%.3f first_seg_start=%.3f raw_delta=%.3f max_shift=%.3f",
                    speech_start,
                    first_seg_start,
                    delta,
                    max_shift,
                )

                # Delay-only: ignore negative correction.
                if delta < 0:
                    delta = 0.0

                if max_shift > 0:
                    delta = min(max_shift, delta)

                if delta >= 0.20:
                    logger.info(
                        "Auto subtitle sync applied (delay-only): provider=%s speech_start=%.3f first_seg=%.3f delta=%.3f",
                        TRANSCRIBE_PROVIDER,
                        speech_start,
                        first_seg_start,
                        delta,
                    )
                    segments = shift_segments(segments, delta)
                logger.info(
                    "SyncDiag: first_seg_after_preshift=%.3f",
                    (segments[0].start if segments else 0.0),
                )
        else:
            logger.info("SyncDiag: pre-shift skipped because ffsubsync is enabled")

        translated_segments = await translate_segments(
            segments=segments,
            source_language=detected_language,
            target_language=target_language,
            ai_service=ai_service,
        )
        srt_content = build_srt_content(translated_segments)

        # Strong final alignment pass (optional) for difficult clips.
        if SUBTITLE_ENABLE_FFSUBSYNC:
            srt_content = await sync_srt_with_ffsubsync(video_path, srt_content)

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
