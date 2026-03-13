import logging
import math
import os
from pathlib import Path
from uuid import uuid4

from ai.service import AIService
from config import (
    DUB_ORIGINAL_AUDIO_VOLUME,
    DUB_TTS_AUDIO_VOLUME,
    DUB_TTS_MIN_SPEED,
    DUB_TTS_MAX_SPEED,
    DUB_TARGET_CHARS_PER_SEC,
    DUB_MIN_SEGMENT_DURATION,
    DUB_MULTI_VOICE,
    DUB_MULTI_VOICE_LIST,
    DUB_MULTI_VOICE_MAP,
    QURAN_AYAH_TTS_MODE,
    OPENAI_TTS_VOICE,
    OPENAI_TTS_FORMAT,
    TEMP_DIR,
)
from services.subtitles import SubtitleSegment, _is_likely_quran_ayah, _run_subprocess

logger = logging.getLogger(__name__)

# Safe defaults; can be tuned via env without code edits.
DUB_TRIM_SPILL_MAX_SEC = float(os.getenv('DUB_TRIM_SPILL_MAX_SEC', '0.28'))
DUB_SHORT_ONSET_COMP_MS = int(os.getenv('DUB_SHORT_ONSET_COMP_MS', '80'))
DUB_SHORT_ONSET_THRESHOLD_SEC = float(os.getenv('DUB_SHORT_ONSET_THRESHOLD_SEC', '1.30'))
DUB_HIGH_FIT_RATIO_THRESHOLD = float(os.getenv('DUB_HIGH_FIT_RATIO_THRESHOLD', '1.60'))
DUB_HIGH_FIT_MAX_SPEED = float(os.getenv('DUB_HIGH_FIT_MAX_SPEED', '1.26'))


async def _timing_rewrite(text: str, target_language: str, budget_chars: int, ai_service: AIService) -> str:
    """Rewrite text to fit strict char budget while preserving core meaning."""
    prompt = (
        f"Перефразируй текст на {target_language} под лимит {budget_chars} символов. "
        "Сохрани ключевой смысл, убери второстепенные детали. "
        "Без новых фактов. Одна строка, без кавычек."
    )
    rewritten = await ai_service.translate_text(
        text=f"{prompt}\n\nТекст:\n{text}",
        source_language="auto",
        target_language=target_language,
    )
    return (rewritten or "").strip().replace("\n", " ")


async def constrain_translated_segments(
    segments: list[SubtitleSegment],
    target_language: str,
    ai_service: AIService,
) -> list[SubtitleSegment]:
    """Timing-aware rewrite per segment.
    Keeps timeline unchanged and avoids hard clipping by adapting text to slot budget.
    """
    constrained: list[SubtitleSegment] = []

    for i, seg in enumerate(segments, start=1):
        original_text = (seg.text or "").strip()
        text = original_text
        if not text:
            constrained.append(seg)
            continue

        seg_dur = max(DUB_MIN_SEGMENT_DURATION, float(seg.end - seg.start))
        budget = max(16, int(seg_dur * DUB_TARGET_CHARS_PER_SEC))
        min_chars_floor = 8 if seg_dur < 2.4 else 14 if seg_dur < 3.6 else 20

        if len(text) > budget:
            try:
                candidate = await _timing_rewrite(text, target_language, budget, ai_service)
                # Anti-collapse guard: reject over-aggressive shrinking on medium/long slots.
                if len(candidate) >= min_chars_floor:
                    text = candidate
            except Exception as exc:
                logger.warning("DubDiag: timing_rewrite failed seg#%d: %s", i, exc)

        # second-pass tighten if still too long
        if len(text) > budget:
            tighter = max(12, int(budget * 0.85))
            try:
                candidate = await _timing_rewrite(text, target_language, tighter, ai_service)
                if len(candidate) >= min_chars_floor:
                    text = candidate
            except Exception as exc:
                logger.warning("DubDiag: timing_rewrite pass2 failed seg#%d: %s", i, exc)

        # last fallback (rare)
        if len(text) > budget:
            text = text[: max(10, budget - 1)].rstrip(" ,.;:-") + "…"

        # Final semantic floor: avoid ultra-short collapsed phrases in long slots.
        if seg_dur >= 3.0 and len(text) < min_chars_floor:
            logger.warning(
                "DubDiag: anti-collapse restore seg#%d seg_dur=%.3f out_len=%d floor=%d",
                i,
                seg_dur,
                len(text),
                min_chars_floor,
            )
            text = original_text[: max(min_chars_floor, min(len(original_text), budget))].strip()

        constrained.append(SubtitleSegment(start=seg.start, end=seg.end, text=text, speaker=seg.speaker))
        logger.info(
            "DubDiag: constrain seg#%d seg_dur=%.3f budget=%d out_len=%d",
            i,
            seg_dur,
            budget,
            len(text),
        )

    logger.info("DubDiag: constrained_segments=%d", len(constrained))
    return constrained


async def _probe_duration(path: Path) -> float:
    stdout, stderr, code = await _run_subprocess(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    )
    if code != 0:
        logger.warning("DubDiag: ffprobe duration failed for %s: %s", path, stderr or stdout)
        return 0.0
    try:
        return float((stdout or "").strip() or 0)
    except ValueError:
        return 0.0


def _build_atempo_chain(speed_ratio: float) -> str:
    """Build ffmpeg atempo chain to fit allowed 0.5..2.0 per filter."""
    if speed_ratio <= 0:
        return ""

    ratio = speed_ratio
    parts: list[float] = []

    while ratio > 2.0:
        parts.append(2.0)
        ratio /= 2.0
    while ratio < 0.5:
        parts.append(0.5)
        ratio /= 0.5

    parts.append(ratio)
    return ",".join(f"atempo={p:.5f}" for p in parts)


def _split_text_evenly(words: list[str], parts: int) -> list[str]:
    if parts <= 1 or not words:
        return [" ".join(words).strip()]
    size = math.ceil(len(words) / parts)
    chunks = [" ".join(words[i:i + size]).strip() for i in range(0, len(words), size)]
    return [c for c in chunks if c]


def normalize_segments_for_dub(
    segments: list[SubtitleSegment],
    *,
    min_duration: float = 2.2,
    max_duration: float = 6.8,
) -> list[SubtitleSegment]:
    """Pre-normalize timestamps for stable dubbing quality.
    - Merge very short neighbors (<min_duration)
    - Keep resulting chunks in a practical 2.2..6.8s range when possible
    """
    if not segments:
        return []

    out: list[SubtitleSegment] = []
    cur = SubtitleSegment(
        start=segments[0].start,
        end=segments[0].end,
        text=segments[0].text,
        speaker=segments[0].speaker,
    )

    for nxt in segments[1:]:
        cur_dur = max(0.0, cur.end - cur.start)
        nxt_dur = max(0.0, nxt.end - nxt.start)
        gap = max(0.0, nxt.start - cur.end)
        merged_dur = max(cur.end, nxt.end) - cur.start

        should_merge = (
            (cur_dur < min_duration or nxt_dur < min_duration)
            and gap <= 0.22
            and merged_dur <= max_duration
        )

        if should_merge:
            cur = SubtitleSegment(
                start=cur.start,
                end=max(cur.end, nxt.end),
                text=f"{cur.text.strip()} {nxt.text.strip()}".strip(),
                speaker=cur.speaker if cur.speaker == nxt.speaker else cur.speaker or nxt.speaker,
            )
        else:
            out.append(cur)
            cur = SubtitleSegment(start=nxt.start, end=nxt.end, text=nxt.text, speaker=nxt.speaker)

    out.append(cur)
    return out


def _merge_short_segments(
    segments: list[SubtitleSegment],
    min_duration: float,
    max_merged_duration: float = 8.0,
) -> list[SubtitleSegment]:
    """Merge only truly short neighboring segments, with hard duration cap.
    Prevents accidental collapse of many segments into one long chunk.
    """
    if not segments:
        return []

    merged: list[SubtitleSegment] = []
    cur = SubtitleSegment(
        start=segments[0].start,
        end=segments[0].end,
        text=segments[0].text,
        speaker=segments[0].speaker,
    )

    for nxt in segments[1:]:
        cur_dur = max(0.0, cur.end - cur.start)
        gap = max(0.0, nxt.start - cur.end)
        candidate_dur = max(cur.end, nxt.end) - cur.start

        # Merge only if current chunk is short, gap is tiny, and merged chunk stays reasonable.
        should_merge = (
            cur_dur < min_duration
            and gap <= 0.18
            and candidate_dur <= max_merged_duration
        )

        if should_merge:
            cur = SubtitleSegment(
                start=cur.start,
                end=max(cur.end, nxt.end),
                text=f"{cur.text.strip()} {nxt.text.strip()}".strip(),
                speaker=cur.speaker if cur.speaker == nxt.speaker else cur.speaker or nxt.speaker,
            )
        else:
            merged.append(cur)
            cur = SubtitleSegment(start=nxt.start, end=nxt.end, text=nxt.text, speaker=nxt.speaker)

    merged.append(cur)
    return merged


def stabilize_speakers(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    """Smooth noisy speaker labels to reduce random voice flips.

    Rules:
    - Fill None from neighbors when both sides agree.
    - For tiny middle segment (<=2.2s), if neighbors have same speaker -> use neighbor speaker.
    - One-hop majority smoothing over triplets.
    """
    if not segments:
        return segments

    out = [SubtitleSegment(start=s.start, end=s.end, text=s.text, speaker=s.speaker) for s in segments]

    # 1) Fill missing speaker from agreeing neighbors
    for i in range(1, len(out) - 1):
        if out[i].speaker is None and out[i - 1].speaker and out[i - 1].speaker == out[i + 1].speaker:
            out[i].speaker = out[i - 1].speaker

    # 2) Smooth tiny middle segments between same speaker
    for i in range(1, len(out) - 1):
        dur = max(0.0, out[i].end - out[i].start)
        left = out[i - 1].speaker
        right = out[i + 1].speaker
        if left and right and left == right and dur <= 2.2:
            out[i].speaker = left

    # 3) Majority vote over local triplet
    for i in range(1, len(out) - 1):
        a, b, c = out[i - 1].speaker, out[i].speaker, out[i + 1].speaker
        if a and c and a == c and b != a:
            out[i].speaker = a

    return out


def _merge_tiny_segments_into_previous(segments: list[SubtitleSegment], tiny_threshold: float = 1.25) -> list[SubtitleSegment]:
    """Final safety pass: tiny segments are merged into previous one to avoid clipped tails.

    1.25s threshold is intentionally conservative for dubbing: sub-1.2s chunks often
    produce clipped-sounding endings after speed-fit/trim.
    """
    if not segments:
        return []

    out: list[SubtitleSegment] = [segments[0]]
    for seg in segments[1:]:
        seg_dur = max(0.0, seg.end - seg.start)
        if seg_dur < tiny_threshold:
            prev = out[-1]
            out[-1] = SubtitleSegment(
                start=prev.start,
                end=max(prev.end, seg.end),
                text=f"{prev.text.strip()} {seg.text.strip()}".strip(),
                speaker=prev.speaker if prev.speaker == seg.speaker else prev.speaker or seg.speaker,
            )
        else:
            out.append(seg)
    return out


def _rebalance_segments_for_tts(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    """Pre-balance for natural dubbing.
    1) Merge short neighboring segments (primary)
    2) Split only very long texts that overload their time slot
    3) Final merge of tiny tails (<~0.85s) to avoid clipped endings
    """
    # Step 1: merge shorts first to prevent explosion into micro segments.
    pre = _merge_short_segments(segments, max(1.6, DUB_MIN_SEGMENT_DURATION))

    balanced: list[SubtitleSegment] = []

    for seg in pre:
        text = (seg.text or "").strip()
        if not text:
            continue

        seg_dur = max(DUB_MIN_SEGMENT_DURATION, float(seg.end - seg.start))
        max_chars_for_slot = max(18, int(seg_dur * DUB_TARGET_CHARS_PER_SEC))

        # If text fits slot, keep as-is.
        if len(text) <= max_chars_for_slot:
            balanced.append(seg)
            continue

        # Step 2: split only when truly needed.
        words = text.split()
        chunks_n = max(2, math.ceil(len(text) / max_chars_for_slot))
        chunks = _split_text_evenly(words, chunks_n)

        chunk_dur = (seg.end - seg.start) / max(1, len(chunks))
        cursor = seg.start
        for idx, chunk_text in enumerate(chunks):
            ch_start = cursor
            ch_end = seg.end if idx == len(chunks) - 1 else cursor + chunk_dur
            balanced.append(SubtitleSegment(start=ch_start, end=ch_end, text=chunk_text, speaker=seg.speaker))
            cursor = ch_end

    # Step 3: final safety merge for tiny tails.
    balanced = _merge_tiny_segments_into_previous(balanced, tiny_threshold=1.25)

    # Extra tail guard: never leave the last segment too tiny, otherwise it gets audibly clipped.
    if len(balanced) >= 2:
        last = balanced[-1]
        last_dur = max(0.0, last.end - last.start)
        if last_dur < 1.30:
            prev = balanced[-2]
            balanced[-2] = SubtitleSegment(
                start=prev.start,
                end=max(prev.end, last.end),
                text=f"{prev.text.strip()} {last.text.strip()}".strip(),
                speaker=prev.speaker if prev.speaker == last.speaker else prev.speaker or last.speaker,
            )
            balanced.pop()

    return balanced


async def synthesize_segment_audios(
    ai_service: AIService,
    segments: list[SubtitleSegment],
    *,
    target_language: str = "Russian",
) -> list[tuple[SubtitleSegment, Path, float]]:
    """Generate one TTS audio file per translated segment.
    Uses duration-fit loop: if synthesized speech doesn't fit its time slot,
    rewrite text shorter and retry up to 2 extra iterations.
    """
    ext = OPENAI_TTS_FORMAT if OPENAI_TTS_FORMAT in {"mp3", "wav", "opus", "aac", "flac"} else "mp3"
    out: list[tuple[SubtitleSegment, Path, float]] = []

    prepared_segments = _rebalance_segments_for_tts(segments)
    logger.info("DubDiag: tts_segments_before=%d after_balance=%d", len(segments), len(prepared_segments))

    speaker_voice_map: dict[str, str] = {}
    voice_pool = DUB_MULTI_VOICE_LIST[:] if DUB_MULTI_VOICE_LIST else [OPENAI_TTS_VOICE]

    def pick_voice(seg: SubtitleSegment, index: int) -> str:
        if not DUB_MULTI_VOICE:
            return OPENAI_TTS_VOICE
        if not voice_pool:
            return OPENAI_TTS_VOICE

        if seg.speaker:
            key = str(seg.speaker)
            # Explicit lock (A/B/C...) has top priority.
            if key in DUB_MULTI_VOICE_MAP:
                return DUB_MULTI_VOICE_MAP[key]
            if key not in speaker_voice_map:
                speaker_voice_map[key] = voice_pool[len(speaker_voice_map) % len(voice_pool)]
            return speaker_voice_map[key]

        # Fallback when speaker labels unavailable: deterministic cycling by index.
        return voice_pool[(index - 1) % len(voice_pool)]

    for i, seg in enumerate(prepared_segments, start=1):
        base_text = (seg.text or "").strip()
        if not base_text:
            continue

        if QURAN_AYAH_TTS_MODE == "mute" and _is_likely_quran_ayah(base_text):
            logger.info("DubDiag: skip TTS for ayah-like segment seg#%d (original audio only)", i)
            continue

        text = base_text
        seg_dur = max(0.6, float(seg.end - seg.start))
        max_allowed_tts = seg_dur * DUB_TTS_MAX_SPEED

        best_path: Path | None = None
        best_dur = 0.0

        for attempt in range(1, 4):
            selected_voice = pick_voice(seg, i)
            audio_bytes = await ai_service.synthesize_speech(
                text=text,
                voice=selected_voice,
                audio_format=OPENAI_TTS_FORMAT,
            )
            tts_path = TEMP_DIR / f"dub_seg_{i}_try{attempt}_{uuid4().hex}.{ext}"
            tts_path.write_bytes(audio_bytes)
            tts_dur = await _probe_duration(tts_path)

            suspicious_short = seg_dur >= 2.0 and 0 < tts_dur < 0.60

            # Keep best (shortest) candidate, but reject suspiciously tiny clips.
            if suspicious_short:
                logger.warning(
                    "DubDiag: suspicious short TTS seg#%d attempt=%d seg_dur=%.3f tts_dur=%.3f text_len=%d",
                    i,
                    attempt,
                    seg_dur,
                    tts_dur,
                    len(text),
                )
                tts_path.unlink(missing_ok=True)
            elif best_path is None or (tts_dur > 0 and tts_dur < best_dur):
                if best_path is not None:
                    best_path.unlink(missing_ok=True)
                best_path = tts_path
                best_dur = tts_dur
            else:
                tts_path.unlink(missing_ok=True)

            logger.info(
                "DubDiag: fit seg#%d attempt=%d voice=%s speaker=%s seg_dur=%.3f max_allowed_tts=%.3f tts_dur=%.3f text_len=%d",
                i,
                attempt,
                selected_voice,
                seg.speaker,
                seg_dur,
                max_allowed_tts,
                tts_dur,
                len(text),
            )

            if not suspicious_short and tts_dur <= max_allowed_tts:
                break

            # Rewrite for next attempt.
            # - too long  -> shorter rewrite
            # - suspiciously short -> fuller rewrite to avoid clipped/missing phrase
            if suspicious_short:
                budget = max(14, int(seg_dur * DUB_TARGET_CHARS_PER_SEC * 0.80))
            else:
                tight_factor = 0.82 if seg_dur < 2.4 else (0.88 if attempt == 1 else 0.75)
                budget = max(10, int((max_allowed_tts * DUB_TARGET_CHARS_PER_SEC) * tight_factor))
            try:
                text = await _timing_rewrite(base_text if suspicious_short else text, target_language, budget, ai_service)
            except Exception as exc:
                logger.warning("DubDiag: fit rewrite failed seg#%d attempt=%d: %s", i, attempt, exc)
                text = text[: max(10, budget - 1)].rstrip(" ,.;:-") + "…"

        if best_path is None:
            continue

        out.append((seg, best_path, best_dur))

    logger.info("DubDiag: synthesized segment audios=%d", len(out))
    return out


async def compose_dubbed_video_from_segments(
    video_path: Path,
    segment_audios: list[tuple[SubtitleSegment, Path, float]],
) -> Path:
    """Mix original audio with per-segment translated voice-over aligned by segment timings."""
    output_path = TEMP_DIR / f"dub_out_{uuid4().hex}.mp4"

    src_dur = await _probe_duration(video_path)

    # Build ffmpeg inputs: [0] is source video, [1..N] are TTS chunks.
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    for _, p, _ in segment_audios:
        cmd += ["-i", str(p)]

    filter_parts: list[str] = [f"[0:a]volume={DUB_ORIGINAL_AUDIO_VOLUME}[orig]"]
    mix_labels = ["[orig]"]

    for idx, (seg, _path, tts_dur) in enumerate(segment_audios, start=1):
        seg_dur = max(0.6, float(seg.end - seg.start))
        delay_ms = max(0, int(round(seg.start * 1000)))

        # Compensate TTS onset latency on very short replicas.
        if seg_dur <= DUB_SHORT_ONSET_THRESHOLD_SEC and DUB_SHORT_ONSET_COMP_MS > 0:
            delay_ms = max(0, delay_ms - DUB_SHORT_ONSET_COMP_MS)

        # Non-overlap guard with safe tail extension:
        # allow a tiny spill into silence (not into next replica) to avoid clipped last words.
        next_start = segment_audios[idx][0].start if idx < len(segment_audios) else None
        tail_grace = 0.16  # max extra seconds
        safety_gap = 0.04  # keep gap before next segment

        if next_start is None:
            available_until = min(float(src_dur), float(seg.end) + tail_grace)
        else:
            safe_next = float(next_start) - safety_gap
            # extend beyond seg.end only if there is silence before next segment
            available_until = min(safe_next, float(seg.end) + tail_grace)
            if available_until < float(seg.end):
                available_until = float(seg.end)

        play_window = max(0.35, float(available_until) - float(seg.start))

        # If synthesized chunk is longer than slot, allow small spill instead of hard clipping.
        overflow = max(0.0, float(tts_dur) - play_window)
        trim_window = play_window
        if overflow > 0.02:
            trim_window = min(play_window + DUB_TRIM_SPILL_MAX_SEC, play_window + overflow)

        # Fit TTS chunk to segment duration by speed adjustment.
        speed_ratio = 1.0
        raw_fit_ratio = 1.0
        if tts_dur > 0:
            raw_fit_ratio = tts_dur / seg_dur
            speed_ratio = raw_fit_ratio
            # Natural speech guardrails to avoid robotic fast/slow jumps.
            max_speed_cap = min(DUB_TTS_MAX_SPEED, 1.20)
            # For clearly overlong segments, allow a small extra speed-up (still non-robotic).
            if raw_fit_ratio >= DUB_HIGH_FIT_RATIO_THRESHOLD:
                max_speed_cap = min(DUB_TTS_MAX_SPEED, DUB_HIGH_FIT_MAX_SPEED)
            speed_ratio = max(DUB_TTS_MIN_SPEED, min(max_speed_cap, speed_ratio))
        atempo_chain = _build_atempo_chain(speed_ratio)

        label = f"d{idx}"
        fade_start = max(0.0, trim_window - 0.08)
        processing_chain = []
        if atempo_chain:
            processing_chain.append(atempo_chain)
        processing_chain.append(f"atrim=0:{trim_window:.3f}")
        processing_chain.append("asetpts=N/SR/TB")
        processing_chain.append(f"afade=t=out:st={fade_start:.3f}:d=0.08")
        processing_chain.append(f"adelay={delay_ms}|{delay_ms}")
        processing_chain.append(f"volume={DUB_TTS_AUDIO_VOLUME}")
        chain = ",".join(processing_chain)
        part = f"[{idx}:a]{chain}[{label}]"

        filter_parts.append(part)
        mix_labels.append(f"[{label}]")

        logger.info(
            "DubDiag: seg#%d start=%.3f end=%.3f next_start=%s seg_dur=%.3f tts_dur=%.3f speed_ratio=%.3f play_window=%.3f trim_window=%.3f overflow=%.3f delay_ms=%d onset_comp_ms=%d",
            idx,
            seg.start,
            seg.end,
            f"{next_start:.3f}" if next_start is not None else "None",
            seg_dur,
            tts_dur,
            speed_ratio,
            play_window,
            trim_window,
            overflow,
            delay_ms,
            (DUB_SHORT_ONSET_COMP_MS if seg_dur <= DUB_SHORT_ONSET_THRESHOLD_SEC else 0),
        )

    mix_inputs = "".join(mix_labels)
    filter_parts.append(
        f"{mix_inputs}amix=inputs={len(mix_labels)}:duration=first:dropout_transition=2:normalize=0[mix]"
    )
    filter_complex = ";".join(filter_parts)

    logger.info("DubDiag: src_video_duration=%.3f segment_tracks=%d", src_dur, len(segment_audios))

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v:0",
        "-map",
        "[mix]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        str(output_path),
    ]

    stdout, stderr, code = await _run_subprocess(*cmd, timeout=600)
    if code != 0:
        raise RuntimeError(f"Failed to compose dubbed video: {stderr or stdout}")

    out_dur = await _probe_duration(output_path)
    logger.info("DubDiag: output_video=%s output_duration=%.3f", output_path, out_dur)
    return output_path
