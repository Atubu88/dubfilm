import asyncio
import json
from pathlib import Path

from ai.provider import AIProvider
from ai.service import AIService
from config import OPENAI_CHAT_MODEL, OPENAI_WHISPER_MODEL, DUB_TTS_MIN_SPEED, DUB_TTS_MAX_SPEED
from services.dub import compose_dubbed_video_from_segments, synthesize_segment_audios
from services.subtitles import SubtitleSegment

INPUT_JSON = Path('/home/fanfan/projects/dubfilm/out/lecture_segments_translated.json')
OUT_DIR = Path('/home/fanfan/projects/dubfilm/out')


async def _probe_duration(path: Path) -> float:
    proc = await asyncio.create_subprocess_exec(
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    try:
        return float((out or b'0').decode().strip() or 0.0)
    except Exception:
        return 0.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


async def _write_quality_report(video: Path, tts_items: list[tuple[SubtitleSegment, Path, float]]) -> Path:
    src_dur = await _probe_duration(video)
    rows = []
    risk_cut = 0
    risk_overlap = 0

    for idx, (seg, _p, tts_dur) in enumerate(tts_items, start=1):
        next_start = tts_items[idx][0].start if idx < len(tts_items) else None

        tail_grace = 0.16
        safety_gap = 0.04
        if next_start is None:
            available_until = min(float(src_dur), float(seg.end) + tail_grace)
        else:
            safe_next = float(next_start) - safety_gap
            available_until = min(safe_next, float(seg.end) + tail_grace)
            if available_until < float(seg.end):
                available_until = float(seg.end)

        play_window = max(0.35, float(available_until) - float(seg.start))
        seg_dur = max(0.6, float(seg.end - seg.start))
        speed_ratio = 1.0
        if tts_dur > 0:
            speed_ratio = tts_dur / seg_dur
            speed_ratio = _clamp(speed_ratio, DUB_TTS_MIN_SPEED, min(DUB_TTS_MAX_SPEED, 1.20))

        effective_tts_after_speed = (tts_dur / speed_ratio) if speed_ratio > 0 else tts_dur
        is_cut_risk = effective_tts_after_speed > (play_window + 0.02)
        # Real overlap risk: source segment boundaries already overlap each other.
        is_overlap_risk = (next_start is not None) and (float(seg.end) > float(next_start))

        if is_cut_risk:
            risk_cut += 1
        if is_overlap_risk:
            risk_overlap += 1

        rows.append({
            'idx': idx,
            'start': round(seg.start, 3),
            'end': round(seg.end, 3),
            'next_start': round(float(next_start), 3) if next_start is not None else None,
            'seg_dur': round(seg_dur, 3),
            'tts_dur': round(float(tts_dur), 3),
            'speed_ratio': round(float(speed_ratio), 3),
            'play_window': round(float(play_window), 3),
            'effective_tts_after_speed': round(float(effective_tts_after_speed), 3),
            'risk_cut': is_cut_risk,
            'risk_overlap': is_overlap_risk,
        })

    report = {
        'video': str(video),
        'segments': len(rows),
        'risk_cut_count': risk_cut,
        'risk_overlap_count': risk_overlap,
        'rows': rows,
    }

    report_path = OUT_DIR / f'{video.stem}_quality_report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report_path


async def main() -> None:
    data = json.loads(INPUT_JSON.read_text(encoding='utf-8'))
    video = Path(data['video'])

    segments = [
        SubtitleSegment(
            start=float(s['start']),
            end=float(s['end']),
            text=str(s.get('translation_ru') or s.get('text') or ''),
            speaker=str(s.get('speaker')) if s.get('speaker') is not None else None,
        )
        for s in data.get('segments', [])
        if (s.get('translation_ru') or s.get('text'))
    ]

    ai = AIService(provider=AIProvider(chat_model=OPENAI_CHAT_MODEL, whisper_model=OPENAI_WHISPER_MODEL))

    # Restored AUTO-FIT mode:
    # - TTS duration fit
    # - timing_rewrite when phrase does not fit in segment slot
    tts_items = await synthesize_segment_audios(ai, segments, target_language='Russian')
    report_path = await _write_quality_report(video, tts_items)
    out = await compose_dubbed_video_from_segments(video, tts_items)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final = OUT_DIR / f'{video.stem}_ru_dub_customtranslate_AUTOFIT.mp4'
    out.replace(final)
    print(final)
    print(report_path)


if __name__ == '__main__':
    asyncio.run(main())
