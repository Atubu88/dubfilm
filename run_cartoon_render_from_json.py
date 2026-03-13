import asyncio
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from ai.provider import AIProvider
from ai.service import AIService
from config import OPENAI_CHAT_MODEL, OPENAI_WHISPER_MODEL, OPENAI_TTS_FORMAT
from services.dub import compose_dubbed_video_from_segments
from services.subtitles import SubtitleSegment

INPUT_JSON = Path('/home/fanfan/projects/dubfilm/out2/cartoon_segments_manual.json')
VOICE_TABLE_JSON = Path('/home/fanfan/projects/dubfilm/out2/character_voice_table.json')
OUT2_DIR = Path('/home/fanfan/projects/dubfilm/out2')
DEBUG_JSON = OUT2_DIR / 'cartoon_render_debug_segments.json'
DEBUG_SUMMARY_JSON = OUT2_DIR / 'cartoon_render_debug_summary.json'
DEBUG_LOG = OUT2_DIR / 'cartoon_render_debug.log'

# Render fit guardrails (safe defaults, override via env)
MIN_CHUNK_SLOT_SEC = float(os.getenv('DUB_MIN_CHUNK_SLOT_SEC', '1.00'))
MIN_WORDS_TO_SPLIT = int(os.getenv('DUB_MIN_WORDS_TO_SPLIT', '4'))
MAX_SPLIT_DEPTH = int(os.getenv('DUB_MAX_SPLIT_DEPTH', '0'))  # 0 = split disabled
FIT_TOLERANCE = float(os.getenv('DUB_FIT_TOLERANCE', '1.08'))

# Cheap two-pass mode: retry only bad segments from pass1
DUB_TWO_PASS_ENABLE = os.getenv('DUB_TWO_PASS_ENABLE', '0').strip().lower() not in {'0', 'false', 'no', 'off'}
DUB_PASS2_MAX_SEGMENTS = int(os.getenv('DUB_PASS2_MAX_SEGMENTS', '4'))
DUB_PASS2_MIN_RATIO = float(os.getenv('DUB_PASS2_MIN_RATIO', '1.20'))

logger = logging.getLogger('cartoon_render')


def _setup_logger() -> None:
    OUT2_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(DEBUG_LOG, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(sh)


def _token_count(s: str) -> int:
    return len((s or '').strip().split())


def _validate_segments_before_render(data: dict) -> tuple[list[dict], list[dict]]:
    issues: list[dict] = []
    risky: list[dict] = []

    segs = data.get('segments') or []
    prev_end = None
    for s in segs:
        sid = s.get('id')
        start = float(s.get('start') or 0.0)
        end = float(s.get('end') or 0.0)
        text = (s.get('text') or '').strip()
        tr = (s.get('translation_ru') or '').strip()

        if end <= start:
            issues.append({'id': sid, 'kind': 'bad_window', 'detail': f'end<=start ({start}->{end})'})
            continue

        if prev_end is not None and start < prev_end - 0.02:
            issues.append({'id': sid, 'kind': 'timeline_overlap', 'detail': f'start={start} < prev_end={prev_end}'})
        prev_end = end

        if not tr:
            continue

        dur = max(0.01, end - start)
        cps = len(tr) / dur
        ar_wc = _token_count(text)
        ru_wc = _token_count(tr)

        if ar_wc >= 3 and ru_wc >= 14 and dur < 2.2:
            issues.append({'id': sid, 'kind': 'too_dense', 'detail': f'ru_words={ru_wc}, dur={dur:.2f}'})
        if ar_wc >= 6 and ru_wc >= (ar_wc * 3.2):
            issues.append({'id': sid, 'kind': 'possible_mismatch', 'detail': f'ar_words={ar_wc}, ru_words={ru_wc}'})

        if cps > 13.5 or (dur < 1.0 and ru_wc > 3):
            risky.append({'id': sid, 'kind': 'fit_risk', 'detail': f'cps={cps:.2f}, dur={dur:.2f}, ru_words={ru_wc}'})

    return issues, risky


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


async def _synthesize_strict_split(
    ai: AIService,
    segments: list[tuple[int, SubtitleSegment, str]],
    *,
    pass_label: str = 'pass1',
) -> tuple[list[dict], list[dict]]:
    ext = OPENAI_TTS_FORMAT if OPENAI_TTS_FORMAT in {'mp3', 'wav', 'opus', 'aac', 'flac'} else 'mp3'
    temp_dir = Path('/home/fanfan/projects/dubfilm/tmp')
    temp_dir.mkdir(parents=True, exist_ok=True)

    out: list[dict] = []
    debug_rows: list[dict] = []
    serial = 0

    async def fit_piece(source_id: int, source_start: float, source_end: float, start: float, end: float, text: str, speaker: str | None, voice: str, depth: int = 0):
        nonlocal serial
        slot = max(MIN_CHUNK_SLOT_SEC, end - start)
        source_slot = max(MIN_CHUNK_SLOT_SEC, source_end - source_start)
        t = (text or '').strip()
        if not t:
            return

        audio = await ai.synthesize_speech(text=t, voice=voice, audio_format=OPENAI_TTS_FORMAT)
        p = temp_dir / f'cartoon_{pass_label}_{serial}.{ext}'
        serial += 1
        p.write_bytes(audio)
        d = await _probe_duration(p)

        words = t.split()
        fit_ratio = (d / slot) if slot > 0 else 0.0
        forced_keep = False

        if d <= slot * FIT_TOLERANCE or len(words) < MIN_WORDS_TO_SPLIT or depth >= MAX_SPLIT_DEPTH:
            if d > slot * FIT_TOLERANCE and (len(words) < MIN_WORDS_TO_SPLIT or depth >= MAX_SPLIT_DEPTH):
                forced_keep = True
                logger.warning('FORCED_KEEP sid=%s depth=%s dur=%.3f slot=%.3f ratio=%.3f words=%s', source_id, depth, d, slot, fit_ratio, len(words))
            seg_obj = SubtitleSegment(start=start, end=end, text=t, speaker=speaker)
            out.append({'source_id': source_id, 'seg': seg_obj, 'path': p, 'dur': d, 'pass': pass_label})
            debug_rows.append({
                'pass': pass_label,
                'source_id': source_id,
                'source_start': round(source_start, 3),
                'source_end': round(source_end, 3),
                'source_slot': round(source_slot, 3),
                'chunk_start': round(start, 3),
                'chunk_end': round(end, 3),
                'chunk_slot': round(slot, 3),
                'depth': depth,
                'speaker': speaker,
                'voice': voice,
                'chunk_text': t,
                'word_count': len(words),
                'audio_path': str(p),
                'audio_duration': round(d, 3),
                'fit_ratio': round(fit_ratio, 3),
                'split_used': depth > 0,
                'forced_keep': forced_keep,
            })
            return

        logger.info('SPLIT sid=%s depth=%s dur=%.3f slot=%.3f ratio=%.3f words=%s', source_id, depth, d, slot, fit_ratio, len(words))
        p.unlink(missing_ok=True)
        left_words = max(1, len(words) // 2)
        left_text = ' '.join(words[:left_words]).strip()
        right_text = ' '.join(words[left_words:]).strip()
        if not right_text:
            seg_obj = SubtitleSegment(start=start, end=end, text=t, speaker=speaker)
            out.append({'source_id': source_id, 'seg': seg_obj, 'path': p, 'dur': d, 'pass': pass_label})
            return

        mid = start + (end - start) * (left_words / len(words))
        await fit_piece(source_id, source_start, source_end, start, mid, left_text, speaker, voice, depth + 1)
        await fit_piece(source_id, source_start, source_end, mid, end, right_text, speaker, voice, depth + 1)

    for sid, seg, voice in segments:
        text = (seg.text or '').strip()
        if not text:
            continue
        await fit_piece(sid, float(seg.start), float(seg.end), float(seg.start), float(seg.end), text, seg.speaker, voice)

    return out, debug_rows


def _load_voice_table() -> tuple[dict[str, str], dict[str, str]]:
    if not VOICE_TABLE_JSON.exists():
        return {}, {}
    try:
        data = json.loads(VOICE_TABLE_JSON.read_text(encoding='utf-8'))
        c2v = data.get('character_to_voice') or {}
        s2c = data.get('speaker_to_character') or {}
        return ({str(k): str(v) for k, v in c2v.items()}, {str(k): str(v) for k, v in s2c.items()})
    except Exception:
        return {}, {}


async def main() -> None:
    _setup_logger()
    run_dt = datetime.now()
    run_started = run_dt.isoformat(timespec='seconds')
    run_tag = run_dt.strftime('%Y%m%d_%H%M%S')
    logger.info('RENDER_START input=%s run_tag=%s', INPUT_JSON, run_tag)

    data = json.loads(INPUT_JSON.read_text(encoding='utf-8'))
    video = Path(data['video'])

    issues, risky = _validate_segments_before_render(data)
    OUT2_DIR.mkdir(parents=True, exist_ok=True)
    qc_path = OUT2_DIR / 'cartoon_render_input_qc.json'
    qc_path.write_text(json.dumps({'input_json': str(INPUT_JSON), 'issue_count': len(issues), 'risk_count': len(risky), 'issues': issues, 'risky': risky}, ensure_ascii=False, indent=2), encoding='utf-8')
    if issues:
        raise RuntimeError(f'Input JSON failed sanity-check: {len(issues)} blocking issue(s). See {qc_path}')

    character_to_voice, speaker_to_character = _load_voice_table()

    segments: list[tuple[int, SubtitleSegment, str]] = []
    skipped_empty: list[int] = []
    for s in data.get('segments', []):
        sid = int(s.get('id') or 0)
        text = (s.get('translation_ru') or '').strip()
        if not text:
            skipped_empty.append(sid)
            logger.info('SKIP_EMPTY sid=%s start=%.3f end=%.3f', sid, float(s.get('start') or 0.0), float(s.get('end') or 0.0))
            continue

        speaker = str(s.get('speaker')) if s.get('speaker') is not None else None
        voice_override = (s.get('voice') or '').strip()
        character = (s.get('character') or '').strip()

        if not character and speaker and speaker in speaker_to_character:
            character = speaker_to_character[speaker]
        if not voice_override and character and character in character_to_voice:
            voice_override = character_to_voice[character]

        selected_voice = voice_override or 'onyx'
        seg = SubtitleSegment(start=float(s['start']), end=float(s['end']), text=text, speaker=speaker)
        segments.append((sid, seg, selected_voice))
        logger.info('SEG_READY sid=%s speaker=%s voice=%s slot=%.3f text_len=%s words=%s', sid, speaker, selected_voice, float(s['end']) - float(s['start']), len(text), _token_count(text))

    if not segments:
        raise RuntimeError('No translated segments found. Fill translation_ru in out2/cartoon_segments_manual.json')

    ai = AIService(provider=AIProvider(chat_model=OPENAI_CHAT_MODEL, whisper_model=OPENAI_WHISPER_MODEL))
    pass1_items, debug_rows = await _synthesize_strict_split(ai, segments, pass_label='pass1')

    # Two-pass cheap mode: retry only problematic source segments.
    pass2_bad_ids: list[int] = []
    pass2_items: list[dict] = []
    if DUB_TWO_PASS_ENABLE:
        by_sid: dict[int, list[dict]] = {}
        for r in debug_rows:
            if r.get('pass') != 'pass1':
                continue
            sid = r.get('source_id')
            if sid is None:
                continue
            sid = int(sid)
            by_sid.setdefault(sid, []).append(r)

        ranked: list[tuple[float, int, int]] = []  # (score, sid, forced_count)
        for sid, rows in by_sid.items():
            forced_count = sum(1 for x in rows if x.get('forced_keep'))
            worst_ratio = max((float(x.get('fit_ratio') or 0.0) for x in rows), default=0.0)
            if forced_count > 0 and worst_ratio >= DUB_PASS2_MIN_RATIO:
                score = worst_ratio + (0.15 * forced_count)
                ranked.append((score, sid, forced_count))

        ranked.sort(key=lambda t: t[0], reverse=True)
        bad_ids = [sid for _score, sid, _forced in ranked]
        if bad_ids:
            pass2_bad_ids = bad_ids[:DUB_PASS2_MAX_SEGMENTS]
            seg_map = {sid: (seg, voice) for sid, seg, voice in segments}
            retry_segments: list[tuple[int, SubtitleSegment, str]] = []
            for sid in pass2_bad_ids:
                if sid not in seg_map:
                    continue
                base_seg, voice = seg_map[sid]
                # Text-lock mode: never rewrite translation text in pass2.
                retry_seg = SubtitleSegment(start=base_seg.start, end=base_seg.end, text=base_seg.text, speaker=base_seg.speaker)
                retry_segments.append((sid, retry_seg, voice))

            if retry_segments:
                logger.info('PASS2_RETRY bad_ids=%s text_lock=ON min_ratio=%.2f strategy=worst_first', pass2_bad_ids, DUB_PASS2_MIN_RATIO)
                pass2_items, pass2_debug = await _synthesize_strict_split(ai, retry_segments, pass_label='pass2')
                debug_rows.extend(pass2_debug)

    # Always render and save pass1 for side-by-side comparison.
    pass1_tts_items = [(x['seg'], x['path'], x['dur']) for x in pass1_items]
    out_pass1 = await compose_dubbed_video_from_segments(video, pass1_tts_items)
    pass1_video = OUT2_DIR / f'{video.stem}_ru_dub_cartoon_manualjson_pass1_{run_tag}.mp4'
    out_pass1.replace(pass1_video)

    accepted_pass2_ids: list[int] = []
    rejected_pass2_ids: list[int] = []

    if pass2_items:
        # Accept pass2 replacement per source_id only if objectively better.
        def _score(rows: list[dict], sid: int) -> tuple[int, float]:
            sid_rows = [r for r in rows if int(r.get('source_id') or -1) == sid]
            forced = sum(1 for r in sid_rows if r.get('forced_keep'))
            worst_ratio = max((float(r.get('fit_ratio') or 0.0) for r in sid_rows), default=0.0)
            return forced, worst_ratio

        chosen_ids: set[int] = set()
        for sid in pass2_bad_ids:
            p1_forced, p1_ratio = _score([r for r in debug_rows if r.get('pass') == 'pass1'], sid)
            p2_forced, p2_ratio = _score([r for r in debug_rows if r.get('pass') == 'pass2'], sid)

            better = (p2_forced < p1_forced) or (p2_forced == p1_forced and p2_ratio < p1_ratio - 0.02)
            if better:
                chosen_ids.add(sid)
                accepted_pass2_ids.append(sid)
            else:
                rejected_pass2_ids.append(sid)

        keep = [x for x in pass1_items if int(x['source_id']) not in chosen_ids]
        take = [x for x in pass2_items if int(x['source_id']) in chosen_ids]
        all_items = keep + take

        tts_items = [(x['seg'], x['path'], x['dur']) for x in all_items]
        out_final = await compose_dubbed_video_from_segments(video, tts_items)
        final = OUT2_DIR / f'{video.stem}_ru_dub_cartoon_manualjson_{run_tag}.mp4'
        out_final.replace(final)
    else:
        tts_items = pass1_tts_items
        final = OUT2_DIR / f'{video.stem}_ru_dub_cartoon_manualjson_{run_tag}.mp4'
        shutil.copy2(pass1_video, final)

    # Keep a stable "latest" filename for convenience
    latest_final = OUT2_DIR / f'{video.stem}_ru_dub_cartoon_manualjson.mp4'
    shutil.copy2(final, latest_final)

    DEBUG_JSON.write_text(json.dumps(debug_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    forced_keep_count = sum(1 for r in debug_rows if r.get('forced_keep'))
    split_count = sum(1 for r in debug_rows if r.get('split_used'))
    summary = {
        'run_started': run_started,
        'run_finished': datetime.now().isoformat(timespec='seconds'),
        'input_json': str(INPUT_JSON),
        'video': str(video),
        'run_tag': run_tag,
        'output_video': str(final),
        'output_video_latest': str(latest_final),
        'pass1_video': str(pass1_video),
        'segments_total_in_json': len(data.get('segments') or []),
        'segments_skipped_empty_translation': skipped_empty,
        'segments_sent_to_tts': len(segments),
        'tts_chunks_final': len(tts_items),
        'chunks_split_used': split_count,
        'chunks_forced_keep_over_slot': forced_keep_count,
        'two_pass_enabled': DUB_TWO_PASS_ENABLE,
        'pass2_text_rewrite': False,
        'pass2_bad_source_ids': pass2_bad_ids,
        'pass2_chunks': len(pass2_items),
        'pass2_accepted_ids': accepted_pass2_ids,
        'pass2_rejected_ids': rejected_pass2_ids,
        'debug_log': str(DEBUG_LOG),
        'debug_rows_json': str(DEBUG_JSON),
        'qc_json': str(qc_path),
    }
    DEBUG_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    logger.info('RENDER_DONE output=%s latest=%s chunks=%s forced_keep=%s pass2_bad=%s pass2_accept=%s pass2_reject=%s pass2_chunks=%s', final, latest_final, len(tts_items), forced_keep_count, pass2_bad_ids, accepted_pass2_ids, rejected_pass2_ids, len(pass2_items))
    print(final)


if __name__ == '__main__':
    asyncio.run(main())
