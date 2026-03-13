import json
from pathlib import Path

INPUT_JSON = Path('/home/fanfan/projects/dubfilm/out2/cartoon_segments_translated.json')
OUT_TABLE = Path('/home/fanfan/projects/dubfilm/out2/character_voice_table.json')

DEFAULT_SPEAKER_TO_CHARACTER = {
    'A': 'boy',
    'B': 'grandpa',
    'C': 'shaytan',
}

DEFAULT_CHARACTER_TO_VOICE = {
    'boy': 'cedar',
    'grandpa': 'cedar',
    'shaytan': 'onyx',
}


def main() -> None:
    data = json.loads(INPUT_JSON.read_text(encoding='utf-8'))
    speakers = sorted({str(s.get('speaker')) for s in data.get('segments', []) if s.get('speaker') is not None})

    speaker_to_character = {}
    for sp in speakers:
        speaker_to_character[sp] = DEFAULT_SPEAKER_TO_CHARACTER.get(sp, '')

    table = {
        'speaker_to_character': speaker_to_character,
        'character_to_voice': DEFAULT_CHARACTER_TO_VOICE,
        'note': 'Edit this table manually to lock voices by character for all future renders.',
    }
    OUT_TABLE.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT_TABLE)


if __name__ == '__main__':
    main()
