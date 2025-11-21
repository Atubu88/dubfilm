"""
Utility functions for cleaning and validating Arabic Whisper text before GPT-cleaning
"""

def is_arabic_letter(ch: str) -> bool:
    """
    Checks if a character is an Arabic letter.
    Arabic Unicode blocks: 0600–06FF, 0750–077F, 08A0–08FF
    """
    return (
        "\u0600" <= ch <= "\u06FF"
        or "\u0750" <= ch <= "\u077F"
        or "\u08A0" <= ch <= "\u08FF"
    )


def is_garbage_arabic(text: str) -> bool:
    """
    Checks if segment text is *not real Arabic speech*.

    Rules:
    - Empty → garbage
    - Keep only Arabic letters; if < 3 letters → garbage
    - This filters out:
        "اه", "ها", "يا", "ههه", "مم", "اوووو", "آآآ", "Ыыы", "ээээ"
        and all similar non-speech vocalizations.
    """

    text = text.strip()

    # empty → garbage
    if not text:
        return True

    # keep only Arabic letters
    letters = [ch for ch in text if is_arabic_letter(ch)]

    # fewer than 3 Arabic letters → not a meaningful phrase
    return len(letters) < 3


__all__ = [
    "is_garbage_arabic",
    "is_arabic_letter",
]
