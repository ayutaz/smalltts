"""Japanese-only phonemization using pyopenjtalk

This module provides Japanese text-to-phoneme conversion using pyopenjtalk-plus.
English and multilingual support have been removed for simplicity.
"""

import logging
from typing import List

from .normalizer_ja import JapaneseTextNormalizer

try:
    import pyopenjtalk
    _PYOPENJTALK_AVAILABLE = True
except ImportError:
    _PYOPENJTALK_AVAILABLE = False
    logging.error("pyopenjtalk-plus not installed. Please install: pip install pyopenjtalk-plus")
    raise ImportError("pyopenjtalk-plus is required for Japanese phonemization")

# Japanese phonemes (based on pyopenjtalk output)
_phonemes_ja = [
    # Vowels
    "a", "i", "u", "e", "o",
    # Vowel variants (devoiced, etc.)
    "I", "U", "E", "O", "A",
    # Consonants
    "k", "s", "t", "n", "h", "m", "y", "r", "w", "g", "z", "d", "b", "p", "f", "v",
    # Special
    "N",  # ん
    "q",  # 促音（っ）
    # Palatalized consonants and digraphs
    "ch", "sh", "ts", "j",
    "ky", "gy", "ny", "hy", "by", "py", "my", "ry",
    "kw", "gw", "ty", "dy", "gy", "zy",
    # Long vowels (may appear in some phonemization schemes)
    "a:", "i:", "u:", "e:", "o:",
    # Pause/silence
    "pau", "cl", "sp", "sil",
    # Punctuation and space
    " ", "、", "。", "！", "？", "…", "ー", ",", ".",
]

def _build_phoneme_mappings():
    """Build Japanese phoneme mappings"""
    _syms = []
    _seen = set()

    for ch in _phonemes_ja:
        if ch not in _seen:
            _seen.add(ch)
            _syms.append(ch)

    p2idx = {ch: i + 1 for i, ch in enumerate(_syms)}
    idx2p = {v: k for k, v in p2idx.items()}
    phoneme_len = len(p2idx) + 1

    return p2idx, idx2p, phoneme_len, _syms

# Initialize phoneme mappings
p2idx, idx2p, phoneme_len, _syms = _build_phoneme_mappings()
phonemes: List[str] = _syms

logging.getLogger().setLevel(logging.CRITICAL)

# Initialize Japanese normalizer
normalizer_ja = JapaneseTextNormalizer(convert_numbers=False)


def _phonemize_ja(text: str) -> str:
    """Phonemize Japanese text using pyopenjtalk

    Args:
        text: Japanese text to phonemize

    Returns:
        Space-separated phoneme string
    """
    if not _PYOPENJTALK_AVAILABLE:
        raise RuntimeError("pyopenjtalk-plus is not installed. Please install it to use Japanese phonemization.")

    # Normalize text first (full-width to half-width, symbol normalization)
    text = normalizer_ja.normalize(text)

    # Use pyopenjtalk to convert text to phonemes
    phonemized = pyopenjtalk.g2p(text, kana=False)
    return phonemized


def get_token_ids(text: str):
    """Convert Japanese text to phoneme token IDs

    Args:
        text: Japanese text to convert

    Returns:
        List of token IDs
    """
    # Phonemize the text
    phonemized = _phonemize_ja(text)

    # Japanese phonemes are space-separated
    phoneme_list = phonemized.split()
    return [p2idx[p] for p in phoneme_list if p in p2idx]


def decode_token_ids(token_ids):
    """Decode phoneme token IDs back to phoneme string

    Args:
        token_ids: List of token IDs

    Returns:
        Space-separated phoneme string
    """
    return " ".join(idx2p.get(t, "") for t in token_ids if t in idx2p)


# For backwards compatibility with training scripts
def set_language(language: str):
    """Dummy function for backwards compatibility

    This function does nothing as the module is Japanese-only.
    Any language other than 'ja' will raise an error.

    Args:
        language: Must be "ja"
    """
    if language != "ja":
        raise ValueError(f"This module only supports Japanese. Got: {language}")


if __name__ == "__main__":
    print("=" * 80)
    print("JAPANESE PHONEMIZATION TEST")
    print("=" * 80)
    print(f"Phoneme vocabulary size: {phoneme_len}\n")

    sentences_ja = [
        "こんにちは",
        "これはテストです。",
        "音声合成システムを作っています。",
        "今日はいい天気ですね。",
        "ありがとうございます。",
    ]

    for s in sentences_ja:
        p = _phonemize_ja(s)
        tids = get_token_ids(s)
        dec = decode_token_ids(tids)
        print("Original  :", s)
        print("Phoneme   :", p)
        print("Token IDs :", tids[:64], "..." if len(tids) > 64 else "")
        print("Decoded   :", dec)
        print()

    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
