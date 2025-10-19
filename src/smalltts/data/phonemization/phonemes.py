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

    # ============================================================================
    # PROSODIC MARKERS (for natural intonation and rhythm)
    # ============================================================================

    # Accent marker
    "↓",  # Pitch accent nucleus (downstep position)

    # Phrase boundaries
    "|",   # Accent phrase boundary (文節境界)
    "#",   # Intonation phrase boundary (イントネーション句境界)
    "[BG]",  # Breath group boundary (呼気段落境界)

    # Pause duration
    "[P1]",  # Short pause (短いポーズ - 読点)
    "[P2]",  # Medium pause (中程度のポーズ - 句点)
    "[P3]",  # Long pause (長いポーズ - 段落)

    # Part-of-speech tags (品詞情報)
    "[N]",    # Noun (名詞)
    "[V]",    # Verb (動詞)
    "[ADJ]",  # Adjective (形容詞)
    "[PART]", # Particle (助詞)
    "[AUX]",  # Auxiliary verb (助動詞)
    "[SYM]",  # Symbol/Punctuation (記号)

    # Mora count (モーラ数) - for accent phrase length
    "[M1]", "[M2]", "[M3]", "[M4]", "[M5]",
    "[M6]", "[M7]", "[M8]", "[M9]", "[M10]",
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

# Suppress all warnings and logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)

# Initialize Japanese normalizer
normalizer_ja = JapaneseTextNormalizer(convert_numbers=False)


def _extract_prosodic_info(text: str) -> dict:
    """Extract comprehensive prosodic information from text using pyopenjtalk fullcontext

    Args:
        text: Japanese text to analyze

    Returns:
        Dictionary containing:
            - phonemes: List of phoneme strings
            - accent_nuclei: List of indices where accent nucleus occurs
            - accent_phrase_boundaries: List of indices for accent phrase boundaries
            - intonation_phrase_boundaries: List of indices for intonation phrase boundaries
            - breath_group_boundaries: List of indices for breath group boundaries
            - pos_tags: Dict mapping phoneme index to POS tag
            - mora_counts: Dict mapping phoneme index (phrase start) to mora count
            - pauses: Dict mapping phoneme index to pause type
    """
    import re

    labels = pyopenjtalk.extract_fullcontext(text)

    # Initialize result structure
    result = {
        'phonemes': [],
        'accent_nuclei': [],
        'accent_phrase_boundaries': [],
        'intonation_phrase_boundaries': [],
        'breath_group_boundaries': [],
        'pos_tags': {},  # {index: tag}
        'mora_counts': {},  # {index: count}
        'pauses': {},  # {index: type}
    }

    # =========================================================================
    # PASS 1: Scan all labels to find pau/sil positions and types
    # =========================================================================
    pause_info = []  # List of (label_index, pause_type)

    for label_idx, label in enumerate(labels):
        phoneme_match = re.match(r'[^^]+\^([^-]+)-([^+]+)', label)
        if phoneme_match:
            curr_phoneme = phoneme_match.group(2)
            if curr_phoneme in ['sil', 'pau']:
                # Detect pause type from surrounding context
                # Check next label for punctuation info
                pause_type = '[P2]'  # Default to medium pause

                # Try to determine pause type from context
                # In future: can check previous/next phonemes for punctuation
                if curr_phoneme == 'pau':
                    pause_type = '[P1]'  # Short pause (for commas)
                elif curr_phoneme == 'sil':
                    pause_type = '[P2]'  # Medium pause (for periods)

                pause_info.append((label_idx, pause_type))

    # =========================================================================
    # PASS 2: Extract phoneme information (skip pau/sil)
    # =========================================================================
    mora_positions = []
    phrase_accent_types = []
    f_fields = []
    i_fields = []
    j_fields = []
    c_fields = []

    # Map from label index to phoneme index
    label_to_phoneme_idx = {}
    phoneme_idx = 0
    prev_f = None
    prev_i = None
    prev_j = None

    for label_idx, label in enumerate(labels):
        # Extract phoneme (format: prev^curr-next+...)
        phoneme_match = re.match(r'[^^]+\^([^-]+)-([^+]+)', label)
        if not phoneme_match:
            continue

        curr_phoneme = phoneme_match.group(2)

        # Skip silence/pause markers (will be added later)
        if curr_phoneme in ['sil', 'pau']:
            continue

        # Extract fields
        mora_match = re.search(r'/A:([^+]+)\+([^+]+)\+([^/]+)/', label)
        f_match = re.search(r'/F:([^/]+)/', label)
        i_match = re.search(r'/I:([^/]+)/', label)
        j_match = re.search(r'/J:([^/]+)/', label)
        c_match = re.search(r'/C:([^_]+)_', label)

        if mora_match and f_match:
            try:
                mora_pos = int(mora_match.group(2))
                phrase_accent_type_str = re.search(r'/F:([^_]+)_([^#]+)#', label)

                if phrase_accent_type_str:
                    phrase_len = int(phrase_accent_type_str.group(1))
                    phrase_accent_type = int(phrase_accent_type_str.group(2))

                    # Add phoneme
                    result['phonemes'].append(curr_phoneme)
                    mora_positions.append(mora_pos)
                    phrase_accent_types.append(phrase_accent_type)

                    curr_f = f_match.group(1)
                    curr_i = i_match.group(1) if i_match else None
                    curr_j = j_match.group(1) if j_match else None
                    curr_c = c_match.group(1) if c_match else None

                    f_fields.append(curr_f)
                    i_fields.append(curr_i)
                    j_fields.append(curr_j)
                    c_fields.append(curr_c)

                    label_to_phoneme_idx[label_idx] = phoneme_idx

                    # Detect boundaries (compare with previous non-pau/sil phoneme)
                    if phoneme_idx > 0:
                        # Accent phrase boundary (F: field changes)
                        if curr_f != prev_f and prev_f is not None:
                            result['accent_phrase_boundaries'].append(phoneme_idx)

                        # Breath group boundary (I: field changes)
                        if curr_i != prev_i and prev_i is not None and curr_i is not None:
                            result['breath_group_boundaries'].append(phoneme_idx)

                        # Intonation phrase boundary (J: field changes)
                        if curr_j != prev_j and prev_j is not None and curr_j is not None:
                            result['intonation_phrase_boundaries'].append(phoneme_idx)

                    # Extract POS tag at accent phrase start (when F: field changes)
                    # Only add POS tag when starting a NEW accent phrase
                    if mora_pos == 1 and curr_c and (prev_f is None or curr_f != prev_f):
                        pos_code = curr_c
                        # Map POS codes to tags
                        pos_map = {
                            '02': '[N]',    # 名詞
                            '10': '[V]',    # 動詞
                            '20': '[ADJ]',  # 形容詞
                            '24': '[PART]', # 助詞
                            '14': '[AUX]',  # 助動詞
                            '01': '[SYM]',  # 記号
                        }
                        result['pos_tags'][phoneme_idx] = pos_map.get(pos_code, '[N]')
                        result['mora_counts'][phoneme_idx] = min(phrase_len, 10)  # Cap at 10

                    # Update previous values
                    prev_f = curr_f
                    prev_i = curr_i
                    prev_j = curr_j

                    phoneme_idx += 1

            except (ValueError, AttributeError):
                pass

    # =========================================================================
    # PASS 3: Map pauses to phoneme positions
    # =========================================================================
    for pause_label_idx, pause_type in pause_info:
        # Find the phoneme immediately before this pause
        # Search backwards from pause position
        for search_idx in range(pause_label_idx - 1, -1, -1):
            if search_idx in label_to_phoneme_idx:
                # Found the phoneme before the pause
                phoneme_before_pause = label_to_phoneme_idx[search_idx]
                result['pauses'][phoneme_before_pause] = pause_type
                break

    # =========================================================================
    # PASS 4: Find accent nucleus positions
    # =========================================================================
    current_accent = None
    for i in range(len(result['phonemes'])):
        mora_pos = mora_positions[i]
        phrase_accent_type = phrase_accent_types[i]

        # New phrase detected (mora_pos == 1)
        if mora_pos == 1:
            current_accent = phrase_accent_type

        # Check if this is the accent nucleus
        if current_accent and current_accent > 0:
            if mora_pos == current_accent:
                # Check if next phoneme starts a new mora
                if i + 1 < len(mora_positions) and mora_positions[i + 1] != mora_pos:
                    result['accent_nuclei'].append(i)  # Fixed: append i, not i+1
                elif i + 1 == len(mora_positions):
                    result['accent_nuclei'].append(i)  # Fixed: append i, not i+1

    return result


def _phonemize_ja(text: str) -> str:
    """Phonemize Japanese text with comprehensive prosodic information

    Args:
        text: Japanese text to phonemize

    Returns:
        Space-separated string with phonemes and prosodic markers including:
        - Accent nucleus (↓)
        - Accent phrase boundaries (|)
        - Intonation phrase boundaries (#)
        - Breath group boundaries ([BG])
        - Part-of-speech tags ([N], [V], etc.)
        - Mora counts ([M1], [M2], etc.)
        - Pauses ([P1], [P2], [P3])
    """
    if not _PYOPENJTALK_AVAILABLE:
        raise RuntimeError("pyopenjtalk-plus is not installed. Please install it to use Japanese phonemization.")

    # Normalize text first (full-width to half-width, symbol normalization)
    text = normalizer_ja.normalize(text)

    # Extract comprehensive prosodic information
    prosody = _extract_prosodic_info(text)

    # Build output with all prosodic markers
    result = []

    for i, phoneme in enumerate(prosody['phonemes']):
        # Insert POS tag and mora count at phrase start
        if i in prosody['pos_tags']:
            result.append(prosody['pos_tags'][i])
            if i in prosody['mora_counts']:
                mora_count = prosody['mora_counts'][i]
                result.append(f"[M{mora_count}]")

        # Insert phoneme
        result.append(phoneme)

        # Insert accent nucleus marker after this phoneme
        if i in prosody['accent_nuclei']:
            result.append("↓")

        # Insert boundaries after this phoneme
        if i in prosody['accent_phrase_boundaries']:
            result.append("|")

        if i in prosody['intonation_phrase_boundaries']:
            result.append("#")

        if i in prosody['breath_group_boundaries']:
            result.append("[BG]")

        # Insert pause after this phoneme
        if i in prosody['pauses']:
            result.append(prosody['pauses'][i])

    return " ".join(result)


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
