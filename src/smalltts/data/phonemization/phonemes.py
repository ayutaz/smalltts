import logging
import re
from typing import List

from phonemizer.backend import EspeakBackend
from phonemizer.logger import get_logger

from .normalizer import EnglishTextNormalizer
from .normalizer_ja import JapaneseTextNormalizer

try:
    import pyopenjtalk
    _PYOPENJTALK_AVAILABLE = True
except ImportError:
    _PYOPENJTALK_AVAILABLE = False
    logging.warning("pyopenjtalk-plus not installed. Japanese phonemization will not be available.")

# Language setting
# Options: "en" (English only), "ja" (Japanese only), "multilingual" (both)
# Default: "multilingual" for flexible model that supports both languages
LANGUAGE = "multilingual"

# English phonemes (existing)
_punct = ';:,.!?¡¿—…"«»"" '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Japanese phonemes (based on pyopenjtalk output)
_punct_ja = "、。！？…ー"
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

def _build_phoneme_mappings(language="en"):
    """Build phoneme mappings based on language

    For multilingual support, creates a unified vocabulary containing both
    English and Japanese phonemes (~250 tokens total).
    """
    _syms = []
    _seen = set()

    if language == "multilingual":
        # Multilingual: Combine English and Japanese phonemes
        # IMPORTANT: Add Japanese phonemes first to ensure they get correct IDs (175+)
        for ch in _phonemes_ja:
            if ch not in _seen:
                _seen.add(ch)
                _syms.append(ch)
        # Add English phonemes (skip duplicates)
        for ch in _punct + _letters + _letters_ipa:
            if ch not in _seen:
                _seen.add(ch)
                _syms.append(ch)
    elif language == "ja":
        # Japanese only
        for ch in _phonemes_ja:
            if ch not in _seen:
                _seen.add(ch)
                _syms.append(ch)
    else:
        # English only
        for ch in _punct + _letters + _letters_ipa:
            if ch not in _seen:
                _seen.add(ch)
                _syms.append(ch)

    p2idx = {ch: i + 1 for i, ch in enumerate(_syms)}
    idx2p = {v: k for k, v in p2idx.items()}
    phoneme_len = len(p2idx) + 1

    return p2idx, idx2p, phoneme_len, _syms

# Initialize with default language
p2idx, idx2p, phoneme_len, _syms = _build_phoneme_mappings(LANGUAGE)
phonemes: List[str] = _syms

logging.getLogger().setLevel(logging.CRITICAL)

# Initialize English backend lazily to avoid requiring espeak-ng when using Japanese only
_es = None
_tok = re.compile(r"\w+|[^\w\s]")
normalizer = EnglishTextNormalizer()
normalizer_ja = JapaneseTextNormalizer(convert_numbers=False)


def _get_espeak_backend():
    """Lazily initialize EspeakBackend"""
    global _es
    if _es is None:
        _es = EspeakBackend(
            language="en-us",
            preserve_punctuation=True,
            with_stress=True,
            words_mismatch="ignore",
            logger=get_logger(verbosity="quiet"),
        )
    return _es


def _phonemize_ja(text: str) -> str:
    """Phonemize Japanese text using pyopenjtalk"""
    if not _PYOPENJTALK_AVAILABLE:
        raise RuntimeError("pyopenjtalk-plus is not installed. Please install it to use Japanese phonemization.")

    # Normalize text first (full-width to half-width, symbol normalization)
    text = normalizer_ja.normalize(text)

    # Use pyopenjtalk to convert text to phonemes
    phonemized = pyopenjtalk.g2p(text, kana=False)
    return phonemized


def _phonemize(text: str, force_language: str = None) -> str:
    """Phonemize text based on the current language setting

    Args:
        text: Text to phonemize
        force_language: Override language detection ("en" or "ja")

    Returns:
        Phonemized text
    """
    # Determine which language to use
    lang = force_language if force_language else LANGUAGE

    # For multilingual mode, auto-detect language (simple heuristic)
    if lang == "multilingual":
        # Check if text contains Japanese characters
        has_japanese = any('\u3040' <= c <= '\u309F' or  # Hiragana
                         '\u30A0' <= c <= '\u30FF' or  # Katakana
                         '\u4E00' <= c <= '\u9FFF'     # Kanji
                         for c in text)
        lang = "ja" if has_japanese else "en"

    # Phonemize based on detected/selected language
    if lang == "ja":
        return _phonemize_ja(text)
    else:
        # English phonemization
        text = normalizer.normalize(text)
        es = _get_espeak_backend()
        phonemized = " ".join(_tok.findall(es.phonemize([text])[0]))
        return phonemized


def set_language(language: str):
    """Set the language for phonemization

    Args:
        language: "en" (English), "ja" (Japanese), or "multilingual" (both)
    """
    global LANGUAGE, p2idx, idx2p, phoneme_len, phonemes, _syms

    if language not in ["en", "ja", "multilingual"]:
        raise ValueError(f"Unsupported language: {language}. Use 'en', 'ja', or 'multilingual'.")

    if language in ["ja", "multilingual"] and not _PYOPENJTALK_AVAILABLE:
        raise RuntimeError("pyopenjtalk-plus is not installed. Please install it to use Japanese phonemization.")

    LANGUAGE = language
    p2idx, idx2p, phoneme_len, _syms = _build_phoneme_mappings(language)
    phonemes = _syms
    logging.info(f"Language set to: {language}, phoneme_len: {phoneme_len}")


def get_token_ids(text: str, force_language: str = None):
    """Convert text to phoneme token IDs

    Args:
        text: Text to convert
        force_language: Override language detection ("en" or "ja")

    Returns:
        List of token IDs
    """
    s = _phonemize(text, force_language)

    # Determine language for tokenization
    lang = force_language if force_language else LANGUAGE
    if lang == "multilingual":
        # Auto-detect based on text content
        has_japanese = any('\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FFF'
                         for c in text)
        lang = "ja" if has_japanese else "en"

    if lang == "ja":
        # Japanese phonemes are space-separated
        phoneme_list = s.split()
        return [p2idx[p] for p in phoneme_list if p in p2idx]
    else:
        # English phonemes are character-by-character
        return [p2idx[c] for c in s if c in p2idx]


def decode_token_ids(token_ids, is_japanese: bool = None):
    """Decode phoneme token IDs back to phoneme string

    Args:
        token_ids: List of token IDs
        is_japanese: If None, uses current LANGUAGE setting to determine format

    Returns:
        Phoneme string
    """
    if is_japanese is None:
        # Use current language setting
        is_japanese = LANGUAGE in ["ja", "multilingual"]

    if is_japanese:
        # Japanese phonemes should be space-separated
        return " ".join(idx2p.get(t, "") for t in token_ids if t in idx2p)
    else:
        # English phonemes are concatenated
        return "".join(idx2p.get(t, "") for t in token_ids)


if __name__ == "__main__":
    print("=" * 80)
    print("ENGLISH PHONEMIZATION TEST")
    print("=" * 80)
    set_language("en")
    print(f"Language: {LANGUAGE}, Phoneme vocabulary size: {phoneme_len}\n")

    sentences_en = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world!",
        "Python is an amazing programming language.",
        "Dr. Smith and Mrs. Johnson met at 3:30pm.",
        "The company earned $1,250,000.50 in Q4 2023.",
    ]
    for s in sentences_en:
        p = _phonemize(s)
        tids = get_token_ids(s)
        dec = decode_token_ids(tids)
        print("Original  :", s)
        print("Phoneme   :", p)
        print("Token IDs :", tids[:64], "..." if len(tids) > 64 else "")
        print("Decoded   :", dec)
        print()

    print("\n" + "=" * 80)
    print("JAPANESE PHONEMIZATION TEST")
    print("=" * 80)

    if _PYOPENJTALK_AVAILABLE:
        set_language("ja")
        print(f"Language: {LANGUAGE}, Phoneme vocabulary size: {phoneme_len}\n")

        sentences_ja = [
            "こんにちは",
            "これはテストです。",
            "音声合成システムを作っています。",
            "今日はいい天気ですね。",
            "ありがとうございます。",
        ]
        for s in sentences_ja:
            p = _phonemize(s)
            tids = get_token_ids(s)
            dec = decode_token_ids(tids)
            print("Original  :", s)
            print("Phoneme   :", p)
            print("Token IDs :", tids[:64], "..." if len(tids) > 64 else "")
            print("Decoded   :", dec)
            print()
    else:
        print("pyopenjtalk-plus is not installed. Skipping Japanese tests.")
        print("Install with: pip install pyopenjtalk-plus")

    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
