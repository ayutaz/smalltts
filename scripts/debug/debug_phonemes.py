"""Debug script to verify phonemization and openjtalk label utilization

This script analyzes:
1. Raw pyopenjtalk fullcontext labels
2. Extracted phoneme sequences with all prosodic markers
3. Token IDs and vocabulary usage
4. Field extraction from fullcontext labels (A:, F:, I:, J:, C:)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smalltts.data.phonemization.phonemes import (
    set_language,
    _phonemize_ja,
    get_token_ids,
    decode_token_ids,
    _extract_prosodic_info,
    p2idx,
    idx2p,
    phoneme_len,
)
import pyopenjtalk
import re


def analyze_fullcontext_labels(text: str):
    """Analyze raw pyopenjtalk fullcontext labels"""
    print("=" * 80)
    print(f"TEXT: {text}")
    print("=" * 80)

    labels = pyopenjtalk.extract_fullcontext(text)

    print(f"\n[1] RAW FULLCONTEXT LABELS ({len(labels)} labels):")
    print("-" * 80)
    for i, label in enumerate(labels):
        print(f"{i:3d}: {label}")

    return labels


def analyze_field_extraction(labels):
    """Analyze field extraction from fullcontext labels"""
    print("\n[2] FIELD EXTRACTION ANALYSIS:")
    print("-" * 80)

    for i, label in enumerate(labels):
        # Extract phoneme
        phoneme_match = re.match(r'[^^]+\^([^-]+)-([^+]+)', label)
        if not phoneme_match:
            continue

        prev_phoneme = phoneme_match.group(1)
        curr_phoneme = phoneme_match.group(2)

        # Extract fields
        a_match = re.search(r'/A:([^+]+)\+([^+]+)\+([^/]+)/', label)
        f_match = re.search(r'/F:([^_]+)_([^#]+)#([^_]+)_([^/]+)/', label)
        i_match = re.search(r'/I:([^/]+)/', label)
        j_match = re.search(r'/J:([^/]+)/', label)
        c_match = re.search(r'/C:([^_]+)_', label)

        print(f"\n{i:3d}: prev='{prev_phoneme}' curr='{curr_phoneme}'")

        if a_match:
            print(f"     A: mora_before={a_match.group(1)}, mora_pos={a_match.group(2)}, mora_after={a_match.group(3)}")

        if f_match:
            print(f"     F: phrase_len={f_match.group(1)}, accent_type={f_match.group(2)}, pos_before={f_match.group(3)}, pos_after={f_match.group(4)}")

        if i_match:
            print(f"     I: {i_match.group(1)}")

        if j_match:
            print(f"     J: {j_match.group(1)}")

        if c_match:
            print(f"     C: pos_code={c_match.group(1)}")


def analyze_prosodic_info(text: str):
    """Analyze extracted prosodic information"""
    print("\n[3] PROSODIC INFORMATION EXTRACTION:")
    print("-" * 80)

    prosody = _extract_prosodic_info(text)

    print(f"Phonemes ({len(prosody['phonemes'])}):")
    print(f"  {' '.join(prosody['phonemes'])}")

    print(f"\nAccent nuclei positions ({len(prosody['accent_nuclei'])}):")
    print(f"  {prosody['accent_nuclei']}")

    print(f"\nAccent phrase boundaries ({len(prosody['accent_phrase_boundaries'])}):")
    print(f"  {prosody['accent_phrase_boundaries']}")

    print(f"\nIntonation phrase boundaries ({len(prosody['intonation_phrase_boundaries'])}):")
    print(f"  {prosody['intonation_phrase_boundaries']}")

    print(f"\nBreath group boundaries ({len(prosody['breath_group_boundaries'])}):")
    print(f"  {prosody['breath_group_boundaries']}")

    print(f"\nPOS tags ({len(prosody['pos_tags'])}):")
    for idx, tag in prosody['pos_tags'].items():
        phoneme = prosody['phonemes'][idx] if idx < len(prosody['phonemes']) else "???"
        print(f"  {idx:3d} ({phoneme}): {tag}")

    print(f"\nMora counts ({len(prosody['mora_counts'])}):")
    for idx, count in prosody['mora_counts'].items():
        phoneme = prosody['phonemes'][idx] if idx < len(prosody['phonemes']) else "???"
        print(f"  {idx:3d} ({phoneme}): [M{count}]")

    print(f"\nPauses ({len(prosody['pauses'])}):")
    for idx, pause_type in prosody['pauses'].items():
        phoneme = prosody['phonemes'][idx] if idx < len(prosody['phonemes']) else "???"
        print(f"  {idx:3d} ({phoneme}): {pause_type}")

    return prosody


def analyze_phonemization(text: str):
    """Analyze final phonemization output"""
    print("\n[4] FINAL PHONEMIZATION OUTPUT:")
    print("-" * 80)

    phonemized = _phonemize_ja(text)
    print(f"Phonemized: {phonemized}")

    tokens = get_token_ids(text)
    print(f"\nToken IDs ({len(tokens)}):")
    print(f"  {tokens}")

    decoded = decode_token_ids(tokens)
    print(f"\nDecoded:")
    print(f"  {decoded}")

    return phonemized, tokens, decoded


def analyze_vocabulary_usage(tokens):
    """Analyze which vocabulary items are being used"""
    print("\n[5] VOCABULARY USAGE ANALYSIS:")
    print("-" * 80)
    print(f"Total vocabulary size: {phoneme_len}")
    print(f"Tokens in this text: {len(tokens)}")
    print(f"Unique tokens: {len(set(tokens))}")

    print("\nToken breakdown:")
    token_counts = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    for token_id, count in sorted(token_counts.items()):
        phoneme = idx2p.get(token_id, "???")
        print(f"  {token_id:3d} '{phoneme}': {count} times")


def main():
    # Set language to Japanese
    set_language("ja")

    print("=" * 80)
    print("JAPANESE PHONEMIZATION DEBUG TOOL")
    print("=" * 80)
    print(f"Phoneme vocabulary size: {phoneme_len}")
    print()

    # Test sentences from JVS dataset
    test_sentences = [
        "こんにちは、世界。",
        "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。",
        "ニューイングランド風は、牛乳をベースとした、白いクリームスープであり、ボストンクラムチャウダーとも呼ばれる。",
    ]

    for i, text in enumerate(test_sentences, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}/{len(test_sentences)}")
        print(f"{'=' * 80}")

        # Analyze fullcontext labels
        labels = analyze_fullcontext_labels(text)

        # Analyze field extraction (only for first sentence to avoid too much output)
        if i == 1:
            analyze_field_extraction(labels)

        # Analyze prosodic info
        prosody = analyze_prosodic_info(text)

        # Analyze phonemization
        phonemized, tokens, decoded = analyze_phonemization(text)

        # Analyze vocabulary usage
        analyze_vocabulary_usage(tokens)

        print()

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
