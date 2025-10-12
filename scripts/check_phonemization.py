"""Check Japanese phonemization"""

from smalltts.data.phonemization.phonemes import set_language, get_token_ids

# Test with multilingual mode
set_language("multilingual")

texts = [
    "こんにちは、世界！",
    "今日はいい天気ですね",
    "ありがとうございます",
]

print("=" * 80)
print("JAPANESE PHONEMIZATION CHECK")
print("=" * 80)

for text in texts:
    print(f"\nText: {text}")
    tokens = get_token_ids(text)
    print(f"Token IDs: {tokens}")
    print(f"Number of tokens: {len(tokens)}")

    # Count Japanese vs English phonemes
    japanese_count = sum(1 for t in tokens if t >= 175)
    english_count = sum(1 for t in tokens if t < 175)

    print(f"English phonemes: {english_count}")
    print(f"Japanese phonemes: {japanese_count}")
    print(f"Japanese ratio: {japanese_count/len(tokens)*100:.1f}%")

    if japanese_count == 0:
        print("⚠️  WARNING: No Japanese phonemes detected!")
    elif japanese_count < len(tokens) * 0.5:
        print("⚠️  WARNING: Most phonemes are English, not Japanese!")

print("\n" + "=" * 80)
print("Phoneme ID ranges:")
print("  English: 0-174")
print("  Japanese: 175-204")
print("=" * 80)
