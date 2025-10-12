"""Debug phonemization output"""

from smalltts.data.phonemization.phonemes import set_language, _phonemize, get_token_ids, p2idx

set_language("multilingual")

text = "こんにちは"
print(f"Text: {text}")
has_jp = any('\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FFF' for c in text)
print(f"Has Japanese chars: {has_jp}")

# Get phonemes
phonemes = _phonemize(text)
print(f"\nPhonemized: {phonemes}")
print(f"Phoneme type: {type(phonemes)}")
print(f"Phonemes repr: {repr(phonemes)}")

# Get token IDs
tokens = get_token_ids(text)
print(f"\nToken IDs: {tokens}")

# Check p2idx mapping
print(f"\nPhoneme to index mapping (sample):")
print(f"Total phonemes in vocabulary: {len(p2idx)}")

# Show first 10 and last 10
items = list(p2idx.items())
print(f"First 10: {items[:10]}")
print(f"Last 10: {items[-10:]}")

# Check if phonemes from pyopenjtalk are in mapping
if phonemes:
    phoneme_list = phonemes.split()
    print(f"\nPhonemes from text: {phoneme_list}")
    for p in phoneme_list:
        if p in p2idx:
            print(f"  '{p}' -> {p2idx[p]}")
        else:
            print(f"  '{p}' -> NOT IN MAPPING!")
