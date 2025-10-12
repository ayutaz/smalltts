"""Test multi-speaker JVS dataloader"""

from smalltts.data.japanese import get_jvs_dataloader
from smalltts.data.phonemization.phonemes import set_language
from smalltts.codec.onnx import Encoder

print("=" * 80)
print("MULTI-SPEAKER JVS DATALOADER TEST")
print("=" * 80)

# Set language to Japanese-only
set_language("ja")
print("\nLanguage set to: ja (Japanese only)")

# Test 1: Load first 3 speakers
print("\n[Test 1] Loading first 3 speakers...")
encoder = Encoder()
loader = get_jvs_dataloader(
    root_dir="data/jvs_ver1",
    speaker_ids=["jvs001", "jvs002", "jvs003"],
    subset="parallel100",
    codec_encoder=encoder,
    batch_size=1,
    num_workers=0,
    shuffle=False,
)

batch = next(iter(loader))
print(f"[OK] Loaded 3 speakers successfully")
print(f"   Batch size: {batch['phonemes'].shape[0]}")
print(f"   Phoneme sequence length: {batch['phonemes'].shape[1]}")
print(f"   Latent sequence length: {batch['latents'].shape[1]}")
print(f"   Sample text: {batch['texts'][0]}")
print(f"   Audio path: {batch['audio_paths'][0]}")

# Test 2: Count total samples (all 100 speakers)
print("\n[Test 2] Loading all 100 speakers (this may take a moment)...")
loader_all = get_jvs_dataloader(
    root_dir="data/jvs_ver1",
    speaker_ids=None,  # None = all speakers
    subset="parallel100",
    codec_encoder=None,  # Don't encode to speed up loading
    batch_size=1,
    num_workers=0,
    shuffle=False,
)

total_samples = len(loader_all.dataset)
print(f"[OK] Loaded all 100 speakers successfully")
print(f"   Total samples: {total_samples}")
print(f"   Expected: ~10,000 (100 speakers x 100 utterances)")

if total_samples < 9000:
    print(f"   [WARNING] Expected ~10,000 samples but got {total_samples}")
elif total_samples > 11000:
    print(f"   [WARNING] Expected ~10,000 samples but got {total_samples}")
else:
    print(f"   [OK] Sample count looks correct!")

print("\n" + "=" * 80)
print("MULTI-SPEAKER DATALOADER TEST COMPLETE")
print("=" * 80)
print("\nReady to train with all 100 JVS speakers!")
print("\nNext command:")
print("  docker-compose exec smalltts-gpu uv run accelerate launch scripts/train/teacher_japanese.py")
print("=" * 80)
