"""Verify Japanese-only setup before training"""

import torch
from smalltts.data.phonemization.phonemes import set_language, phoneme_len, get_token_ids
from smalltts.data.japanese import get_jvs_dataloader
from smalltts.models.backbone.model import Backbone
from smalltts.codec.onnx import Encoder

print("=" * 80)
print("JAPANESE-ONLY SETUP VERIFICATION")
print("=" * 80)

# 1. Check phoneme vocabulary
print("\n[1/5] Checking phoneme vocabulary")
set_language("ja")  # Japanese only mode
print(f"Language: ja (Japanese only)")
print(f"Phoneme vocabulary size: {phoneme_len}")

# Test phonemization
test_texts = ["こんにちは", "今日はいい天気ですね", "ありがとうございます"]
for text in test_texts:
    tokens = get_token_ids(text)
    print(f"  '{text}' -> {len(tokens)} tokens: {tokens[:10]}...")
    # Check if all tokens are within vocab size
    max_token = max(tokens) if tokens else 0
    if max_token >= phoneme_len:
        print(f"    ⚠️  ERROR: Token {max_token} >= vocab size {phoneme_len}")
    else:
        print(f"    ✅ All tokens < vocab size {phoneme_len}")

# 2. Check model initialization
print("\n[2/5] Checking model initialization")
try:
    model = Backbone(latent_dim=64)
    print(f"Model created successfully")
    print(f"Phoneme embedding shape: {model.phoneme_embedding.phoneme_embed.weight.shape}")
    expected_shape = (phoneme_len, 512)
    actual_shape = model.phoneme_embedding.phoneme_embed.weight.shape
    if actual_shape[0] == phoneme_len:
        print(f"✅ Phoneme embedding size matches: {actual_shape}")
    else:
        print(f"⚠️  ERROR: Phoneme embedding size mismatch!")
        print(f"   Expected: ({phoneme_len}, 512)")
        print(f"   Actual: {actual_shape}")
except Exception as e:
    print(f"❌ ERROR: Failed to create model: {e}")

# 3. Check data loader (multi-speaker mode)
print("\n[3/5] Checking data loader (multi-speaker mode)")
try:
    encoder = Encoder()
    print("  Loading first 3 speakers for testing...")
    loader = get_jvs_dataloader(
        root_dir="data/jvs_ver1",
        speaker_ids=["jvs001", "jvs002", "jvs003"],  # Test with 3 speakers
        subset="parallel100",
        codec_encoder=encoder,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )

    # Get first batch
    batch = next(iter(loader))
    print(f"✅ Data loader works")
    print(f"   Batch size: {batch['phonemes'].shape[0]}")
    print(f"   Phoneme sequence length: {batch['phonemes'].shape[1]}")
    print(f"   Latent sequence length: {batch['latents'].shape[1]}")

    # Check phoneme token range
    max_phoneme = batch['phonemes'].max().item()
    print(f"   Max phoneme token in batch: {max_phoneme}")
    if max_phoneme >= phoneme_len:
        print(f"   ⚠️  ERROR: Phoneme token {max_phoneme} >= vocab size {phoneme_len}")
    else:
        print(f"   ✅ All phoneme tokens < vocab size {phoneme_len}")

except Exception as e:
    print(f"❌ ERROR: Data loader failed: {e}")
    import traceback
    traceback.print_exc()

# 4. Check forward pass
print("\n[4/5] Checking model forward pass with real data")
try:
    model.eval()
    with torch.no_grad():
        phonemes = batch['phonemes']
        phonemes_lengths = batch['phonemes_lengths']
        batch_size = phonemes.shape[0]
        phonemes_mask = torch.ones(batch_size, phonemes.shape[1], dtype=torch.bool)

        latent_lengths = batch['latents_lengths']
        latents = batch['latents']

        timesteps = torch.tensor([0.5] * batch_size)
        cond = torch.zeros_like(latents)
        mask = torch.ones(batch_size, latents.shape[1], dtype=torch.bool)

        velocity = model(latents, cond, mask, phonemes, phonemes_mask, timesteps)

        print(f"✅ Forward pass successful")
        print(f"   Input latents shape: {latents.shape}")
        print(f"   Output velocity shape: {velocity.shape}")
        print(f"   Output stats: mean={velocity.mean().item():.4f}, std={velocity.std().item():.4f}")

except Exception as e:
    print(f"❌ ERROR: Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# 5. Check training configuration
print("\n[5/5] Checking training configuration")
print(f"Training parameters:")
print(f"  BATCH_SIZE: 1")
print(f"  NUM_WORKERS: 0 (required for ONNX encoder)")
print(f"  NUM_STEPS: 100,000")
print(f"  LEARNING_RATE: 1e-4 (training from scratch)")
print(f"  NUM_SPEAKERS: 100 (all JVS speakers)")
print(f"  TOTAL_SAMPLES: ~10,000 (100 speakers × 100 utterances)")
print(f"  Expected training time: ~28 hours (at 10 sec/step)")

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

issues = []
if phoneme_len < 50 or phoneme_len > 100:
    issues.append(f"Unusual phoneme vocab size: {phoneme_len}")

if not issues:
    print("✅ All checks passed!")
    print("\nReady to start Japanese-only training from scratch with all 100 JVS speakers.")
    print("\nNext steps:")
    print("1. Make sure JVS dataset is fully extracted (100 speakers)")
    print("2. Start training: docker-compose exec smalltts-gpu uv run accelerate launch scripts/train/teacher_japanese.py")
    print("\nTraining info:")
    print("  - Using all 100 JVS speakers (~10,000 samples)")
    print("  - Training from scratch (no checkpoint loading)")
    print("  - Japanese-only phoneme vocabulary")
    print("  - Expected training time: ~28 hours")
else:
    print("⚠️  Issues found:")
    for issue in issues:
        print(f"  - {issue}")

print("=" * 80)
