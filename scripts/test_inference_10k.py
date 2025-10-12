"""Test inference with 10,000-step checkpoint"""

import torch
import torchaudio
from pathlib import Path
import numpy as np

from smalltts.models.backbone.model import Backbone
from smalltts.data.phonemization.phonemes import set_language, get_token_ids
from smalltts.codec.onnx import Encoder, Decoder
from smalltts.train.utils import get_mask

# Configuration
CHECKPOINT_PATH = "assets/teacher_checkpoints_ja/checkpoint_final.pt"
OUTPUT_DIR = "out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test texts
TEST_TEXTS = [
    "こんにちは、世界",
    "今日はいい天気ですね",
    "音声合成システムのテストです"
]

print("=" * 80)
print("JAPANESE TTS INFERENCE TEST (10,000 steps)")
print("=" * 80)

# Set language
print("\n[1/6] Setting language to Japanese")
set_language("ja")
print(f"Language: ja (Japanese only)")

# Load checkpoint
print("\n[2/6] Loading checkpoint")
print(f"Path: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown'):.4f}" if 'loss' in checkpoint else "Checkpoint loss: unknown")

# Initialize model
print("\n[3/6] Initializing model")
model = Backbone(latent_dim=64).to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()
print(f"Model loaded successfully")

# Initialize codec
print("\n[4/6] Loading codec (encoder/decoder)")
encoder = Encoder()
decoder = Decoder()

# Process each test text
Path(OUTPUT_DIR).mkdir(exist_ok=True)

for idx, test_text in enumerate(TEST_TEXTS, 1):
    print(f"\n{'=' * 80}")
    print(f"TEST {idx}/{len(TEST_TEXTS)}: {test_text}")
    print('=' * 80)

    # Phonemize text
    print("\n[5/6] Phonemizing text")
    phoneme_tokens = get_token_ids(test_text)
    print(f"Phoneme tokens: {phoneme_tokens[:20]}{'...' if len(phoneme_tokens) > 20 else ''}")
    print(f"Number of tokens: {len(phoneme_tokens)}")

    # Prepare phonemes
    phonemes = torch.tensor([phoneme_tokens], dtype=torch.int64).to(DEVICE)
    phonemes_lengths = torch.tensor([len(phoneme_tokens)], dtype=torch.int64).to(DEVICE)
    phonemes_mask = torch.ones(1, len(phoneme_tokens), dtype=torch.bool).to(DEVICE)

    # Estimate output length
    estimated_latent_length = len(phoneme_tokens) * 2
    print(f"Estimated latent length: {estimated_latent_length} frames")

    # Generate audio
    print("\n[6/6] Generating audio")

    with torch.no_grad():
        # Create empty latent sequence
        latents = torch.randn(1, estimated_latent_length, 64).to(DEVICE)

        # Create masks
        latent_lengths = torch.tensor([estimated_latent_length], dtype=torch.int64).to(DEVICE)
        mask = torch.ones(1, estimated_latent_length, dtype=torch.bool).to(DEVICE)

        # No conditioning for this test
        cond = torch.zeros_like(latents)

        # Single step inference (simplified test)
        timestep = torch.tensor([0.5]).to(DEVICE)

        output = model(
            latents,
            cond,
            mask,
            phonemes,
            phonemes_mask,
            timestep
        )

        print(f"Output shape: {output.shape}")
        print(f"Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}, max={output.abs().max().item():.4f}")

        # Decode latents to audio
        audio = decoder.decode(output)

        print(f"Audio shape: {audio.shape}")
        print(f"Audio stats: mean={audio.mean().item():.6f}, std={audio.std().item():.6f}, max={audio.abs().max().item():.6f}")

    # Save audio
    output_path = Path(OUTPUT_DIR) / f"test_10k_steps_{idx}.wav"

    # Normalize and save
    audio_np = audio.squeeze().cpu().numpy()
    if audio_np.max() > 0:
        audio_np = audio_np / audio_np.max() * 0.95
    else:
        print("WARNING: Audio is silent!")

    torchaudio.save(str(output_path), torch.from_numpy(audio_np).unsqueeze(0), 24000)
    print(f"\nAudio saved to: {output_path}")

# Print summary
print("\n" + "=" * 80)
print("INFERENCE TEST COMPLETE - 10,000 STEPS")
print("=" * 80)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Training steps: {checkpoint.get('step', 'unknown')}")
print(f"Generated {len(TEST_TEXTS)} audio files:")
for idx in range(1, len(TEST_TEXTS) + 1):
    print(f"  - out/test_10k_steps_{idx}.wav")
print("\nNOTE: This is still early in training (10k/100k steps).")
print("Quality will improve significantly with more training.")
print("=" * 80)
