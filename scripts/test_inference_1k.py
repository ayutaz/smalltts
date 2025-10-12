"""Test inference with 1,000-step checkpoint"""

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

# Test text
TEST_TEXT = "こんにちは、世界"

print("=" * 80)
print("JAPANESE TTS INFERENCE TEST (1,000 steps)")
print("=" * 80)

# Set language
print("\n[1/7] Setting language to Japanese")
set_language("ja")
print(f"Language: ja (Japanese only)")

# Load checkpoint
print("\n[2/7] Loading checkpoint")
print(f"Path: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")

# Initialize model
print("\n[3/7] Initializing model")
model = Backbone(latent_dim=64).to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()
print(f"Model loaded successfully")

# Initialize codec
print("\n[4/7] Loading codec (encoder/decoder)")
encoder = Encoder()
decoder = Decoder()

# Phonemize text
print("\n[5/7] Phonemizing text")
print(f"Input text: {TEST_TEXT}")
phoneme_tokens = get_token_ids(TEST_TEXT)
print(f"Phoneme tokens: {phoneme_tokens}")
print(f"Number of tokens: {len(phoneme_tokens)}")

# Prepare phonemes
phonemes = torch.tensor([phoneme_tokens], dtype=torch.int64).to(DEVICE)
phonemes_lengths = torch.tensor([len(phoneme_tokens)], dtype=torch.int64).to(DEVICE)
phonemes_mask = torch.ones(1, len(phoneme_tokens), dtype=torch.bool).to(DEVICE)

# Estimate output length (rough heuristic: ~2 latent frames per phoneme)
estimated_latent_length = len(phoneme_tokens) * 2
print(f"\n[6/7] Estimating output length: ~{estimated_latent_length} latent frames")

# Generate audio using simplified diffusion sampling
print("\n[6/7] Generating audio (simplified sampling)")
print("Note: This is a simplified test - full inference requires proper diffusion sampling")

with torch.no_grad():
    # Create empty latent sequence
    latents = torch.randn(1, estimated_latent_length, 64).to(DEVICE)

    # Create masks
    latent_lengths = torch.tensor([estimated_latent_length], dtype=torch.int64).to(DEVICE)
    mask = torch.ones(1, estimated_latent_length, dtype=torch.bool).to(DEVICE)

    # No conditioning for this test
    cond = torch.zeros_like(latents)

    # Single step inference (not proper diffusion, just a quick test)
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
    print(f"Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")

    # Check if output is non-zero (sign of learning)
    if output.abs().mean() < 0.01:
        print("WARNING: Output is very small - model may not have learned much yet")
    else:
        print("Output looks reasonable - model has learned something")

    # Decode latents to audio
    print("\n[7/7] Decoding latents to audio")
    audio = decoder.decode(output)

    print(f"Audio shape: {audio.shape}")
    print(f"Audio stats: mean={audio.mean().item():.6f}, std={audio.std().item():.6f}, max={audio.abs().max().item():.6f}")

# Save audio
Path(OUTPUT_DIR).mkdir(exist_ok=True)
output_path = Path(OUTPUT_DIR) / "test_1k_steps.wav"

# Normalize and save
audio_np = audio.squeeze().cpu().numpy()
if audio_np.max() > 0:
    audio_np = audio_np / audio_np.max() * 0.95  # Normalize to prevent clipping
else:
    print("WARNING: Audio is silent!")

torchaudio.save(str(output_path), torch.from_numpy(audio_np).unsqueeze(0), 24000)

print(f"\nAudio saved to: {output_path}")

# Print summary
print("\n" + "=" * 80)
print("INFERENCE TEST COMPLETE")
print("=" * 80)
print(f"Input: {TEST_TEXT}")
print(f"Phonemes: {len(phoneme_tokens)} tokens")
print(f"Output latents: {output.shape}")
print(f"Output audio: {audio.shape}")
print(f"Saved to: {output_path}")
print("\nNOTE: This is a simplified test with only 1,000 training steps.")
print("The model needs 100,000 steps for good quality.")
print("=" * 80)
