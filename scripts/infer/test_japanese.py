"""Test Japanese fine-tuned model

This script tests the Japanese fine-tuned teacher model by generating speech
from Japanese text.

Usage:
    uv run python scripts/infer/test_japanese.py
    uv run python scripts/infer/test_japanese.py --text "こんにちは、世界！"
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from smalltts.data.phonemization.phonemes import set_language, get_token_ids
from smalltts.models.backbone.model import Backbone
from smalltts.codec.onnx import Decoder

# Set Japanese-only mode (matching training)
set_language("ja")

# Configuration
CHECKPOINT_PATH = "assets/teacher_checkpoints_ja/checkpoint_final.pt"
OUTPUT_DIR = "out"
NUM_INFERENCE_STEPS = 128  # Teacher model uses 128 steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str, device: str) -> Backbone:
    """Load fine-tuned model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model = Backbone(latent_dim=64).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[OK] Model loaded on {device}")
    return model


def get_alpha_sigma(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Get noise schedule parameters"""
    alpha = torch.cos(t * torch.pi / 2)
    sigma = torch.sin(t * torch.pi / 2)
    return alpha, sigma


@torch.no_grad()
def generate_speech(
    model: Backbone,
    text: str,
    device: str,
    num_steps: int = 128,
    cfg_scale: float = 3.0,
) -> torch.Tensor:
    """Generate speech from text using diffusion model

    Args:
        model: Fine-tuned Backbone model
        text: Input Japanese text
        device: Device to run on
        num_steps: Number of diffusion steps
        cfg_scale: Classifier-free guidance scale

    Returns:
        Generated latents (shape: [1, T, 64])
    """
    # Get phoneme tokens
    phoneme_tokens = get_token_ids(text)
    phonemes = torch.tensor([phoneme_tokens], dtype=torch.int64).to(device)

    # Create masks
    batch_size = 1
    phoneme_length = phonemes.shape[1]
    phonemes_mask = torch.ones(batch_size, phoneme_length, dtype=torch.bool).to(device)

    # Estimate latent length (based on JVS data analysis: ~0.8 latent frames per phoneme)
    # Training data shows: 80 phonemes → 65 frames, 92 phonemes → 68 frames
    latent_length = int(phoneme_length * 0.8)

    # Initialize random latents
    latents = torch.randn(batch_size, latent_length, 64).to(device)
    mask = torch.ones(batch_size, latent_length, dtype=torch.bool).to(device)

    # No conditioning (zero-shot)
    cond = torch.zeros(batch_size, latent_length, 64).to(device)
    cond_mask = torch.zeros(batch_size, latent_length, 64, dtype=torch.bool).to(device)

    # Diffusion sampling
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1).to(device)

    print(f"\nGenerating speech for: '{text}'")
    print(f"Phonemes: {len(phoneme_tokens)}, Latent frames: {latent_length}")

    for i in tqdm(range(num_steps), desc="Diffusion steps"):
        t = timesteps[i].expand(batch_size)
        t_next = timesteps[i + 1].expand(batch_size)

        # Predict velocity with CFG
        # Conditional prediction
        v_cond = model(
            latents,
            cond,
            mask,
            phonemes,
            phonemes_mask,
            t,
        )

        # Unconditional prediction (zero phonemes)
        phonemes_uncond = torch.zeros_like(phonemes)
        phonemes_mask_uncond = torch.zeros_like(phonemes_mask)

        v_uncond = model(
            latents,
            cond,
            mask,
            phonemes_uncond,
            phonemes_mask_uncond,
            t,
        )

        # Classifier-free guidance
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Update latents
        alpha_t, sigma_t = get_alpha_sigma(t)
        alpha_next, sigma_next = get_alpha_sigma(t_next)

        alpha_t = alpha_t.view(-1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1)
        alpha_next = alpha_next.view(-1, 1, 1)
        sigma_next = sigma_next.view(-1, 1, 1)

        # Predict x0 from velocity
        x0 = alpha_t * latents - sigma_t * v

        # DDIM step
        latents = alpha_next * x0 + sigma_next * v

    print(f"[OK] Generation complete")
    print(f"Generated latents shape: {latents.shape}")
    print(f"Latents stats: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}, min={latents.min().item():.4f}, max={latents.max().item():.4f}")
    return latents


def main():
    parser = argparse.ArgumentParser(description="Test Japanese fine-tuned model")
    parser.add_argument(
        "--text",
        type=str,
        default="こんにちは、世界！",
        help="Japanese text to synthesize",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT_PATH,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=NUM_INFERENCE_STEPS,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("JAPANESE TTS TEST")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Text: {args.text}")
    print(f"Diffusion steps: {args.steps}")
    print(f"CFG scale: {args.cfg_scale}")
    print("=" * 80 + "\n")

    # Load model
    model = load_model(args.checkpoint, DEVICE)

    # Load decoder
    print("Loading VibeVoice decoder...")
    decoder = Decoder()
    print("[OK] Decoder loaded\n")

    # Generate latents
    latents = generate_speech(
        model,
        args.text,
        DEVICE,
        num_steps=args.steps,
        cfg_scale=args.cfg_scale,
    )

    # Decode to audio
    print("\nDecoding latents to audio...")
    with torch.no_grad():
        audio = decoder.decode(latents.cpu())  # [1, 1, T]
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze(0)  # [1, T]
        else:
            audio = torch.from_numpy(audio).squeeze(0)  # [1, T]

    # Save audio
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "japanese_test.wav"
    torchaudio.save(str(output_path), audio, 24000)

    print(f"[OK] Audio saved to: {output_path}")
    print(f"   Duration: {audio.shape[1] / 24000:.2f} seconds")
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
