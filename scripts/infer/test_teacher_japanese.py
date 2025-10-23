"""Test trained Japanese Teacher model inference

Usage:
    uv run python scripts/infer/test_teacher_japanese.py --checkpoint assets/teacher_checkpoints_ja/checkpoint_final.pt --text "こんにちは、これはテストです。" --reference data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav --transcription "大学で機械学習を勉強しています。"
"""

import argparse
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from smalltts.codec.onnx import Encoder, Decoder
from smalltts.data.phonemization.phonemes import set_language, get_token_ids, decode_token_ids
from smalltts.infer.utils import resample_hq
from smalltts.models.backbone.model import Backbone
from smalltts.train.utils import get_alpha_sigma, get_mask


def estimate_duration_with_prosody(
    phoneme_tokens: list,
    reference_audio_duration_sec: float,
    reference_phoneme_tokens: list,
    codec_fps: float,
) -> int:
    """Estimate target duration considering all phoneme tokens equally

    Args:
        phoneme_tokens: Target phoneme token IDs (including prosodic markers)
        reference_audio_duration_sec: Reference audio duration in seconds
        reference_phoneme_tokens: Reference phoneme token IDs (including prosodic markers)
        codec_fps: Codec frame rate (latent frames per second)

    Returns:
        Estimated target length in latent frames
    """
    # Calculate duration based on phoneme ratio
    if len(reference_phoneme_tokens) > 0:
        base_duration_sec = reference_audio_duration_sec * len(phoneme_tokens) / len(reference_phoneme_tokens)
    else:
        base_duration_sec = reference_audio_duration_sec

    # Apply a slight slowdown factor to match reference speaking rate (4.52 chars/sec)
    # Current issue: generated speech is too fast (5.5+ chars/sec)
    # Slowdown factor: ~1.2x (to bring 5.5 -> 4.6 chars/sec)
    adjusted_duration_sec = base_duration_sec * 1.2

    # Convert to latent frames
    adjusted_duration_frames = int(adjusted_duration_sec * codec_fps)

    # Ensure minimum length
    adjusted_duration_frames = max(10, adjusted_duration_frames)

    print(f"Duration estimation:")
    print(f"  Token count: {len(phoneme_tokens)} (ref: {len(reference_phoneme_tokens)})")
    print(f"  Reference audio duration: {reference_audio_duration_sec:.2f} sec")
    print(f"  Codec frame rate: {codec_fps:.2f} fps")
    print(f"  Base duration: {base_duration_sec:.2f} sec")
    print(f"  Adjusted duration (1.2x slowdown): {adjusted_duration_sec:.2f} sec = {adjusted_duration_frames} frames")

    return adjusted_duration_frames


def load_teacher_model(checkpoint_path: str, device: str = "cuda") -> Backbone:
    """Load trained Teacher model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")

    # Set language to Japanese
    set_language("ja")

    # Initialize model
    model = Backbone(latent_dim=64).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        step = checkpoint.get("step", "unknown")
        loss = checkpoint.get("loss", "unknown")
        print(f"Loaded checkpoint from step {step}, loss: {loss}")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.no_grad()
def generate_speech(
    model: Backbone,
    encoder: Encoder,
    decoder: Decoder,
    reference_audio_path: str,
    reference_transcription: str,
    target_text: str,
    num_steps: int = 128,
    cfg_scale: float = 3.0,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Generate speech using Teacher model"""
    # Ensure device is torch.device object
    if isinstance(device, str):
        device = torch.device(device)

    # Load and encode reference audio
    print(f"Loading reference audio: {reference_audio_path}")
    y, sr = sf.read(reference_audio_path, dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)

    x = torch.from_numpy(y).view(1, -1)
    x = resample_hq(x, sr, 24_000)

    print("Encoding reference audio...")
    print(f"Audio shape before encoding: {x.shape} ({x.shape[1]/24000:.2f} seconds)")
    reference_latents = encoder.encode(x.unsqueeze(0))[0].to(device)  # [T, 64] or [1, T, 64]
    print(f"Encoder output shape: {reference_latents.shape}")

    # Ensure reference_latents has shape [1, T, 64]
    if reference_latents.ndim == 2:
        reference_latents = reference_latents.unsqueeze(0)  # [1, T, 64]

    # Calculate codec frame rate (fps)
    reference_audio_duration_sec = x.shape[1] / 24000
    codec_fps = reference_latents.shape[1] / reference_audio_duration_sec

    print(f"Reference latents shape: {reference_latents.shape} ({reference_audio_duration_sec:.2f} sec, {codec_fps:.2f} fps)")
    print(f"Reference latents - mean: {reference_latents.mean():.4f}, std: {reference_latents.std():.4f}, min/max: {reference_latents.min():.4f}/{reference_latents.max():.4f}")

    # Phonemize texts
    print("Phonemizing texts...")
    reference_phonemes = get_token_ids(reference_transcription)
    target_phonemes = get_token_ids(target_text)

    print(f"Reference phonemes: {len(reference_phonemes)}")
    print(f"Target phonemes: {len(target_phonemes)}")

    # Prepare inputs
    # For voice cloning, we use the reference latents as conditioning
    # and generate latents for the target text

    # Estimate target length with prosodic marker consideration
    target_length = estimate_duration_with_prosody(
        target_phonemes,
        reference_audio_duration_sec,
        reference_phonemes,
        codec_fps
    )

    # Create phoneme tensors
    max_phoneme_len = max(len(reference_phonemes), len(target_phonemes))
    phonemes = torch.zeros(1, max_phoneme_len, dtype=torch.long, device=device)
    phonemes[0, :len(target_phonemes)] = torch.tensor(target_phonemes, dtype=torch.long)

    phonemes_lengths = torch.tensor([len(target_phonemes)], device=device)
    phonemes_mask = get_mask(1, max_phoneme_len, phonemes_lengths, device)

    # Create FIXED UNNOISED conditioning from reference audio
    # Use 30% of TARGET length (matching training distribution: 0-50% of target, avg ~25%)
    # Training uses get_random_cond() which conditions on 0-50% of TARGET length
    cond_length = int(target_length * 0.3)  # 30% of target length
    cond_length = max(1, min(cond_length, reference_latents.shape[1]))  # Clamp to available reference

    # Create conditioning tensor (UNNOISED, stays fixed throughout sampling)
    cond = torch.zeros(1, target_length, 64, device=device)
    if reference_latents.shape[1] < cond_length:
        # Pad reference if too short
        cond[:, :reference_latents.shape[1], :] = reference_latents
    else:
        # Use beginning of reference
        cond[:, :cond_length, :] = reference_latents[:, :cond_length, :]

    print(f"Conditioning length: {cond_length} frames ({cond_length/codec_fps:.2f} seconds, {cond_length/target_length*100:.1f}% of target)")

    # Initialize ALL latents with noise (including conditioning region)
    latents = torch.randn(1, target_length, 64, device=device)

    latent_lengths = torch.tensor([target_length], device=device)
    mask = get_mask(1, target_length, latent_lengths, device)

    # Diffusion sampling (DDIM-style)
    print(f"Generating with {num_steps} diffusion steps...")
    print(f"Initial latents - mean: {latents.mean():.4f}, std: {latents.std():.4f}, min/max: {latents.min():.4f}/{latents.max():.4f}")
    print(f"Conditioning - mean: {cond.mean():.4f}, std: {cond.std():.4f}, nonzero: {(cond != 0).sum().item()}/{cond.numel()}")
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in tqdm(range(num_steps), desc="Sampling"):
        t_curr = timesteps[i].view(1)
        t_next = timesteps[i + 1].view(1)

        # NOTE: `cond` is FIXED (unnoised reference) and does NOT change during sampling
        # This matches training where cond = data * cond_mask (unnoised ground truth)
        # while latents are noised and updated each step

        # Predict velocity with CFG
        # Conditional prediction
        velocity_cond = model(
            latents, cond, mask, phonemes, phonemes_mask, t_curr
        )

        # Unconditional prediction (zero phonemes)
        phonemes_uncond = torch.zeros_like(phonemes)
        phonemes_mask_uncond = torch.zeros_like(phonemes_mask)
        velocity_uncond = model(
            latents, cond, mask, phonemes_uncond, phonemes_mask_uncond, t_curr
        )

        # Classifier-free guidance
        velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)

        # Get alpha and sigma for current and next timesteps
        alpha_curr, sigma_curr = get_alpha_sigma(t_curr)
        alpha_next, sigma_next = get_alpha_sigma(t_next)

        # DDIM update
        alpha_curr = alpha_curr.view(1, 1, 1)
        sigma_curr = sigma_curr.view(1, 1, 1)
        alpha_next = alpha_next.view(1, 1, 1)
        sigma_next = sigma_next.view(1, 1, 1)

        # Predict x0 from velocity: v = α * ε - σ * x₀ → x₀ = α * x_t - σ * v
        x0_pred = alpha_curr * latents - sigma_curr * velocity

        # DDIM step to next timestep
        if i < num_steps - 1:
            # Convert velocity to noise: v = α * ε - σ * x₀ → ε = (v + σ * x₀) / α
            noise_pred = (velocity + sigma_curr * x0_pred) / alpha_curr
            # DDIM step: x_{t-1} = α_{t-1} * x₀ + σ_{t-1} * ε
            latents = alpha_next * x0_pred + sigma_next * noise_pred
        else:
            latents = x0_pred

    # Check final latents
    print(f"\nFinal latents - mean: {latents.mean():.4f}, std: {latents.std():.4f}, min/max: {latents.min():.4f}/{latents.max():.4f}")

    # Decode latents to audio
    print("Decoding latents to audio...")
    audio = decoder.decode(latents)[0]  # latents is already [1, T, 64], decoder returns [batch, 1, T_audio]

    return audio


def main():
    parser = argparse.ArgumentParser(description="Test trained Japanese Teacher model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="assets/teacher_checkpoints_ja/checkpoint_final.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference audio file",
    )
    parser.add_argument(
        "--transcription",
        type=str,
        required=True,
        help="Transcription of reference audio (Japanese)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Target text to synthesize (Japanese)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="out/teacher_output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Convert device to torch.device
    device = torch.device(args.device)

    # Load model
    model = load_teacher_model(args.checkpoint, str(device))

    # Initialize encoder and decoder
    print("Loading codec...")
    encoder = Encoder()
    decoder = Decoder()

    # Generate speech
    audio = generate_speech(
        model=model,
        encoder=encoder,
        decoder=decoder,
        reference_audio_path=args.reference,
        reference_transcription=args.transcription,
        target_text=args.text,
        num_steps=args.steps,
        cfg_scale=args.cfg_scale,
        device=device,
    )

    # Save output
    print(f"Saving output to: {args.output}")
    sf.write(args.output, audio.squeeze(0).t().cpu().numpy(), 24_000, subtype="PCM_16")
    print("Done!")


if __name__ == "__main__":
    main()
