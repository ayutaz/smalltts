"""Train Teacher model on Japanese JVS dataset from scratch

This script trains a Japanese Teacher model from scratch using the JVS corpus.
It uses Japanese-only phoneme vocabulary (205 tokens for multilingual support).

Usage:
    # Single GPU
    uv run accelerate launch scripts/train/teacher_japanese.py

    # Multi-GPU
    uv run accelerate launch --multi-gpu scripts/train/teacher_japanese.py

Requirements:
    - JVS dataset downloaded and prepared (data/jvs_ver1)
    - VibeVoice encoder checkpoint (auto-downloaded)

Configuration:
    - JVS_ROOT_DIR: Path to JVS dataset directory
    - NUM_STEPS: Total training steps (default: 10,000 for testing, 100,000 recommended for production)
    - BATCH_SIZE and NUM_WORKERS: Adjust for your hardware
"""

from typing import Tuple
from pathlib import Path

import torch
from accelerate import Accelerator
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import mse_loss
from tqdm import tqdm

from smalltts.data.japanese import get_jvs_dataloader
from smalltts.data.phonemization.phonemes import set_language, phoneme_len
from smalltts.models.backbone.model import Backbone
from smalltts.models.utils import adapt_state_dict_for_expanded_phonemes
from smalltts.train.utils import get_alpha_sigma, get_mask, get_random_cond
from smalltts.codec.onnx import Encoder

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths - Multi-speaker mode (all 100 JVS speakers)
JVS_ROOT_DIR = "data/jvs_ver1"
JVS_CACHE_DIR = "data/jvs_ver1_latents"  # Pre-cached latents directory
SPEAKER_IDS = None  # None = use all speakers (jvs001-jvs100)
SUBSET = "parallel100"  # JVS subset to use

# Training parameters
BATCH_SIZE = 10  # Reduced for gradient accumulation (effective batch size = 10 × 2 = 20)
NUM_WORKERS = 4  # Can use multiple workers when loading from cache (no ONNX encoding)
NUM_STEPS = 300_000  # Total training steps (300k - reduced training time)
NUM_SAVE_STEPS = 50_000  # Save checkpoint every 50,000 steps

# Checkpoint paths
LOAD_FROM_CHECKPOINT = None  # Training from scratch with corrected accent information
OUTPUT_DIR = "assets/teacher_checkpoints_ja_accent_corrected"  # Corrected accent nucleus position
RESUME_FROM_STEP = 0  # Starting from step 0

# Training from scratch learning rate
LEARNING_RATE = 1e-4  # Standard training rate
WARMUP_STEPS = 10_000  # Warmup steps (10,000 steps, then cosine annealing for 290,000 steps)
WEIGHT_DECAY = 1e-2

# Phoneme vocabulary settings (Japanese-only with comprehensive prosodic information)
# Vocabulary size: 92 tokens (updated with Style-Bert-VITS2 features)
#   - 64 base phonemes
#   - 1 accent nucleus marker (↓)
#   - 2 pitch markers ([, ])
#   - 3 phrase boundaries (|, #, [BG])
#   - 3 sentence boundaries (^, $, ?)
#   - 3 pause markers ([P1], [P2], [P3])
#   - 6 POS tags ([N], [V], [ADJ], [PART], [AUX], [SYM])
#   - 10 mora count markers ([M1]-[M10])
VOCAB_SIZE = 92  # Japanese phonemes + comprehensive prosodic markers + Style-Bert-VITS2 features

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def get_noised_latents(
    latents: Float[Tensor, "batch sequence_length latent_dim"],
    timesteps: Float[Tensor, "batch"],
) -> Tuple[
    Float[Tensor, "batch sequence_length latent_dim"],
    Float[Tensor, "batch sequence_length latent_dim"],
]:
    """Apply noise to latents for diffusion training"""
    alpha, sigma = get_alpha_sigma(timesteps)
    noise = torch.randn_like(latents)
    alpha = alpha.view(-1, 1, 1)
    sigma = sigma.view(-1, 1, 1)
    noised = alpha * latents + sigma * noise
    true_velocity = alpha * noise - sigma * latents
    return noised, true_velocity


# Note: load_and_adapt_checkpoint function removed - training from scratch with accent info


if __name__ == "__main__":
    print("=" * 80)
    print("JAPANESE TEACHER MODEL TRAINING WITH COMPREHENSIVE PROSODIC INFORMATION")
    print("=" * 80)

    # Set language to Japanese-only mode (train from scratch)
    print("\n[1/6] Setting up Japanese phoneme vocabulary with prosodic markers")
    set_language("ja")
    print(f"Phoneme vocabulary size: {phoneme_len}")
    print("Prosodic features: Accent nucleus, Phrase boundaries, POS tags, Mora counts, Pauses")

    # Check if cached latents are available
    print("\n[2/6] Checking for cached latents")
    from pathlib import Path
    cache_dir = Path(JVS_CACHE_DIR)
    use_cache = cache_dir.exists()

    if use_cache:
        print(f"[OK] Using cached latents from: {JVS_CACHE_DIR}")
        print("[INFO] Skipping ONNX encoder initialization (using cache)")
        encoder = None
    else:
        print(f"[WARN] Cache directory not found: {JVS_CACHE_DIR}")
        print("[INFO] Loading VibeVoice encoder for on-the-fly encoding")
        encoder = Encoder()

    # Initialize dataloader
    print("\n[3/6] Loading JVS dataset (multi-speaker)")
    print(f"JVS root directory: {JVS_ROOT_DIR}")
    print(f"Cache directory: {JVS_CACHE_DIR if use_cache else 'None (on-the-fly encoding)'}")
    print(f"Speaker IDs: {'All speakers' if SPEAKER_IDS is None else SPEAKER_IDS}")
    print(f"Subset: {SUBSET}")

    train_loader = get_jvs_dataloader(
        root_dir=JVS_ROOT_DIR,
        speaker_ids=SPEAKER_IDS,
        subset=SUBSET,
        codec_encoder=encoder,
        cache_dir=JVS_CACHE_DIR if use_cache else None,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    # Initialize accelerator (FP32 - FP16 was tested but slower on this setup)
    # gradient_accumulation_steps=2: effective batch size = 10 × 2 = 20
    print("\n[4/6] Setting up distributed training")
    accelerator = Accelerator(gradient_accumulation_steps=2)

    # Initialize model with Japanese vocabulary
    print("\n[5/6] Initializing model")
    model = Backbone(latent_dim=64).to(accelerator.device)

    # Note: torch.compile() is incompatible with jaxtyping decorators
    # When using cached latents, data loading is much faster so GPU becomes the bottleneck

    # Training from scratch with comprehensive prosodic information
    print("\n[6/6] Training from scratch with comprehensive prosodic information")
    print(f"Phoneme vocabulary size: {phoneme_len} (includes all prosodic markers)")
    print("Features: Accent, Boundaries, POS, Mora count, Pauses")
    checkpoint_step = 0

    # Setup optimizer with lower learning rate for fine-tuning
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Batch size: {BATCH_SIZE} (per GPU)")
    print(f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
    print(f"Effective batch size: {BATCH_SIZE * accelerator.gradient_accumulation_steps * accelerator.num_processes}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Total steps: {NUM_STEPS:,}")
    print(f"Warmup steps: {WARMUP_STEPS:,}")
    print(f"Save interval: {NUM_SAVE_STEPS:,} steps")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY,
    )

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,  # Start from 1e-3 of learning rate
        end_factor=1.0,
        total_iters=WARMUP_STEPS,
    )

    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_STEPS - WARMUP_STEPS,
        eta_min=1e-6,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_STEPS],
    )

    # Prepare for distributed training
    train_loader, model, scheduler, optimizer = accelerator.prepare(
        train_loader, model, scheduler, optimizer
    )

    # Create training iterator
    train = iter(train_loader)

    # Create output directory
    if accelerator.is_main_process:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Training loop (starting from step 0 with accent-enhanced phonemization)
    start_step = 0
    pbar = tqdm(range(start_step, NUM_STEPS), desc="Training", disable=not accelerator.is_main_process)

    for step in pbar:
        with accelerator.accumulate(model):
            try:
                batch = next(train)
            except StopIteration:
                # Reset iterator when dataset is exhausted
                train = iter(train_loader)
                batch = next(train)

            phonemes = batch["phonemes"].to(accelerator.device)
            phonemes_lengths = batch["phonemes_lengths"].to(accelerator.device)
            batch_size = phonemes.shape[0]
            phonemes_mask = get_mask(
                batch_size, phonemes.shape[1], phonemes_lengths, accelerator.device
            )

            latent_lengths = batch["latents_lengths"].to(accelerator.device)

            # Classifier-free guidance: 10% probability to drop phoneme conditioning
            cfg_mask = torch.rand(phonemes.shape[0], device=accelerator.device) < 0.1
            phonemes[cfg_mask] = 0
            phonemes_mask[cfg_mask] = False

            latents = batch["latents"].to(accelerator.device)

            # Sample random timesteps and add noise
            timesteps = torch.rand(batch_size).to(accelerator.device)
            noised, true_velocity = get_noised_latents(latents, timesteps)

            # Get random conditioning (partial audio for context)
            cond, cond_mask = get_random_cond(latents, latent_lengths, accelerator.device)
            mask = get_mask(
                batch_size, latents.shape[1], latent_lengths, accelerator.device
            )

            # Forward pass
            velocity = model(
                noised,
                cond,
                mask,
                phonemes,
                phonemes_mask,
                timesteps,
            )

            # Compute loss only on valid (non-padding, non-conditioning) positions
            valid = mask.unsqueeze(-1).expand(-1, -1, 64) * ~cond_mask

            velocity = velocity * valid
            true_velocity = true_velocity * valid

            # Backpropagation with gradient accumulation
            optimizer.zero_grad()
            loss = mse_loss(velocity, true_velocity, reduction="sum") / valid.sum()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

        # Logging
        gathered_loss = accelerator.gather(loss).mean().item()  # type: ignore
        accelerator.log({"teacher_loss_ja": gathered_loss}, step=step)
        pbar.set_postfix(loss=f"{gathered_loss:.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if accelerator.is_main_process and step % NUM_SAVE_STEPS == 0 and step > 0:
            print(f"\n[SAVE] Saving checkpoint at step {step}")

            # Save full accelerator state (for resuming training)
            checkpoint_dir = f"{OUTPUT_DIR}/checkpoint_step_{step}"
            accelerator.save_state(checkpoint_dir)

            # Save model weights only (for inference)
            torch.save(
                {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "step": step,
                    "loss": gathered_loss,
                },
                f"{OUTPUT_DIR}/checkpoint_step_{step}.pt",
            )

            # Also save as latest
            torch.save(
                {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "step": step,
                    "loss": gathered_loss,
                },
                f"{OUTPUT_DIR}/checkpoint_latest.pt",
            )

            print(f"[OK] Checkpoint saved to {OUTPUT_DIR}")

        # Clean up
        del batch, noised, cond_mask, cond, mask, velocity, true_velocity, loss

    # Final save
    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"[SAVE] Saving final checkpoint")

        accelerator.save_state(f"{OUTPUT_DIR}/checkpoint_final")
        torch.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                "step": NUM_STEPS,
            },
            f"{OUTPUT_DIR}/checkpoint_final.pt",
        )

        print(f"[OK] Final checkpoint saved to {OUTPUT_DIR}")
        print("=" * 80)
