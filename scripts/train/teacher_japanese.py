"""Fine-tune Teacher model on Japanese JVS dataset

This script fine-tunes an existing English Teacher model on Japanese data.
It automatically expands the phoneme embeddings from 175 tokens (English) to 205 tokens (multilingual).

Usage:
    # Single GPU
    uv run accelerate launch scripts/train/teacher_japanese.py

    # Multi-GPU
    uv run accelerate launch --multi-gpu scripts/train/teacher_japanese.py

Requirements:
    - JVS dataset downloaded and prepared
    - VibeVoice encoder checkpoint (auto-downloaded)
    - English teacher checkpoint (auto-downloaded or specify path)

Configuration:
    - Set AUDIO_DIR and TRANSCRIPT_FILE to point to your JVS dataset
    - Set LOAD_FROM_CHECKPOINT to the English checkpoint path
    - Adjust BATCH_SIZE and NUM_WORKERS for your hardware
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
SPEAKER_IDS = None  # None = use all speakers (jvs001-jvs100)
SUBSET = "parallel100"  # JVS subset to use

# Training parameters
BATCH_SIZE = 1
NUM_WORKERS = 0  # Must be 0 when using ONNX encoder (CUDA context issue)
NUM_STEPS = 100_000  # Training from scratch requires more steps (100 speakers × 100 utterances = 10,000 samples)
NUM_SAVE_STEPS = 5_000

# Checkpoint paths
LOAD_FROM_CHECKPOINT = None  # Train from scratch (Japanese only)
OUTPUT_DIR = "assets/teacher_checkpoints_ja"

# Training from scratch learning rate
LEARNING_RATE = 1e-4  # Standard training rate
WARMUP_STEPS = 1_000  # Warmup for first 10% of training
WEIGHT_DECAY = 1e-2

# Phoneme vocabulary settings
OLD_VOCAB_SIZE = 175  # English-only
NEW_VOCAB_SIZE = 205  # Multilingual (English + Japanese)

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


def load_and_adapt_checkpoint(checkpoint_path: str, device: str) -> dict:
    """Load checkpoint and adapt phoneme embeddings for multilingual support

    Args:
        checkpoint_path: Path to English checkpoint
        device: Device to load checkpoint on

    Returns:
        Adapted state dictionary
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    print(f"Adapting phoneme embeddings: {OLD_VOCAB_SIZE} → {NEW_VOCAB_SIZE} tokens")
    adapted_state_dict = adapt_state_dict_for_expanded_phonemes(
        state_dict=state_dict,
        old_vocab_size=OLD_VOCAB_SIZE,
        new_vocab_size=NEW_VOCAB_SIZE,
        embedding_key="phoneme_embedding.phoneme_embed.weight",
        initialization="mean",  # Use mean of existing embeddings (better than zeros)
    )

    print("✅ Checkpoint adapted successfully")
    return adapted_state_dict


if __name__ == "__main__":
    print("=" * 80)
    print("JAPANESE TEACHER MODEL TRAINING (FROM SCRATCH)")
    print("=" * 80)

    # Set language to Japanese-only mode (train from scratch)
    print("\n[1/6] Setting up Japanese phoneme vocabulary")
    set_language("ja")
    print(f"Phoneme vocabulary size: {phoneme_len}")

    # Initialize codec encoder
    print("\n[2/6] Loading VibeVoice encoder")
    encoder = Encoder()

    # Initialize dataloader
    print("\n[3/6] Loading JVS dataset (multi-speaker)")
    print(f"JVS root directory: {JVS_ROOT_DIR}")
    print(f"Speaker IDs: {'All speakers' if SPEAKER_IDS is None else SPEAKER_IDS}")
    print(f"Subset: {SUBSET}")

    train_loader = get_jvs_dataloader(
        root_dir=JVS_ROOT_DIR,
        speaker_ids=SPEAKER_IDS,
        subset=SUBSET,
        codec_encoder=encoder,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    # Initialize accelerator
    print("\n[4/6] Setting up distributed training")
    accelerator = Accelerator()

    # Initialize model with Japanese vocabulary
    print("\n[5/6] Initializing model")
    model = Backbone(latent_dim=64).to(accelerator.device)

    # Load and adapt checkpoint if specified
    if LOAD_FROM_CHECKPOINT is not None:
        print("\n[6/6] Loading and adapting English checkpoint")
        adapted_state_dict = load_and_adapt_checkpoint(
            LOAD_FROM_CHECKPOINT,
            accelerator.device
        )

        # Load adapted weights
        missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)

        if missing_keys:
            print(f"⚠️  Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

        print("✅ Checkpoint loaded successfully")
    else:
        print("\n[6/6] Skipping checkpoint loading (training from scratch)")

    # Setup optimizer with lower learning rate for fine-tuning
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Batch size: {BATCH_SIZE}")
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

    train = iter(train_loader)

    # Create output directory
    if accelerator.is_main_process:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Training loop
    pbar = tqdm(range(0, NUM_STEPS), desc="Training", disable=not accelerator.is_main_process)

    for step in pbar:
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

        # Backpropagation
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
            print(f"\n💾 Saving checkpoint at step {step}")

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

            print(f"✅ Checkpoint saved to {OUTPUT_DIR}")

        # Clean up
        del batch, noised, cond_mask, cond, mask, velocity, true_velocity, loss

    # Final save
    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"💾 Saving final checkpoint")

        accelerator.save_state(f"{OUTPUT_DIR}/checkpoint_final")
        torch.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                "step": NUM_STEPS,
            },
            f"{OUTPUT_DIR}/checkpoint_final.pt",
        )

        print(f"✅ Final checkpoint saved to {OUTPUT_DIR}")
        print("=" * 80)
