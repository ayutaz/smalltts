"""Find optimal batch size for Teacher model training on available hardware

This script tests different batch sizes to find the maximum that fits in VRAM.
It runs a few training steps for each batch size and monitors GPU memory usage.

Usage:
    # Single GPU test
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train/find_optimal_batch_size.py

    # Multi-GPU test (recommended)
    uv run accelerate launch --multi-gpu scripts/train/find_optimal_batch_size.py
"""

from pathlib import Path
import torch
from accelerate import Accelerator
from tqdm import tqdm

from smalltts.data.japanese import get_jvs_dataloader
from smalltts.data.phonemization.phonemes import set_language, phoneme_len
from smalltts.models.backbone.model import Backbone
from smalltts.train.utils import get_alpha_sigma, get_mask, get_random_cond
from torch.nn.functional import mse_loss

# ============================================================================
# CONFIGURATION
# ============================================================================

JVS_ROOT_DIR = "data/jvs_ver1"
JVS_CACHE_DIR = "data/jvs_ver1_latents"
SPEAKER_IDS = None
SUBSET = "parallel100"
NUM_WORKERS = 4

# Batch sizes to test (will stop at first OOM)
BATCH_SIZES_TO_TEST = [20, 21, 22, 23, 24, 25]

# Number of steps to test each batch size
TEST_STEPS = 5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_noised_latents(latents, timesteps):
    """Apply noise to latents for diffusion training"""
    alpha, sigma = get_alpha_sigma(timesteps)
    noise = torch.randn_like(latents)
    alpha = alpha.view(-1, 1, 1)
    sigma = sigma.view(-1, 1, 1)
    noised = alpha * latents + sigma * noise
    true_velocity = alpha * noise - sigma * latents
    return noised, true_velocity


def test_batch_size(batch_size: int, accelerator: Accelerator) -> dict:
    """Test a specific batch size for TEST_STEPS steps

    Returns:
        dict with keys:
            - success: bool
            - max_memory_gb: float (peak GPU memory used)
            - error: str or None
    """
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print(f"TESTING BATCH_SIZE = {batch_size}")
        print(f"{'='*80}")

    try:
        # Initialize dataloader
        train_loader = get_jvs_dataloader(
            root_dir=JVS_ROOT_DIR,
            speaker_ids=SPEAKER_IDS,
            subset=SUBSET,
            codec_encoder=None,
            cache_dir=JVS_CACHE_DIR,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            shuffle=True,
        )

        # Initialize model
        model = Backbone(latent_dim=64).to(accelerator.device)

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )

        # Prepare for distributed training
        train_loader, model, optimizer = accelerator.prepare(
            train_loader, model, optimizer
        )

        train = iter(train_loader)

        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(accelerator.device)

        # Run test steps
        for step in range(TEST_STEPS):
            try:
                batch = next(train)
            except StopIteration:
                train = iter(train_loader)
                batch = next(train)

            phonemes = batch["phonemes"].to(accelerator.device)
            phonemes_lengths = batch["phonemes_lengths"].to(accelerator.device)
            batch_size_actual = phonemes.shape[0]
            phonemes_mask = get_mask(
                batch_size_actual, phonemes.shape[1], phonemes_lengths, accelerator.device
            )

            latent_lengths = batch["latents_lengths"].to(accelerator.device)

            # Classifier-free guidance
            cfg_mask = torch.rand(phonemes.shape[0], device=accelerator.device) < 0.1
            phonemes[cfg_mask] = 0
            phonemes_mask[cfg_mask] = False

            latents = batch["latents"].to(accelerator.device)

            # Sample random timesteps and add noise
            timesteps = torch.rand(batch_size_actual).to(accelerator.device)
            noised, true_velocity = get_noised_latents(latents, timesteps)

            # Get random conditioning
            cond, cond_mask = get_random_cond(latents, latent_lengths, accelerator.device)
            mask = get_mask(
                batch_size_actual, latents.shape[1], latent_lengths, accelerator.device
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

            # Compute loss
            valid = mask.unsqueeze(-1).expand(-1, -1, 64) * ~cond_mask
            velocity = velocity * valid
            true_velocity = true_velocity * valid

            # Backpropagation
            optimizer.zero_grad()
            loss = mse_loss(velocity, true_velocity, reduction="sum") / valid.sum()
            accelerator.backward(loss)
            optimizer.step()

            # Log progress
            if accelerator.is_main_process:
                if torch.cuda.is_available():
                    mem_gb = torch.cuda.max_memory_allocated(accelerator.device) / 1e9
                    print(f"  Step {step+1}/{TEST_STEPS}: loss={loss.item():.3f}, VRAM={mem_gb:.2f}GB")
                else:
                    print(f"  Step {step+1}/{TEST_STEPS}: loss={loss.item():.3f}")

            # Clean up
            del batch, noised, cond_mask, cond, mask, velocity, true_velocity, loss, phonemes, phonemes_mask, latents, timesteps, valid, latent_lengths, cfg_mask
            torch.cuda.empty_cache()

        # Get peak memory usage
        if torch.cuda.is_available():
            max_memory_gb = torch.cuda.max_memory_allocated(accelerator.device) / 1e9
        else:
            max_memory_gb = 0.0

        # Clean up
        del model, optimizer, train_loader
        torch.cuda.empty_cache()

        if accelerator.is_main_process:
            print(f"\n✅ SUCCESS: batch_size={batch_size}, peak_memory={max_memory_gb:.2f}GB")

        return {
            "success": True,
            "max_memory_gb": max_memory_gb,
            "error": None,
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            if accelerator.is_main_process:
                print(f"\n❌ OOM ERROR: batch_size={batch_size} is too large")
            return {
                "success": False,
                "max_memory_gb": None,
                "error": "Out of memory",
            }
        else:
            raise
    except Exception as e:
        torch.cuda.empty_cache()
        if accelerator.is_main_process:
            print(f"\n❌ ERROR: {str(e)}")
        return {
            "success": False,
            "max_memory_gb": None,
            "error": str(e),
        }


if __name__ == "__main__":
    print("=" * 80)
    print("BATCH SIZE OPTIMIZATION TEST")
    print("=" * 80)

    # Set language to Japanese
    print("\n[1/3] Setting up Japanese phoneme vocabulary")
    set_language("ja")
    print(f"Phoneme vocabulary size: {phoneme_len}")

    # Check cache
    print("\n[2/3] Checking for cached latents")
    cache_dir = Path(JVS_CACHE_DIR)
    if not cache_dir.exists():
        print(f"ERROR: Cache directory not found: {JVS_CACHE_DIR}")
        print("Please run cache creation first:")
        print("  bash scripts/preprocess/parallel_cache.sh")
        exit(1)
    print(f"[OK] Using cached latents from: {JVS_CACHE_DIR}")

    # Initialize accelerator
    print("\n[3/3] Setting up distributed training")
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"\nNum processes: {accelerator.num_processes}")
        print(f"Device: {accelerator.device}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(accelerator.device)
            gpu_memory = torch.cuda.get_device_properties(accelerator.device).total_memory / 1e9
            print(f"GPU: {gpu_name}")
            print(f"Total VRAM: {gpu_memory:.2f}GB")

    # Test each batch size
    results = []
    optimal_batch_size = None

    print("\n" + "=" * 80)
    print("STARTING BATCH SIZE TESTS")
    print("=" * 80)
    print(f"Testing batch sizes: {BATCH_SIZES_TO_TEST}")
    print(f"Steps per test: {TEST_STEPS}")
    print("=" * 80)

    for batch_size in BATCH_SIZES_TO_TEST:
        result = test_batch_size(batch_size, accelerator)
        results.append({
            "batch_size": batch_size,
            **result
        })

        if result["success"]:
            optimal_batch_size = batch_size
        else:
            # Stop at first failure
            if accelerator.is_main_process:
                print(f"\nStopping tests (OOM encountered at batch_size={batch_size})")
            break

    # Print summary
    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)

        for result in results:
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            memory = f"{result['max_memory_gb']:.2f}GB" if result['max_memory_gb'] else "N/A"
            print(f"  batch_size={result['batch_size']:3d}: {status:12s} | Peak VRAM: {memory}")

        print("\n" + "=" * 80)
        if optimal_batch_size:
            print(f"RECOMMENDED BATCH_SIZE: {optimal_batch_size}")
            print("=" * 80)
            print(f"\nUpdate scripts/train/teacher_japanese.py:")
            print(f"  BATCH_SIZE = {optimal_batch_size}")

            # Calculate estimated training time
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(accelerator.device)
                if "T4" in gpu_name:
                    time_multiplier = 1.5  # T4 is ~1.5x slower than RTX 4070 Ti
                else:
                    time_multiplier = 1.0

                # Rough estimate based on RTX 4070 Ti SUPER baseline
                baseline_hours_per_10k = 16  # 16 hours for 10k steps on RTX 4070 Ti
                estimated_hours_100k = baseline_hours_per_10k * 10 * time_multiplier
                estimated_days = estimated_hours_100k / 24

                print(f"\nEstimated training time (100k steps):")
                print(f"  ~{estimated_hours_100k:.0f} hours (~{estimated_days:.1f} days)")
        else:
            print("ERROR: No batch size succeeded. Check your setup.")
            print("=" * 80)
