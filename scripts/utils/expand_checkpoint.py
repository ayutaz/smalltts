"""Expand phoneme embeddings in checkpoint for multilingual support

This script adapts an existing checkpoint trained with English-only phonemes (175 tokens)
to support multilingual phonemes (205 tokens) by expanding the phoneme embedding layer.

Usage:
    python scripts/utils/expand_checkpoint.py \
        --input assets/teacher_checkpoints/checkpoint_latest.pt \
        --output assets/teacher_checkpoints/checkpoint_multilingual.pt \
        --old-vocab-size 175 \
        --new-vocab-size 205 \
        --initialization zeros

Args:
    --input: Path to input checkpoint file
    --output: Path to output checkpoint file
    --old-vocab-size: Original vocabulary size (default: 175 for English)
    --new-vocab-size: Target vocabulary size (default: 205 for multilingual)
    --initialization: How to initialize new embeddings (zeros, mean, or random)
"""

import argparse
import logging
from pathlib import Path

import torch

from smalltts.models.utils import adapt_state_dict_for_expanded_phonemes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def expand_checkpoint(
    input_path: str,
    output_path: str,
    old_vocab_size: int = 175,
    new_vocab_size: int = 205,
    initialization: str = "zeros",
):
    """Expand phoneme embeddings in checkpoint

    Args:
        input_path: Path to input checkpoint
        output_path: Path to save expanded checkpoint
        old_vocab_size: Original vocabulary size
        new_vocab_size: Target vocabulary size
        initialization: Initialization method for new embeddings
    """
    logger.info("=" * 80)
    logger.info("CHECKPOINT PHONEME EMBEDDING EXPANSION")
    logger.info("=" * 80)

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")

    # Check checkpoint structure
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        logger.info("Checkpoint contains 'model' key (training checkpoint)")
    else:
        state_dict = checkpoint
        logger.info("Checkpoint is a direct state dict")

    # Log original embedding size
    embedding_key = "phoneme_embedding.phoneme_embed.weight"
    if embedding_key in state_dict:
        original_shape = state_dict[embedding_key].shape
        logger.info(f"Original phoneme embedding shape: {original_shape}")
    else:
        logger.error(f"Phoneme embedding key '{embedding_key}' not found in checkpoint!")
        logger.info("Available keys (first 10):")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            logger.info(f"  {i+1}. {key}")
        raise KeyError(f"Embedding key '{embedding_key}' not found")

    # Expand phoneme embeddings
    logger.info(f"Expanding vocabulary: {old_vocab_size} → {new_vocab_size} tokens")
    logger.info(f"Initialization method: {initialization}")

    adapted_state_dict = adapt_state_dict_for_expanded_phonemes(
        state_dict=state_dict,
        old_vocab_size=old_vocab_size,
        new_vocab_size=new_vocab_size,
        embedding_key=embedding_key,
        initialization=initialization,
    )

    # Log new embedding size
    new_shape = adapted_state_dict[embedding_key].shape
    logger.info(f"New phoneme embedding shape: {new_shape}")

    # Save expanded checkpoint
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        # Preserve other checkpoint data (optimizer, scheduler, etc.)
        checkpoint["model"] = adapted_state_dict
        output_checkpoint = checkpoint
        logger.info("Preserving other checkpoint data (optimizer, scheduler, etc.)")
    else:
        output_checkpoint = adapted_state_dict

    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving expanded checkpoint to: {output_path}")
    torch.save(output_checkpoint, output_path)

    # Verify saved checkpoint
    logger.info("Verifying saved checkpoint...")
    verification = torch.load(output_path, map_location="cpu")
    if isinstance(verification, dict) and "model" in verification:
        verification_state_dict = verification["model"]
    else:
        verification_state_dict = verification

    saved_shape = verification_state_dict[embedding_key].shape
    logger.info(f"Verified embedding shape: {saved_shape}")

    if saved_shape[0] == new_vocab_size:
        logger.info("✅ Checkpoint expansion successful!")
    else:
        logger.error(f"❌ Verification failed! Expected vocab size {new_vocab_size}, got {saved_shape[0]}")

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Expand phoneme embeddings in checkpoint for multilingual support"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input checkpoint file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output checkpoint file"
    )
    parser.add_argument(
        "--old-vocab-size",
        type=int,
        default=175,
        help="Original vocabulary size (default: 175 for English)"
    )
    parser.add_argument(
        "--new-vocab-size",
        type=int,
        default=205,
        help="Target vocabulary size (default: 205 for multilingual)"
    )
    parser.add_argument(
        "--initialization",
        type=str,
        default="zeros",
        choices=["zeros", "mean", "random"],
        help="Initialization method for new embeddings (default: zeros)"
    )

    args = parser.parse_args()

    expand_checkpoint(
        input_path=args.input,
        output_path=args.output,
        old_vocab_size=args.old_vocab_size,
        new_vocab_size=args.new_vocab_size,
        initialization=args.initialization,
    )


if __name__ == "__main__":
    main()
