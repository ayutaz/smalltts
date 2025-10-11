"""Model utility functions for checkpoint compatibility and conversions"""

import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


def expand_phoneme_embedding(
    old_embedding: nn.Embedding,
    new_vocab_size: int,
    initialization: str = "zeros"
) -> nn.Embedding:
    """Expand phoneme embedding layer to support larger vocabulary

    This is useful when migrating from a smaller vocabulary (e.g., English-only with ~175 tokens)
    to a larger vocabulary (e.g., multilingual with ~205 tokens).

    Args:
        old_embedding: Existing embedding layer with smaller vocabulary
        new_vocab_size: Target vocabulary size
        initialization: How to initialize new embeddings ("zeros", "mean", or "random")
            - "zeros": Initialize new embeddings to zero (default)
            - "mean": Initialize new embeddings to mean of existing embeddings
            - "random": Initialize new embeddings randomly (same distribution as existing)

    Returns:
        New embedding layer with expanded vocabulary

    Example:
        >>> old_emb = nn.Embedding(175, 512)  # English-only model
        >>> new_emb = expand_phoneme_embedding(old_emb, 205)  # Multilingual model
        >>> assert new_emb.num_embeddings == 205
        >>> assert new_emb.embedding_dim == 512
    """
    old_vocab_size, embedding_dim = old_embedding.weight.shape

    if new_vocab_size <= old_vocab_size:
        logger.warning(
            f"New vocabulary size ({new_vocab_size}) is not larger than "
            f"old vocabulary size ({old_vocab_size}). Returning original embedding."
        )
        return old_embedding

    # Create new embedding layer with expanded vocabulary
    new_embedding = nn.Embedding(new_vocab_size, embedding_dim)

    # Copy existing weights
    with torch.no_grad():
        new_embedding.weight[:old_vocab_size] = old_embedding.weight

        # Initialize new embeddings
        if initialization == "zeros":
            new_embedding.weight[old_vocab_size:] = 0
        elif initialization == "mean":
            mean_embedding = old_embedding.weight.mean(dim=0)
            new_embedding.weight[old_vocab_size:] = mean_embedding
        elif initialization == "random":
            # Match the distribution of existing embeddings
            std = old_embedding.weight.std()
            mean = old_embedding.weight.mean()
            new_embedding.weight[old_vocab_size:].normal_(mean.item(), std.item())
        else:
            raise ValueError(
                f"Unknown initialization method: {initialization}. "
                "Use 'zeros', 'mean', or 'random'."
            )

    logger.info(
        f"Expanded phoneme embedding from {old_vocab_size} to {new_vocab_size} tokens "
        f"(initialization: {initialization})"
    )

    return new_embedding


def adapt_state_dict_for_expanded_phonemes(
    state_dict: dict,
    old_vocab_size: int,
    new_vocab_size: int,
    embedding_key: str = "phoneme_embedding.phoneme_embed.weight",
    initialization: str = "zeros"
) -> dict:
    """Adapt state dict to support expanded phoneme vocabulary

    Modifies the phoneme embedding weights in a state dict to support a larger vocabulary.
    This allows loading checkpoints trained with smaller vocabulary into models with
    larger vocabulary.

    Args:
        state_dict: Model state dictionary
        old_vocab_size: Original vocabulary size in the checkpoint
        new_vocab_size: Target vocabulary size for the model
        embedding_key: Key for phoneme embedding weights in state dict
        initialization: How to initialize new embeddings ("zeros", "mean", or "random")

    Returns:
        Modified state dictionary with expanded phoneme embeddings

    Example:
        >>> # Load checkpoint trained with English-only (175 tokens)
        >>> checkpoint = torch.load("english_model.pt")
        >>> state_dict = checkpoint["model"]
        >>>
        >>> # Adapt for multilingual model (205 tokens)
        >>> adapted_state_dict = adapt_state_dict_for_expanded_phonemes(
        ...     state_dict, old_vocab_size=175, new_vocab_size=205
        ... )
        >>>
        >>> # Load into multilingual model
        >>> model.load_state_dict(adapted_state_dict, strict=False)
    """
    if embedding_key not in state_dict:
        logger.warning(
            f"Embedding key '{embedding_key}' not found in state dict. "
            "Available keys: " + ", ".join(list(state_dict.keys())[:5]) + "..."
        )
        return state_dict

    old_embedding_weight = state_dict[embedding_key]
    current_vocab_size, embedding_dim = old_embedding_weight.shape

    if current_vocab_size != old_vocab_size:
        logger.warning(
            f"Expected vocabulary size {old_vocab_size}, but found {current_vocab_size} "
            f"in state dict for key '{embedding_key}'"
        )

    if new_vocab_size <= current_vocab_size:
        logger.info(
            f"No expansion needed: current vocab size ({current_vocab_size}) >= "
            f"target vocab size ({new_vocab_size})"
        )
        return state_dict

    # Create expanded embedding weights
    new_embedding_weight = torch.zeros(new_vocab_size, embedding_dim)

    # Copy existing weights
    new_embedding_weight[:current_vocab_size] = old_embedding_weight

    # Initialize new embeddings
    if initialization == "zeros":
        pass  # Already initialized to zero
    elif initialization == "mean":
        mean_embedding = old_embedding_weight.mean(dim=0)
        new_embedding_weight[current_vocab_size:] = mean_embedding
    elif initialization == "random":
        std = old_embedding_weight.std()
        mean = old_embedding_weight.mean()
        new_embedding_weight[current_vocab_size:].normal_(mean.item(), std.item())
    else:
        raise ValueError(
            f"Unknown initialization method: {initialization}. "
            "Use 'zeros', 'mean', or 'random'."
        )

    # Update state dict
    state_dict[embedding_key] = new_embedding_weight

    logger.info(
        f"Adapted state dict: expanded phoneme embedding from {current_vocab_size} "
        f"to {new_vocab_size} tokens (initialization: {initialization})"
    )

    return state_dict


if __name__ == "__main__":
    print("=" * 80)
    print("PHONEME EMBEDDING EXPANSION TEST")
    print("=" * 80)

    # Test 1: Expand embedding layer
    print("\n[Test 1] Expand embedding layer from 175 to 205 tokens")
    old_emb = nn.Embedding(175, 512)
    nn.init.normal_(old_emb.weight, mean=0.0, std=0.02)

    new_emb = expand_phoneme_embedding(old_emb, 205, initialization="zeros")
    print(f"Old embedding shape: {old_emb.weight.shape}")
    print(f"New embedding shape: {new_emb.weight.shape}")
    print(f"First token preserved: {torch.allclose(old_emb.weight[0], new_emb.weight[0])}")
    print(f"New tokens initialized to zero: {torch.allclose(new_emb.weight[175:], torch.zeros(30, 512))}")

    # Test 2: Adapt state dict
    print("\n[Test 2] Adapt state dict for expanded vocabulary")
    state_dict = {
        "phoneme_embedding.phoneme_embed.weight": old_emb.weight,
        "other_param": torch.randn(10, 10)
    }

    adapted_state_dict = adapt_state_dict_for_expanded_phonemes(
        state_dict.copy(), old_vocab_size=175, new_vocab_size=205
    )

    print(f"Original embedding shape: {state_dict['phoneme_embedding.phoneme_embed.weight'].shape}")
    print(f"Adapted embedding shape: {adapted_state_dict['phoneme_embedding.phoneme_embed.weight'].shape}")
    print(f"Other params preserved: {torch.equal(state_dict['other_param'], adapted_state_dict['other_param'])}")

    # Test 3: Different initialization methods
    print("\n[Test 3] Compare initialization methods")
    for init_method in ["zeros", "mean", "random"]:
        new_emb = expand_phoneme_embedding(old_emb, 205, initialization=init_method)
        new_tokens_mean = new_emb.weight[175:].mean().item()
        new_tokens_std = new_emb.weight[175:].std().item()
        print(f"{init_method:8s}: mean={new_tokens_mean:7.4f}, std={new_tokens_std:7.4f}")

    print("\n" + "=" * 80)
    print("[SUCCESS] All tests completed!")
    print("=" * 80)
