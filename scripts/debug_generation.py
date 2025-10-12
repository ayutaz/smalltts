"""Debug generated latents"""

import torch
from smalltts.data.phonemization.phonemes import set_language, get_token_ids
from smalltts.models.backbone.model import Backbone

# Set multilingual mode
set_language("multilingual")

# Load model
checkpoint_path = "assets/teacher_checkpoints_ja/checkpoint_final.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

model = Backbone(latent_dim=64)
model.load_state_dict(state_dict)
model.eval()

# Test text
text = "こんにちは、世界！"
phoneme_tokens = get_token_ids(text)

print(f"Text: {text}")
print(f"Phonemes: {phoneme_tokens}")
print(f"Number of phonemes: {len(phoneme_tokens)}")

# Check phoneme IDs
print(f"\nPhoneme ID range: {min(phoneme_tokens)} to {max(phoneme_tokens)}")
print(f"Expected Japanese phoneme range: 175-204")

# Check if Japanese phonemes are used
japanese_phonemes = [p for p in phoneme_tokens if p >= 175]
print(f"Japanese phonemes count: {len(japanese_phonemes)} / {len(phoneme_tokens)}")

if len(japanese_phonemes) == 0:
    print("\n⚠️  WARNING: No Japanese phonemes detected!")
    print("This might explain why the model doesn't generate sound.")
else:
    print(f"\n✅ Japanese phonemes detected: {japanese_phonemes}")

# Check model's phoneme embedding
embedding_weight = model.phoneme_embedding.phoneme_embed.weight
print(f"\nPhoneme embedding shape: {embedding_weight.shape}")
print(f"Japanese phoneme embeddings stats:")
japanese_embeddings = embedding_weight[175:205]
print(f"  Mean: {japanese_embeddings.mean().item():.6f}")
print(f"  Std: {japanese_embeddings.std().item():.6f}")
print(f"  Min: {japanese_embeddings.min().item():.6f}")
print(f"  Max: {japanese_embeddings.max().item():.6f}")

# Compare with English embeddings
english_embeddings = embedding_weight[:175]
print(f"\nEnglish phoneme embeddings stats:")
print(f"  Mean: {english_embeddings.mean().item():.6f}")
print(f"  Std: {english_embeddings.std().item():.6f}")

# Test forward pass with simple input
phonemes = torch.tensor([phoneme_tokens], dtype=torch.int64)
phonemes_mask = torch.ones(1, len(phoneme_tokens), dtype=torch.bool)
latent_length = int(len(phoneme_tokens) * 2.5)
latents = torch.randn(1, latent_length, 64)
mask = torch.ones(1, latent_length, dtype=torch.bool)
cond = torch.zeros(1, latent_length, 64)
timesteps = torch.tensor([0.5])

with torch.no_grad():
    velocity = model(latents, cond, mask, phonemes, phonemes_mask, timesteps)

print(f"\nModel output (velocity) stats:")
print(f"  Mean: {velocity.mean().item():.6f}")
print(f"  Std: {velocity.std().item():.6f}")
print(f"  Min: {velocity.min().item():.6f}")
print(f"  Max: {velocity.max().item():.6f}")

if velocity.abs().max().item() < 1e-6:
    print("\n⚠️  CRITICAL: Model output is essentially zero!")
    print("The model is not generating meaningful predictions.")
else:
    print(f"\n✅ Model output has reasonable magnitude")
