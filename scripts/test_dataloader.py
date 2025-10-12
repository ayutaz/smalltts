"""Test JVS dataloader"""
import torch
from smalltts.data.japanese import get_jvs_dataloader
from smalltts.data.phonemization.phonemes import set_language
from smalltts.codec.onnx import Encoder

# Set language
set_language("ja")

# Configuration
JVS_ROOT_DIR = "data/jvs_ver1"
SPEAKER_IDS = None  # All speakers
SUBSET = "parallel100"
BATCH_SIZE = 1
NUM_WORKERS = 0

print("=" * 80)
print("JVS DATALOADER TEST")
print("=" * 80)

# Initialize encoder
print("\nInitializing encoder...")
encoder = Encoder()
print("[OK] Encoder initialized")

# Initialize dataloader
print(f"\nLoading JVS dataset from: {JVS_ROOT_DIR}")
train_loader = get_jvs_dataloader(
    root_dir=JVS_ROOT_DIR,
    speaker_ids=SPEAKER_IDS,
    subset=SUBSET,
    codec_encoder=encoder,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,  # No shuffle for testing
)

print(f"[OK] Dataloader created")
print(f"Dataset size: {len(train_loader.dataset)}")

# Test first batch
print("\n" + "=" * 80)
print("TESTING FIRST BATCH")
print("=" * 80)

batch = next(iter(train_loader))

print(f"\nBatch keys: {batch.keys()}")
print(f"Phonemes shape: {batch['phonemes'].shape}")
print(f"Phonemes sample: {batch['phonemes'][0][:20].tolist()}")
print(f"Phonemes lengths: {batch['phonemes_lengths'].tolist()}")
print(f"Latents shape: {batch['latents'].shape}")
print(f"Latents stats: mean={batch['latents'].mean().item():.4f}, std={batch['latents'].std().item():.4f}, min={batch['latents'].min().item():.4f}, max={batch['latents'].max().item():.4f}")
print(f"Latents lengths: {batch['latents_lengths'].tolist()}")

# Test a few more batches
print("\n" + "=" * 80)
print("TESTING 5 MORE BATCHES")
print("=" * 80)

for i, batch in enumerate(train_loader):
    if i >= 5:
        break
    print(f"\nBatch {i+1}:")
    print(f"  Phonemes: {batch['phonemes'].shape}, length={batch['phonemes_lengths'].item()}")
    print(f"  Latents: {batch['latents'].shape}, length={batch['latents_lengths'].item()}")
    print(f"  Latents mean={batch['latents'].mean().item():.4f}, std={batch['latents'].std().item():.4f}")

print("\n" + "=" * 80)
print("DATALOADER TEST COMPLETE")
print("=" * 80)
