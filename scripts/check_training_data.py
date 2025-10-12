"""Check if training data can be reconstructed properly"""

import torch
import torchaudio

from smalltts.data.japanese import get_jvs_dataloader
from smalltts.data.phonemization.phonemes import set_language
from smalltts.codec.onnx import Encoder, Decoder

# Set multilingual
set_language("multilingual")

# Load one sample from training data
encoder = Encoder()
decoder = Decoder()

loader = get_jvs_dataloader(
    audio_dir="data/jvs_ver1/jvs001/parallel100/wav24kHz16bit",
    transcript_file="data/jvs_ver1/jvs001/parallel100/transcripts_utf8.txt",
    codec_encoder=encoder,
    batch_size=1,
    num_workers=0,
    shuffle=False,
)

# Get first sample
batch = next(iter(loader))
print(f"Text: {batch['texts'][0]}")
print(f"Latents shape: {batch['latents'].shape}")

# Decode latents
latents = batch['latents']  # [1, T, 64]
print(f"\nLatent statistics:")
print(f"  Min: {latents.min().item():.6f}")
print(f"  Max: {latents.max().item():.6f}")
print(f"  Mean: {latents.mean().item():.6f}")
print(f"  Std: {latents.std().item():.6f}")

# Decode
with torch.no_grad():
    audio = decoder.decode(latents.cpu())
    if isinstance(audio, torch.Tensor):
        audio = audio
    else:
        audio = torch.from_numpy(audio)

print(f"\nReconstructed audio shape: {audio.shape}")
print(f"Audio statistics:")
print(f"  Min: {audio.min().item():.6f}")
print(f"  Max: {audio.max().item():.6f}")
print(f"  Mean: {audio.mean().item():.6f}")
print(f"  Abs mean: {audio.abs().mean().item():.6f}")

# Save reconstructed audio
output_path = "out/reconstruction_test.wav"
audio_squeezed = audio.squeeze(0) if audio.dim() > 2 else audio
torchaudio.save(output_path, audio_squeezed, 24000)
print(f"\n✅ Reconstructed audio saved to: {output_path}")
print(f"   This should sound like the original training data")
