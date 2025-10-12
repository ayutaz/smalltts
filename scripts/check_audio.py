import torch
import torchaudio

# Load audio
audio_path = "out/japanese_test.wav"
waveform, sample_rate = torchaudio.load(audio_path)

print(f"Audio file: {audio_path}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Shape: {waveform.shape}")
print(f"Duration: {waveform.shape[1] / sample_rate:.2f} seconds")
print(f"\nWaveform statistics:")
print(f"  Min: {waveform.min().item():.6f}")
print(f"  Max: {waveform.max().item():.6f}")
print(f"  Mean: {waveform.mean().item():.6f}")
print(f"  Std: {waveform.std().item():.6f}")
print(f"  Abs mean: {waveform.abs().mean().item():.6f}")

# Check if silent
if waveform.abs().max().item() < 1e-6:
    print("\n⚠️  WARNING: Audio is essentially silent (max amplitude < 1e-6)")
elif waveform.abs().max().item() < 0.01:
    print("\n⚠️  WARNING: Audio is very quiet (max amplitude < 0.01)")
else:
    print(f"\n✅ Audio has reasonable amplitude (max: {waveform.abs().max().item():.4f})")
