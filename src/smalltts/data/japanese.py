"""Japanese dataset loader for JVS corpus

Supports JVS (Japanese versatile speech corpus) format.

Expected directory structure:

JVS:
    dataset_root/
        jvs001/
            parallel100/
                wav24kHz16bit/
                    VOICEACTRESS100_001.wav
                    VOICEACTRESS100_002.wav
                    ...
                transcripts_utf8.txt
        jvs002/
            parallel100/
                wav24kHz16bit/
                    VOICEACTRESS100_001.wav
                    ...
                transcripts_utf8.txt
        ...

Transcript format:
    VOICEACTRESS100_001:それは確かにそうです。
    VOICEACTRESS100_002:今日はいい天気ですね。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from .phonemization.phonemes import get_token_ids

logger = logging.getLogger(__name__)


class JVSDataset(Dataset):
    """Dataset for JVS (Japanese versatile speech corpus)

    Args:
        audio_dir: Directory containing audio files (e.g., wav24kHz16bit/)
        transcript_file: Path to transcript file (format: "filename:text" per line)
        codec_encoder: Optional VibeVoice encoder for converting audio to latents
        target_sample_rate: Target sample rate (default: 24000 Hz)
        max_audio_length_sec: Maximum audio length in seconds (default: 30)
        file_extension: Audio file extension (default: ".wav")
    """

    def __init__(
        self,
        audio_dir: str,
        transcript_file: str,
        codec_encoder=None,
        target_sample_rate: int = 24000,
        max_audio_length_sec: float = 30.0,
        file_extension: str = ".wav",
    ):
        super().__init__()
        self.audio_dir = Path(audio_dir)
        self.transcript_file = Path(transcript_file)
        self.codec_encoder = codec_encoder
        self.target_sample_rate = target_sample_rate
        self.max_audio_length_sec = max_audio_length_sec
        self.file_extension = file_extension

        # Load metadata (filename -> text mapping)
        self.metadata = self._load_metadata()

        logger.info(
            f"Loaded JVS dataset: {len(self.metadata)} samples from {audio_dir}"
        )

    def _load_metadata(self) -> List[Tuple[str, str]]:
        """Load metadata from JVS transcript file

        JVS format: "filename:text" (colon-separated)
        Also supports alternative formats for compatibility:
        - "filename|text" (pipe-separated)
        - "filename\ttext" (tab-separated)
        - "filename text" (space-separated)

        Returns:
            List of (audio_filename, text) tuples
        """
        metadata = []

        if not self.transcript_file.exists():
            raise FileNotFoundError(
                f"Transcript file not found: {self.transcript_file}"
            )

        with open(self.transcript_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Try different separators (JVS uses colon by default)
                for separator in [":", "|", "\t", " "]:
                    if separator in line:
                        parts = line.split(separator, 1)
                        if len(parts) == 2:
                            filename, text = parts
                            filename = filename.strip()
                            text = text.strip()

                            # Handle filename with or without extension
                            if not filename.endswith(self.file_extension):
                                filename = filename + self.file_extension

                            audio_path = self.audio_dir / filename

                            if not audio_path.exists():
                                logger.warning(
                                    f"Audio file not found: {audio_path} (line {line_num})"
                                )
                                continue

                            metadata.append((str(audio_path), text))
                            break
                else:
                    logger.warning(
                        f"Could not parse line {line_num}: {line[:50]}..."
                    )

        if not metadata:
            raise ValueError(
                f"No valid samples found in {self.transcript_file}. "
                f"Please check file format and audio directory."
            )

        return metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        audio_path, text = self.metadata[idx]

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sample_rate
            )
            waveform = resampler(waveform)

        # Trim if too long
        max_samples = int(self.max_audio_length_sec * self.target_sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
            logger.debug(f"Trimmed audio {audio_path} to {self.max_audio_length_sec}s")

        # Remove channel dimension: [1, T] -> [T]
        waveform = waveform.squeeze(0)

        # Convert to latents if encoder is provided
        if self.codec_encoder is not None:
            with torch.no_grad():
                # Add batch dimension: [T] -> [1, T]
                waveform_batch = waveform.unsqueeze(0)
                latents = self.codec_encoder(waveform_batch)  # [1, T', 64]
                latents = latents.squeeze(0)  # [T', 64]
        else:
            # Return raw waveform if no encoder provided
            latents = waveform.unsqueeze(-1)  # [T, 1] for compatibility

        # Get phoneme token IDs (auto-detects Japanese)
        phoneme_tokens = torch.tensor(get_token_ids(text), dtype=torch.int64)

        return {
            "text": text,
            "phonemes": phoneme_tokens,
            "latents": latents,
            "audio_path": audio_path,
        }


def jvs_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for JVS dataset

    Args:
        batch: List of samples from JVSDataset

    Returns:
        Dictionary with batched and padded tensors
    """
    texts = [item["text"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]

    # Pad phoneme sequences
    phonemes = [item["phonemes"] for item in batch]
    phonemes_lengths = torch.tensor([len(p) for p in phonemes], dtype=torch.int64)
    phonemes_padded = pad_sequence(phonemes, batch_first=True, padding_value=0)

    # Pad latent sequences
    latents = [item["latents"] for item in batch]
    latents_lengths = torch.tensor([len(lat) for lat in latents], dtype=torch.int64)
    latents_padded = pad_sequence(latents, batch_first=True, padding_value=0.0)

    return {
        "texts": texts,
        "phonemes": phonemes_padded,
        "phonemes_lengths": phonemes_lengths,
        "latents": latents_padded,
        "latents_lengths": latents_lengths,
        "audio_paths": audio_paths,
    }


def get_jvs_dataloader(
    audio_dir: str,
    transcript_file: str,
    codec_encoder=None,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    target_sample_rate: int = 24000,
    max_audio_length_sec: float = 30.0,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for JVS (Japanese versatile speech corpus)

    Args:
        audio_dir: Directory containing audio files (e.g., wav24kHz16bit/)
        transcript_file: Path to transcript file (transcripts_utf8.txt)
        codec_encoder: Optional VibeVoice encoder for converting audio to latents
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the dataset
        target_sample_rate: Target sample rate (default: 24000 Hz)
        max_audio_length_sec: Maximum audio length in seconds
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        DataLoader instance

    Example:
        >>> # Load JVS dataset (single speaker)
        >>> loader = get_jvs_dataloader(
        ...     audio_dir="data/jvs_ver1/jvs001/parallel100/wav24kHz16bit",
        ...     transcript_file="data/jvs_ver1/jvs001/parallel100/transcripts_utf8.txt",
        ...     batch_size=16,
        ...     num_workers=4
        ... )
        >>>
        >>> # Load with codec encoder for latent conversion
        >>> from smalltts.codec import VibeVoiceEncoder
        >>> encoder = VibeVoiceEncoder()
        >>> loader = get_jvs_dataloader(
        ...     audio_dir="data/jvs_ver1/jvs001/parallel100/wav24kHz16bit",
        ...     transcript_file="data/jvs_ver1/jvs001/parallel100/transcripts_utf8.txt",
        ...     codec_encoder=encoder,
        ...     batch_size=16
        ... )
    """
    dataset = JVSDataset(
        audio_dir=audio_dir,
        transcript_file=transcript_file,
        codec_encoder=codec_encoder,
        target_sample_rate=target_sample_rate,
        max_audio_length_sec=max_audio_length_sec,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=jvs_collate_fn,
        pin_memory=True,
        **kwargs,
    )

    return dataloader


if __name__ == "__main__":
    # Example usage / testing
    print("=" * 80)
    print("JVS DATASET LOADER TEST")
    print("=" * 80)

    # This is a minimal test that demonstrates the API
    # For actual testing, you would need a real JVS dataset

    print("\nExpected JVS dataset structure:")
    print("  jvs_ver1/")
    print("    jvs001/")
    print("      parallel100/")
    print("        wav24kHz16bit/")
    print("          VOICEACTRESS100_001.wav")
    print("          VOICEACTRESS100_002.wav")
    print("          ...")
    print("        transcripts_utf8.txt")
    print("    jvs002/")
    print("      parallel100/")
    print("        ...")

    print("\nJVS transcript format:")
    print("  VOICEACTRESS100_001:それは確かにそうです。")
    print("  VOICEACTRESS100_002:今日はいい天気ですね。")
    print("  ...")

    print("\nSupported alternative formats:")
    print("  - filename:text (colon-separated, JVS default)")
    print("  - filename|text (pipe-separated)")
    print("  - filename\\ttext (tab-separated)")
    print("  - filename text (space-separated)")

    print("\n" + "=" * 80)
    print("Example usage:")
    print("=" * 80)

    print("""
# Load JVS dataset (single speaker)
loader = get_jvs_dataloader(
    audio_dir="data/jvs_ver1/jvs001/parallel100/wav24kHz16bit",
    transcript_file="data/jvs_ver1/jvs001/parallel100/transcripts_utf8.txt",
    batch_size=16,
    num_workers=4
)

for batch in loader:
    texts = batch["texts"]  # List of strings
    phonemes = batch["phonemes"]  # [batch, max_phoneme_len]
    latents = batch["latents"]  # [batch, max_latent_len, latent_dim]
    break

# Load with codec encoder for latent conversion
from smalltts.codec import VibeVoiceEncoder
encoder = VibeVoiceEncoder()
loader = get_jvs_dataloader(
    audio_dir="data/jvs_ver1/jvs001/parallel100/wav24kHz16bit",
    transcript_file="data/jvs_ver1/jvs001/parallel100/transcripts_utf8.txt",
    codec_encoder=encoder,
    batch_size=16
)
""")

    print("=" * 80)
    print("[INFO] JVS dataset loader ready for use")
    print("=" * 80)
