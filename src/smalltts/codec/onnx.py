# ONNX exports from Microsoft's VibeVoice (MIT) (https://github.com/microsoft/VibeVoice)

from typing import Iterable, Optional

import numpy as np
import onnxruntime as ort
import torch


def _default_providers() -> list[str]:
    av = set(ort.get_available_providers())
    providers = []

    if "CUDAExecutionProvider" in av:
        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


class _ONNXRunner:
    def __init__(self, path: str, providers: Optional[Iterable[str]] = None) -> None:
        prov = list(providers) if providers is not None else _default_providers()

        # Optimize ONNX session for faster inference
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Limit threads for better cache utilization
        sess_options.inter_op_num_threads = 1  # Single inter-op thread

        self.sess = ort.InferenceSession(path, sess_options=sess_options, providers=prov)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
        out = self.sess.run([self.out_name], {self.in_name: arr})[0]
        return torch.from_numpy(out)


class Decoder(_ONNXRunner):
    def __init__(
        self,
        path: str = "assets/codec/decoder.onnx",
        providers: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(path, providers)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents (torch.Tensor): encoded latent representations
                - dtype: float32
                - shape: (batch, T, 64)
        returns:
            torch.Tensor: audio
                - dtype: float32
                - shape: (batch, 1, T)
        """
        return self._run(latents)


class Encoder(_ONNXRunner):
    def __init__(
        self,
        path: str = "assets/codec/encoder.onnx",
        providers: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(path, providers)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        args:
            audio (torch.Tensor): 1 channel, 24kHz audio
                - dtype: float32
                - shape: (batch, 1, time)
        returns:
            torch.Tensor: latents
                - dtype: float32
                - shape: (batch, T, 64)
        """
        return self._run(audio)


if __name__ == "__main__":
    dummy_audio = torch.randn(1, 1, 24_000, dtype=torch.float32)
    encoder = Encoder()
    decoder = Decoder()
    latents = encoder.encode(dummy_audio)
    print("latents shape:", tuple(latents.shape))
    recon_audio = decoder.decode(latents)
    print("reconstructed audio shape:", tuple(recon_audio.shape))
