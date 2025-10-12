# smalltts

<p align="left">
  <a href="https://huggingface.co/smallbraineng/smalltts"><img alt="🤗 huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97_huggingface-smallbraineng/smalltts-yellow?logo=huggingface" /></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="python" src="https://img.shields.io/badge/python-3.10-blue" /></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="license" src="https://img.shields.io/badge/license-MIT-green" /></a>
  <a href="https://huggingface.co/smallbraineng/smalltts"><img alt="weights license" src="https://img.shields.io/badge/weights-CC--BY--NA-orange" /></a>
</p>

expressive tts that can clone any voice

- open training code and onnx model
- much faster than realtime on cpu & gpu
- emotive, built for characters

## get started

### install + quick test (2 lines)

```bash
git clone https://github.com/smallbraineng/smalltts && cd smalltts
uv run python scripts/tryme.py "Hello from smallTTS!"
```

uses [uv](https://github.com/astral-sh/uv) for setup.

### interactive (realtime on cpu)

```bash
uv run python scripts/infer/interactive.py
```

### install as a package

```bash
uv pip install "git+https://github.com/smallbraineng/smalltts"
```

```python
from smalltts import SmallTTS
tts = SmallTTS()
```

### more examples

#### batch inference

```bash
uv run python scripts/infer/batch.py
```

#### voice cloning

```bash
uv run python scripts/infer/clone.py \
  --wav assets/test_audio/1.wav \
  --transcription "the reference transcription here" \
  --text "what you want it to say"
```

## using docker compose

docker compose simplifies the workflow with minimal commands. volume mounting automatically syncs output files to `./out/`.

### initial setup

```bash
# build and start (automatically builds the image)
docker-compose up -d

# download assets (first time only, ~1-2GB)
docker-compose exec smalltts uv run python -m smalltts.assets.ensure tryme codec e2e length
```

### running inference

```bash
# generate audio (output automatically saved to ./out/tryme.wav)
docker-compose exec smalltts uv run python scripts/tryme.py "Hello from smallTTS!"
```

### using gpu version

```bash
# start gpu service
docker-compose up -d smalltts-gpu

# download assets
docker-compose exec smalltts-gpu uv run python -m smalltts.assets.ensure tryme codec e2e length

# run inference
docker-compose exec smalltts-gpu uv run python scripts/tryme.py "Hello from smallTTS!"
```

### other inference scripts

```bash
# interactive mode
docker-compose exec smalltts uv run python scripts/infer/interactive.py

# voice cloning
docker-compose exec smalltts uv run python scripts/infer/clone.py \
  --wav assets/test_audio/1.wav \
  --transcription "reference audio transcription" \
  --text "text to generate"

# batch inference
docker-compose exec smalltts uv run python scripts/infer/batch.py
```

### cleanup

```bash
# stop and remove containers
docker-compose down
```

**note for windows**: if volume mounting doesn't work correctly, use `docker cp` to extract files:
```bash
docker cp smalltts:/app/out/tryme.wav ./out/tryme.wav
```

## benchmarks

| hardware | batch | RTF | runtime |
| --- | ---: | ---: | --- |
| nvidia t4 | 1 | 0.02 | onnx + cuda |
| nvidia t4 | 16 | 0.004 | onnx + cuda |

## fine-tune or train your own

### papers (context)

[dmd2](https://arxiv.org/abs/2405.14867) · [dmdspeech](https://arxiv.org/abs/2410.11097) · [f5‑tts](https://arxiv.org/abs/2410.06885)

checkpoints live on huggingface: [smallbraineng/smalltts](https://huggingface.co/smallbraineng/smalltts).

### checkpoints

| name | download |
| --- | --- |
| teacher_checkpoints | [assets/teacher_checkpoints](https://huggingface.co/smallbraineng/smalltts/tree/main/assets/teacher_checkpoints) |
| dmd_checkpoints | [assets/dmd_checkpoints](https://huggingface.co/smallbraineng/smalltts/tree/main/assets/dmd_checkpoints) |
| asr_checkpoints | [assets/asr_checkpoints](https://huggingface.co/smallbraineng/smalltts/tree/main/assets/asr_checkpoints) |
| sv_checkpoints | [assets/sv_checkpoints](https://huggingface.co/smallbraineng/smalltts/tree/main/assets/sv_checkpoints) |

from-scratch training order, in `scripts/train`:
1. teacher model (128 steps): `uv run accelerate launch scripts/train/teacher.py`
2. dmd2 distillation (4 steps): `uv run accelerate launch scripts/train/dmd2/distill.py`
3. speaker verification: `uv run accelerate launch scripts/train/dmd2/sv.py`
4. asr: `uv run accelerate launch scripts/train/dmd2/asr.py`

### method
- teacher is a diffusion tts model generating audio in 128 sampling steps
- we train on encoded latents, not audio/mels. [microsoft vibevoice](https://github.com/microsoft/VibeVoice) encoder/decoder gives ~3200x compression
- we distill to 4 steps with distribution matching distillation ([dmd2 paper](https://arxiv.org/abs/2405.14867))
- inspired by [dmdspeech](https://arxiv.org/abs/2410.11097) and [dmospeech2](https://arxiv.org/abs/2507.14988)
- during distillation we train an asr and a speaker verification model to keep phonemes and style aligned

### data
- the default dataloader is dummy. bring your own
- we have tested [webdataset](https://github.com/webdataset/webdataset)
- checkpoints on huggingface make fine‑tuning easy

### japanese fine-tuning

smalltts supports **multilingual phonemes** (205 tokens: english ~175 + japanese ~64) with automatic language detection.

**quick start**:
```bash
# 1. expand checkpoint for multilingual support (175→205 tokens)
python scripts/utils/expand_checkpoint.py \
  --input assets/teacher_checkpoints/checkpoint_latest.pt \
  --output assets/teacher_checkpoints/checkpoint_multilingual.pt \
  --old-vocab-size 175 \
  --new-vocab-size 205

# 2. fine-tune on JVS dataset (configure paths in script)
uv run accelerate launch scripts/train/teacher_japanese.py
```

**features**:
- automatic phoneme embedding expansion
- JVS (Japanese Versatile Speech corpus) dataloader included
- fine-tuning optimized parameters (lr: 1e-5, 50k steps)
- preserves english knowledge while learning japanese

**details**:
- Full fine-tuning guide: `docs/training_japanese.md`
- Training from scratch with all 100 JVS speakers: `docs/jvs_training_guide_ja.md` (日本語)

**usage example**:
```python
from smalltts import SmallTTS
from smalltts.data.phonemization.phonemes import set_language

# multilingual mode (auto language detection)
set_language("multilingual")

tts = SmallTTS()
# japanese text - automatically detected
# english text - automatically detected
```

## inference

use onnx exports for production inference, download from huggingface, see `infer/` for examples

| model | download |
| --- | --- |
| end‑to‑end model | [assets/e2e](https://huggingface.co/smallbraineng/smalltts/tree/main/assets/e2e) |
| length predictor | [assets/length](https://huggingface.co/smallbraineng/smalltts/tree/main/assets/length) |
| codec (encoder/decoder) | [assets/codec](https://huggingface.co/smallbraineng/smalltts/tree/main/assets/codec) |

see `src/smalltts/infer/onnx.py` for the simplest api, or just use the scripts above.

## roadmap

- [x] **09.09.25** 🚀 v0 released, model and ONNX exports on 🤗 Hugging Face
- [ ] 📱 quantized version & mobile example
- [ ] 🏋️‍♂️ larger training run

## licenses

code is licensed under CC-BY-NC. see [LICENSE](LICENSE).

model weights on huggingface are licensed CC‑BY‑NA.

## next steps

1. quantized builds and realtime mobile demo
2. full version with stability upgrades

## citations

thanks to the authors and communities of dmdspeech, nanospeech, f5‑tts, and vibevoice — this repo is heavily inspired by their ideas and codebases.

```bibtex
@article{peng2025vibevoice,
    title         = {VibeVoice Technical Report},
    author        = {Zhiliang Peng and Jianwei Yu and Wenhui Wang and Yaoyao Chang and Yutao Sun and Li Dong and Yi Zhu and Weijiang Xu and Hangbo Bao and Zehua Wang and Shaohan Huang and Yan Xia and Furu Wei},
    year          = {2025},
    eprint        = {2508.19205},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL},
    doi           = {10.48550/arXiv.2508.19205},
    url           = {https://arxiv.org/abs/2508.19205}
}

@article{li2024dmdspeech,
    title         = {DMDSpeech: Distilled Diffusion Model Surpassing the Teacher in Zero-shot Speech Synthesis via Direct Metric Optimization},
    author        = {Yinghao Aaron Li and Rithesh Kumar and Zeyu Jin},
    year          = {2024},
    eprint        = {2410.11097},
    archivePrefix = {arXiv},
    primaryClass  = {eess.AS},
    doi           = {10.48550/arXiv.2410.11097},
    url           = {https://arxiv.org/abs/2410.11097}
}

@article{yin2024dmd2,
    title         = {Improved Distribution Matching Distillation for Fast Image Synthesis},
    author        = {Tianwei Yin and Micha{\"e}l Gharbi and Taesung Park and Richard Zhang and Eli Shechtman and Fredo Durand and William T. Freeman},
    year          = {2024},
    eprint        = {2405.14867},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
    doi           = {10.48550/arXiv.2405.14867},
    url           = {https://arxiv.org/abs/2405.14867}
}

@article{li2025dmospeech2,
    title         = {DMOSpeech 2: Reinforcement Learning for Duration Prediction in Metric-Optimized Speech Synthesis},
    author        = {Yinghao Aaron Li and Xilin Jiang and Fei Tao and Cheng Niu and Kaifeng Xu and Juntong Song and Nima Mesgarani},
    year          = {2025},
    eprint        = {2507.14988},
    archivePrefix = {arXiv},
    primaryClass  = {eess.AS},
    doi           = {10.48550/arXiv.2507.14988},
    url           = {https://arxiv.org/abs/2507.14988}
}

@article{chen2024f5tts,
    title         = {F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching},
    author        = {Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
    year          = {2024},
    eprint        = {2410.06885},
    archivePrefix = {arXiv},
    primaryClass  = {eess.AS},
    doi           = {10.48550/arXiv.2410.06885},
    url           = {https://arxiv.org/abs/2410.06885}
}

@misc{newman2025nanospeech,
    title        = {Nanospeech: A simple, hackable text-to-speech system in PyTorch and MLX},
    author       = {Lucas Newman},
    year         = {2025},
    note         = {GitHub repository},
    howpublished = {\url{https://github.com/lucasnewman/nanospeech}}
}

@article{eskimez2024e2tts,
    title         = {E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS},
    author        = {Sefik Emre Eskimez and Xiaofei Wang and Manthan Thakker and Canrun Li and Chung-Hsien Tsai and Zhen Xiao and Hemin Yang and Zirun Zhu and Min Tang and Xu Tan and Yanqing Liu and Sheng Zhao and Naoyuki Kanda},
    year          = {2024},
    eprint        = {2406.18009},
    archivePrefix = {arXiv},
    primaryClass  = {eess.AS},
    doi           = {10.48550/arXiv.2406.18009},
    url           = {https://arxiv.org/abs/2406.18009}
}

@article{zhu2025zipvoice,
    title         = {ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching},
    author        = {Han Zhu and Wei Kang and Zengwei Yao and Liyong Guo and Fangjun Kuang and Zhaoqing Li and Weiji Zhuang and Long Lin and Daniel Povey},
    year          = {2025},
    eprint        = {2506.13053},
    archivePrefix = {arXiv},
    primaryClass  = {eess.AS},
    doi           = {10.48550/arXiv.2506.13053},
    url           = {https://arxiv.org/abs/2506.13053}
}
```
