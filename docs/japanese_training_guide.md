# SmallTTS 日本語学習ガイド

このドキュメントでは、JVS corpus（全100話者）を使用して、**日本語専用TTSモデルを一から学習**する方法を説明します。

## 概要

SmallTTSは日本語専用の音声合成システムとして、JVS corpusで一から学習できます。

**主な特徴**:
- 日本語専用音素語彙（64トークン）
- 全100話者のマルチスピーカー学習
- pyopenjtalk-plusによる日本語音素化
- Python 3.12.7 + CUDA 12.4対応
- uvパッケージマネージャー使用

**学習方式**: Training from scratch（既存モデルを使わず、ゼロから初期化）

## 環境セットアップ

### 必要な環境

**推奨スペック**:
- GPU: VRAM 16GB以上（RTX 4070 Ti SUPER 16GB で動作確認済み）
- RAM: 32GB以上
- ストレージ: 約10GB（JVSデータセット + チェックポイント + 潜在表現キャッシュ）

**対応OS**:
- Windows（Python 3.12.7 + CUDA 12.4）
- Linux（推奨）
- macOS（CPUのみ、非常に遅い）

### Pythonとパッケージのセットアップ

#### 1. uvのインストール

**Linux/macOS**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. リポジトリのクローン

```bash
git clone https://github.com/yourusername/smalltts.git
cd smalltts
```

#### 3. Python 3.12.7とCUDA版PyTorchのインストール

```bash
# Python 3.12.7をインストール
uv python install 3.12.7

# 依存関係のインストール（pyopenjtalk-plusを含む）
uv sync

# CPU版PyTorchをアンインストールしてCUDA版をインストール
uv pip uninstall torch torchvision torchaudio
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

#### 4. 環境確認

```bash
# PyTorchとCUDAの確認
uv run --no-sync python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 出力例:
# PyTorch: 2.6.0+cu124
# CUDA available: True
```

## JVSデータセットの準備

### ダウンロード

公式サイトから`jvs_ver1.zip`（約3.4GB）をダウンロード:
- [JVS Corpus公式サイト](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)

### 解凍

**Linux**:
```bash
unzip jvs_ver1.zip -d data/
```

**Windows（7-Zip推奨）**:
```bash
"C:\Program Files\7-Zip\7z.exe" x jvs_ver1.zip -o"data" -aoa
```

注意: Windowsの標準解凍は数時間～数日かかる場合があります。7-Zipなら約2分です。

### ディレクトリ構造の確認

```
data/
  jvs_ver1/
    jvs001/
      parallel100/
        wav24kHz16bit/          # 音声ファイル（100個）
          VOICEACTRESS100_001.wav
          VOICEACTRESS100_002.wav
          ...
        transcripts_utf8.txt    # 転写テキスト
    jvs002/
      parallel100/
        wav24kHz16bit/
          ...
        transcripts_utf8.txt
    ...
    jvs100/                     # 100話者まで
```

**転写ファイル形式** (`transcripts_utf8.txt`):
```
VOICEACTRESS100_001:それは確かにそうです。
VOICEACTRESS100_002:今日はいい天気ですね。
VOICEACTRESS100_003:音声合成システムを作っています。
```

**データセット確認**:
```bash
# 話者数の確認
ls -d data/jvs_ver1/jvs* | wc -l
# 出力: 100

# Windows (PowerShell)
(Get-ChildItem data\jvs_ver1\jvs*).Count
```

総サンプル数: 約9,997サンプル（100話者 × 約100発話）

## 潜在表現キャッシュの作成（重要）

学習を高速化するため、JVS音声ファイルを事前にONNXエンコーダーで潜在表現に変換してキャッシュします。

### キャッシュ作成の効果

**キャッシュなし**:
- BATCH_SIZE=1（ONNX encoderはバッチ処理不可）
- NUM_WORKERS=0（ONNX encoderはマルチプロセス不可）
- 速度: 約4秒/ステップ（非常に遅い）

**キャッシュあり**:
- BATCH_SIZE=60（RTX 4070 Ti SUPER最適化）
- NUM_WORKERS=4（マルチプロセス可能）
- 速度: 約0.6秒/ステップ（約7倍高速）

### キャッシュの作成方法

```bash
# 全100話者のキャッシュを作成（推奨）
uv run python scripts/preprocess/cache_jvs_latents.py

# 特定の話者のみキャッシュ（テスト用）
uv run python scripts/preprocess/cache_jvs_latents.py --speakers jvs001 jvs002

# バッチサイズを調整（デフォルト: 16）
uv run python scripts/preprocess/cache_jvs_latents.py --batch-size 32

# 中断したキャッシュ作成を再開
uv run python scripts/preprocess/cache_jvs_latents.py --resume
```

### キャッシュ作成時間

**実測時間** (RTX 4070 Ti SUPER 16GB):
- 全100話者（約10,000サンプル）: 約4-6時間
- バッチサイズ16（デフォルト）で長さベースのバッチング使用

### キャッシュの確認

```bash
# キャッシュディレクトリの確認
ls -la data/jvs_ver1_latents/

# Windows (PowerShell)
dir data\jvs_ver1_latents\

# キャッシュサイズ
du -sh data/jvs_ver1_latents/
# 出力例: 1.5GB
```

**キャッシュ構造**:
```
data/jvs_ver1_latents/
  metadata.json              # キャッシュメタデータ
  jvs001/
    parallel100/
      VOICEACTRESS100_001.pt  # 潜在表現テンソル
      VOICEACTRESS100_002.pt
      ...
  jvs002/
    parallel100/
      ...
  ...
  jvs100/
```

### トラブルシューティング（キャッシュ作成）

**問題: CUDA out of memory during caching**

```bash
# バッチサイズを減らす
uv run python scripts/preprocess/cache_jvs_latents.py --batch-size 8
```

**問題: 中断してしまった場合**

```bash
# --resumeオプションで既存キャッシュをスキップ
uv run python scripts/preprocess/cache_jvs_latents.py --resume
```

## 学習の実行

### 学習前の検証（推奨）

データローダーとモデルが正しく動作するか確認:

```bash
# マルチスピーカーデータローダーのテスト
uv run python scripts/test_multispeaker.py

# 総合的なセットアップ検証
uv run python scripts/verify_japanese_setup.py
```

### 訓練パラメータの確認

`scripts/train/teacher_japanese.py`のデフォルト設定:

```python
# Dataset paths - Multi-speaker mode (all 100 JVS speakers)
JVS_ROOT_DIR = "data/jvs_ver1"
SPEAKER_IDS = None  # None = use all speakers (jvs001-jvs100)
SUBSET = "parallel100"

# Training parameters
BATCH_SIZE = 60  # Optimized for RTX 4070 Ti SUPER (~14-15GB VRAM, with latent cache)
NUM_WORKERS = 4  # Can use multiple workers when using cached latents
JVS_CACHE_DIR = "data/jvs_ver1_latents"  # Pre-cached latents directory
NUM_STEPS = 10_000  # Training steps (10k for testing, 100k recommended for production)
NUM_SAVE_STEPS = 1_000  # Checkpoint save interval

# Checkpoint paths
LOAD_FROM_CHECKPOINT = None  # Train from scratch
OUTPUT_DIR = "assets/teacher_checkpoints_ja"

# Learning rate
LEARNING_RATE = 1e-4  # Standard training rate
WARMUP_STEPS = 1_000  # Warmup steps
WEIGHT_DECAY = 1e-2
```

**パラメータ説明**:
- `NUM_STEPS`: 総学習ステップ数（10,000ステップは約16時間、100,000ステップは約147時間 = 6.1日）
- `BATCH_SIZE`: 60（潜在表現キャッシュ使用時、RTX 4070 Ti SUPERで最適）
- `JVS_CACHE_DIR`: 潜在表現キャッシュのディレクトリ（`scripts/preprocess/cache_jvs_latents.py`で作成）
- `NUM_WORKERS`: 4（キャッシュ使用時は0より大きい値が可能、ONNX encoder使用時は0必須）
- `SPEAKER_IDS`: `None`で全100話者を使用
- `LOAD_FROM_CHECKPOINT`: `None`で一から学習
- `LEARNING_RATE`: 1e-4（一から学習用の標準値）

### 学習の開始

**Single GPU**:
```bash
uv run --no-sync accelerate launch scripts/train/teacher_japanese.py
```

**Multi-GPU**:
```bash
uv run --no-sync accelerate launch --multi-gpu scripts/train/teacher_japanese.py
```

重要: `--no-sync`オプションは必須です。これにより、実行時にCPU版PyTorchが再インストールされるのを防ぎます。

### 訓練の監視

**コンソール出力**:
```
================================================================================
JAPANESE TEACHER MODEL TRAINING (FROM SCRATCH)
================================================================================

[1/6] Setting up Japanese phoneme vocabulary
Phoneme vocabulary size: 64

[2/6] Loading VibeVoice encoder

[3/6] Loading JVS dataset (multi-speaker)
Loaded 9997 samples from 100 speakers in data/jvs_ver1

[4/6] Setting up distributed training

[5/6] Initializing model

[6/6] Skipping checkpoint loading (training from scratch)

================================================================================
TRAINING CONFIGURATION
================================================================================
Batch size: 1
Learning rate: 0.0001
Total steps: 10,000
Warmup steps: 1,000
Save interval: 1,000 steps
Output directory: assets/teacher_checkpoints_ja
================================================================================

Training: 100%|████████████| 10000/10000 [2:47:32<00:00, loss=0.234, lr=8.95e-05]
```

**実測訓練時間** (RTX 4070 Ti SUPER 16GB、BATCH_SIZE=60、潜在表現キャッシュ使用):
- 10,000ステップ: 約16時間（2025-10-13実測）
- 100,000ステップ: 約147時間（6.1日、実測データから推定）
- 平均速度: 1.47時間/1,000ステップ

**注意**: より高性能なGPU（A100、H100など）では大幅に高速化されます。ドキュメントの「約28時間」は高性能GPU前提です。

**重要な指標**:
- `loss`: 速度予測のMSE損失（初期: 0.5-1.0、収束: 0.1-0.3が良好）
- `lr`: 現在の学習率（ウォームアップとコサインアニーリング）

### チェックポイントの確認

**保存場所**: `assets/teacher_checkpoints_ja/`

**保存されるファイル**:
```
assets/teacher_checkpoints_ja/
  checkpoint_step_1000/           # 訓練再開用（optimizer含む）
  checkpoint_step_1000.pt         # モデル重みのみ（推論用）
  checkpoint_step_2000/
  checkpoint_step_2000.pt
  ...
  checkpoint_latest.pt            # 最新チェックポイント
  checkpoint_final.pt             # 最終チェックポイント
```

### 訓練の中断と再開

**中断**: `Ctrl+C`で安全に中断できます（次のチェックポイント保存時に停止）

**再開**:
```python
# scripts/train/teacher_japanese.py を編集
LOAD_FROM_CHECKPOINT = "assets/teacher_checkpoints_ja/checkpoint_step_5000"
```

再開時は、optimizer状態とscheduler状態も復元されるため、学習を継続できます。

## 推論とテスト

### 学習済みモデルでの推論

```bash
# 日本語教師モデルの推論スクリプト
uv run python scripts/infer/test_teacher_japanese.py \
  --checkpoint assets/teacher_checkpoints_ja/checkpoint_final.pt \
  --reference data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav \
  --transcription "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。" \
  --text "こんにちは、世界。これはテストです。" \
  --output "out/output.wav" \
  --steps 128 \
  --cfg-scale 2.0
```

**パラメータ**:
- `--checkpoint`: 学習済みチェックポイント（.ptファイル）
- `--reference`: 参照音声ファイル（声質のクローニング）
- `--transcription`: 参照音声の転写テキスト
- `--text`: 生成したいテキスト
- `--steps`: DDIMサンプリングステップ数（128推奨、256でより高品質）
- `--cfg-scale`: Classifier-Free Guidanceスケール（1.0〜3.0、デフォルト2.0推奨）

### Pythonコードでの使用

```python
import torch
from smalltts.models.backbone.model import Backbone
from smalltts.data.phonemization.phonemes import set_language

# 日本語モードに設定
set_language("ja")

# モデルの初期化
model = Backbone(latent_dim=64)

# チェックポイントのロード
checkpoint = torch.load("assets/teacher_checkpoints_ja/checkpoint_final.pt")
model.load_state_dict(checkpoint["model"])
model.eval()

print(f"Loaded checkpoint from step {checkpoint['step']}")
print(f"Final loss: {checkpoint.get('loss', 'N/A')}")
```

### 音質について

**10,000ステップでの音質**:
- 日本語の発音は認識可能
- 音質は「ガビガビ」で実用的ではない
- 全区間で音声生成は可能（DDIMサンプリング修正により解決）

**推奨学習量**:
- 実用的な音質: 50,000〜100,000ステップ
- 参考: 英語モデルは600,000ステップで学習されている

## トラブルシューティング

### 問題1: CUDA out of memory

**エラー**: `RuntimeError: CUDA out of memory`

**解決策**:
```python
# BATCH_SIZEを減らす（すでに1なら無理）
BATCH_SIZE = 1

# max_audio_length_secを減らす
max_audio_length_sec = 20.0  # デフォルト30から減らす
```

### 問題2: pyopenjtalk not found

**エラー**: `pyopenjtalk-plus not installed`

**解決策**:
```bash
uv pip install pyopenjtalk-plus
```

### 問題3: NUM_WORKERS > 0 でCUDAエラー

**エラー**: `CUDA failure 3: initialization error`

**原因**: ONNX encoderをオンザフライで使用する場合、マルチプロセスでCUDAコンテキストを共有できません

**解決策1: キャッシュを使用（推奨）**:
```bash
# 潜在表現キャッシュを作成
uv run python scripts/preprocess/cache_jvs_latents.py

# キャッシュ使用時はNUM_WORKERS > 0が可能
NUM_WORKERS = 4
```

**解決策2: キャッシュなしの場合**:
```python
NUM_WORKERS = 0  # ONNX encoder使用時は必ず0に設定
BATCH_SIZE = 1   # ONNX encoderはバッチ処理不可
```

### 問題4: 学習率スケジューラーエラー

**エラー**: `ZeroDivisionError: float division by zero`

**原因**: `NUM_STEPS`が`WARMUP_STEPS`以下の場合に発生

**解決策**:
```python
# WARMUP_STEPSをNUM_STEPSより小さくする
NUM_STEPS = 10_000
WARMUP_STEPS = 1_000  # NUM_STEPSの10%程度
```

### 問題5: GPU memory issues

**症状**: 学習開始前からVRAMが大量に使用されている

**解決策**:
```bash
# 実行中のプロセスを確認（Windows）
nvidia-smi

# 不要なプロセスを停止
```

## FAQ

### Q1: 学習時間を短縮できますか？

**現実的な方法**:
- より高速なGPU（A100、H100など）
- Multi-GPU訓練
- BATCH_SIZEを増やす（VRAMが許せば）

### Q2: 1話者だけで学習できますか？

可能ですが、推奨しません。理由:
- データ量不足（100発話は少ない）
- 過学習しやすい
- 汎化性能が低い

**代替案**:
- 複数話者で学習後、特定話者でファインチューニング
- 最低10-20話者を使用

### Q3: ステップ数はどれくらい必要？

**推奨** (RTX 4070 Ti SUPER 16GB、潜在表現キャッシュ使用):
- テスト/実験: 10,000ステップ（約16時間）
- 本格的な学習: 50,000-100,000ステップ（約74-147時間 = 3.1-6.1日）

loss値が0.1-0.3に収束するまで学習することを推奨します。

**注意**: より高性能なGPU（A100、H100など）では大幅に高速化されます。

## ベストプラクティス

### 1. データの品質
- JVS corpus使用（プロの声優による高品質音声）
- 24kHz、16bit、モノラル
- 正確な日本語転写

### 2. ハイパーパラメータ
- `LEARNING_RATE = 1e-4`（一から学習用）
- `NUM_STEPS = 100_000`（全100話者用）
- `WARMUP_STEPS = 1_000`（総ステップの1%）
- `BATCH_SIZE = 60`（潜在表現キャッシュ使用時、RTX 4070 Ti SUPER最適化）
- `NUM_WORKERS = 4`（キャッシュ使用時）

### 3. チェックポイント管理
- 定期的に音声生成をテスト（loss値だけでなく音質を確認）
- 複数のチェックポイントを保持（5k、10k、20k、50k、100k）
- ベストチェックポイントを選択（最終ステップとは限らない）

### 4. 学習の監視
- `loss`が0.1-0.3に収束することを確認
- 過学習の兆候（loss減少が止まる）に注意
- 定期的に音声品質をリスニングチェック

## 次のステップ

### DMD2蒸留（オプション）

Teacherモデル（128ステップ）を4ステップに蒸留して推論速度を向上:

```bash
# ASRモデルの訓練（日本語データ）
uv run accelerate launch scripts/train/dmd2/asr.py

# SVモデルの訓練（日本語データ）
uv run accelerate launch scripts/train/dmd2/sv.py

# DMD2蒸留（128ステップ → 4ステップ）
uv run accelerate launch scripts/train/dmd2/distill.py
```

詳細は`scripts/train/dmd2/`のスクリプトを参照してください。

## 参考資料

### データセット
- [JVS Corpus公式サイト](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- [JVS論文](https://arxiv.org/abs/1908.06248)

### ツール
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
- [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus)
- [Accelerate (Hugging Face)](https://huggingface.co/docs/accelerate/)

### SmallTTS関連
- [SmallTTS README](../README.md)
- [CLAUDE.md](../CLAUDE.md)

## まとめ

このガイドで実現できること:
- JVS corpus（全100話者、約10,000サンプル）を使用
- 日本語専用TTSモデルを一から学習
- マルチスピーカー対応
- 実測訓練時間: 約147時間（6.1日）で100,000ステップ（RTX 4070 Ti SUPER 16GB）
- 出力: 日本語専用Teacherモデル（128ステップ推論）

### 完全な学習手順（一から環境構築）

```bash
# 1. 環境セットアップ
uv python install 3.12.7
uv sync
uv pip uninstall torch torchvision torchaudio
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# 2. JVSデータセットのダウンロードと解凍
# https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
# jvs_ver1.zipをdata/に解凍

# 3. 潜在表現キャッシュの作成（4-6時間）
uv run python scripts/preprocess/cache_jvs_latents.py

# 4. 学習開始（100,000ステップで約147時間 = 6.1日）
# scripts/train/teacher_japanese.py でNUM_STEPS=100_000に設定
uv run --no-sync accelerate launch scripts/train/teacher_japanese.py

# 5. 推論テスト
uv run python scripts/infer/test_teacher_japanese.py \
  --checkpoint assets/teacher_checkpoints_ja/checkpoint_final.pt \
  --reference data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav \
  --transcription "参照音声の転写" \
  --text "生成したいテキスト"
```

質問や問題がある場合は、GitHubのIssuesで報告してください。

---

**最終更新**: 2025-10-14
**対応バージョン**: SmallTTS v0.1.0+
