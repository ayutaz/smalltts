# JVS日本語TTSモデル学習ガイド

このドキュメントでは、JVS corpus（全100話者）を使用して、**一から日本語TTSモデルを学習**する方法を説明します。

## 概要

このガイドで実装する内容:
- ✅ **一から学習（Training from scratch）**: 既存の英語モデルを使わず、日本語のみでゼロから学習
- ✅ **全100話者を使用**: JVS corpusの全話者（約10,000サンプル）
- ✅ **マルチスピーカー対応**: 複数話者の多様性を学習
- ✅ **日本語専用音素語彙**: 英語フォニームを含まない、純粋な日本語モデル

## 前提条件

### 1. 環境

**推奨**: GPU環境（CUDA対応）
- VRAM: 16GB以上推奨（BATCH_SIZE=1で約15GB使用）
- RAM: 32GB以上推奨（学習データロード用）
- ストレージ: 約10GB（JVSデータセット + チェックポイント）

**対応OS**:
- Linux（推奨）
- Windows（GPU対応、Python 3.12.7 + CUDA 12.4）
- macOS（CPUのみ、非常に遅い）

### 2. リポジトリのクローン

```bash
git clone https://github.com/yourusername/smalltts.git
cd smalltts
git checkout japanese-phonemization  # 日本語対応ブランチ
```

### 3. Python環境のセットアップ

#### uvのインストール

**Linux/macOS**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Python 3.12とCUDA版PyTorchのインストール

```bash
# Python 3.12.7をインストール
uv python install 3.12.7

# .python-versionファイルを作成（リポジトリに既に存在）
echo "3.12.7" > .python-version

# 依存関係のインストール（日本語音素化ライブラリを含む）
uv sync

# CPU版PyTorchをアンインストールしてCUDA版をインストール
uv pip uninstall torch torchvision torchaudio
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

#### 環境確認

```bash
# PyTorchとCUDAの確認
uv run --no-sync python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 出力例:
# PyTorch: 2.6.0+cu124
# CUDA available: True
```

### 4. JVSデータセットの準備

#### ダウンロード

公式サイトから`jvs_ver1.zip`（約3.4GB）をダウンロード:
- [JVS Corpus公式サイト](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)

#### 解凍

```bash
# Linuxの場合
unzip jvs_ver1.zip -d data/

# Windowsの場合（7-Zip推奨）
"C:\Program Files\7-Zip\7z.exe" x jvs_ver1.zip -o"data" -aoa
```

**解凍時間**:
- Linux: 数分
- Windows（標準）: 数時間～数日
- Windows（7-Zip）: 約2分

#### ディレクトリ構造の確認

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
      parallel100/
        ...
```

**話者数の確認**:
```bash
ls -d data/jvs_ver1/jvs* | wc -l
# 出力: 100
```

**総サンプル数**: 約9,997サンプル（100話者 × 約100発話）

## 学習の実行

### ステップ1: セットアップ検証（推奨）

学習を開始する前に、データローダーとモデルが正しく動作するか確認:

```bash
# マルチスピーカーデータローダーのテスト
uv run python scripts/test_multispeaker.py

# 出力例:
# [OK] Loaded 3 speakers successfully
# [OK] Loaded all 100 speakers successfully
#    Total samples: 9997
```

```bash
# 総合的なセットアップ検証
uv run python scripts/verify_japanese_setup.py

# 出力例:
# [OK] All checks passed!
# Ready to start Japanese-only training from scratch with all 100 JVS speakers.
```

### ステップ2: 訓練パラメータの確認

`scripts/train/teacher_japanese.py`のデフォルト設定:

```python
# Dataset paths - Multi-speaker mode (all 100 JVS speakers)
JVS_ROOT_DIR = "data/jvs_ver1"
SPEAKER_IDS = None  # None = use all speakers (jvs001-jvs100)
SUBSET = "parallel100"  # JVS subset to use

# Training parameters
BATCH_SIZE = 1
NUM_WORKERS = 0  # Must be 0 when using ONNX encoder (CUDA context issue)
NUM_STEPS = 100_000  # Training from scratch requires more steps
NUM_SAVE_STEPS = 5_000

# Checkpoint paths
LOAD_FROM_CHECKPOINT = None  # Train from scratch (Japanese only)
OUTPUT_DIR = "assets/teacher_checkpoints_ja"

# Training from scratch learning rate
LEARNING_RATE = 1e-4  # Standard training rate
WARMUP_STEPS = 1_000  # Warmup for first 10% of training
WEIGHT_DECAY = 1e-2
```

**パラメータの説明**:
- `JVS_ROOT_DIR`: JVSデータセットのルートディレクトリ
- `SPEAKER_IDS`: 使用する話者ID（`None`で全話者を使用）
- `NUM_STEPS`: 総学習ステップ数（100,000ステップ）
- `LEARNING_RATE`: 学習率（一から学習用に1e-4）
- `LOAD_FROM_CHECKPOINT`: `None`で一から学習

### ステップ3: 学習の開始

#### Single GPU（推奨）

```bash
uv run --no-sync accelerate launch scripts/train/teacher_japanese.py
```

#### Multi-GPU環境

```bash
uv run --no-sync accelerate launch --multi-gpu scripts/train/teacher_japanese.py
```

**重要**: `--no-sync`オプションは必須です。これにより、実行時にCPU版PyTorchが再インストールされるのを防ぎます。

### ステップ4: 訓練の監視

**コンソール出力**:
```
================================================================================
JAPANESE TEACHER MODEL TRAINING (FROM SCRATCH)
================================================================================

[1/6] Setting up Japanese phoneme vocabulary
Phoneme vocabulary size: 64

[2/6] Loading VibeVoice encoder

[3/6] Loading JVS dataset (multi-speaker)
JVS root directory: data/jvs_ver1
Speaker IDs: All speakers
Subset: parallel100
Loaded 9997 samples from 100 speakers in data/jvs_ver1

[4/6] Setting up distributed training

[5/6] Initializing model

[6/6] Skipping checkpoint loading (training from scratch)

================================================================================
TRAINING CONFIGURATION
================================================================================
Batch size: 1
Learning rate: 0.0001
Weight decay: 0.01
Total steps: 100,000
Warmup steps: 1,000
Save interval: 5,000 steps
Output directory: assets/teacher_checkpoints_ja
================================================================================

Training: 100%|████████████| 100000/100000 [27:46:32<00:00, loss=0.234, lr=8.95e-05]
```

**推定訓練時間**:
- **バッチサイズ1、単一GPU**: 約28時間（10秒/ステップ）
- **バッチサイズ2、単一GPU**: 約14時間（VRAMが許せば）
- **Multi-GPU**: GPUの数に応じて短縮

**重要な指標**:
- `loss`: 速度予測のMSE損失
  - 初期: 0.5-1.0
  - 収束: 0.1-0.3（良好）
- `lr`: 現在の学習率（ウォームアップとコサインアニーリング）

### ステップ5: チェックポイントの確認

**保存場所**: `assets/teacher_checkpoints_ja/`

**保存されるファイル**:
```
assets/teacher_checkpoints_ja/
  checkpoint_step_5000/           # 訓練再開用（optimizerとschedulerを含む）
  checkpoint_step_5000.pt         # モデル重みのみ（推論用）
  checkpoint_step_10000/
  checkpoint_step_10000.pt
  ...
  checkpoint_step_95000/
  checkpoint_step_95000.pt
  checkpoint_latest.pt            # 最新チェックポイント
  checkpoint_final.pt             # 最終チェックポイント（100,000ステップ）
```

**チェックポイントの構造**:
```python
{
    "model": model.state_dict(),      # モデルの重み
    "step": 5000,                     # ステップ番号
    "loss": 0.234,                    # 損失値
}
```

## 訓練の中断と再開

### 訓練の中断

`Ctrl+C`で安全に中断できます（次のチェックポイント保存時に停止）。

### 訓練の再開

```python
# scripts/train/teacher_japanese.py を編集
LOAD_FROM_CHECKPOINT = "assets/teacher_checkpoints_ja/checkpoint_step_50000"
```

再開時は、optimizer状態とscheduler状態も復元されるため、学習を継続できます。

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

### 問題2: NUM_WORKERS > 0 でCUDAエラー

**エラー**: `CUDA failure 3: initialization error`

**原因**: ONNX encoderはマルチプロセスでCUDAコンテキストを共有できません。

**解決策**:
```python
NUM_WORKERS = 0  # 必ず0に設定
```

### 問題3: pyopenjtalk not found

**エラー**: `pyopenjtalk-plus not installed`

**解決策**:
```bash
uv pip install pyopenjtalk-plus
```

### 問題4: 音声が無音

**原因**: 音素マッピングの問題（multilingual modeで日本語テキストが英語フォニームにマッピングされている）

**解決策**: このブランチ（japanese-phonemization）では修正済みです。
```python
# scripts/train/teacher_japanese.py で必ず ja モードを使用
set_language("ja")  # 日本語専用（推奨）
# または
set_language("multilingual")  # 多言語対応（修正済み）
```

### 問題5: 学習が遅い

**原因**: ONNX encoderがボトルネック（リアルタイム音声エンコーディング）

**現状の速度**:
- 約10秒/ステップ（BATCH_SIZE=1）
- NUM_WORKERS=0必須（CUDAコンテキストの制約）

**改善策**:
- バッチサイズを増やす（VRAMが許せば）
- より高速なGPUを使用
- 事前にlatentをキャッシュ（データローダーの改造が必要）

## 推論とテスト

### 学習中のテスト

学習を中断せずに、別のターミナルで推論をテスト:

```bash
# 簡易テスト
uv run python scripts/infer/test_japanese.py
```

### チェックポイントを使った推論

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
print(f"Final loss: {checkpoint['loss']:.4f}")
```

## 次のステップ

### 1. DMD2蒸留（オプション）

Teacherモデル（128ステップ）を4ステップに蒸留して推論速度を向上:

```bash
# ASRモデルの訓練（日本語データ）
uv run accelerate launch scripts/train/dmd2/asr.py

# SVモデルの訓練（日本語データ）
uv run accelerate launch scripts/train/dmd2/sv.py

# DMD2蒸留（128ステップ → 4ステップ）
uv run accelerate launch scripts/train/dmd2/distill.py
```

### 2. ファインチューニング

特定の話者や用途に合わせてファインチューニング:

```python
# 特定の話者でファインチューニング
LOAD_FROM_CHECKPOINT = "assets/teacher_checkpoints_ja/checkpoint_final.pt"
SPEAKER_IDS = ["jvs001"]  # 1話者のみ
NUM_STEPS = 10_000  # 短いステップ数
LEARNING_RATE = 1e-5  # 低い学習率
```

### 3. モデルのエクスポート

ONNXエクスポートで他の環境での推論を可能に:

```bash
# ONNXエクスポート（実装予定）
uv run python scripts/export/export_onnx.py \
  --checkpoint assets/teacher_checkpoints_ja/checkpoint_final.pt \
  --output assets/e2e_ja/model.onnx
```

## ベストプラクティス

### 1. データの品質

- ✅ JVS corpus使用（プロの声優による高品質音声）
- ✅ 24kHz、16bit、モノラル
- ✅ 正確な日本語転写

### 2. ハイパーパラメータ

- ✅ `LEARNING_RATE = 1e-4`（一から学習用）
- ✅ `NUM_STEPS = 100_000`（全100話者用）
- ✅ `WARMUP_STEPS = 1_000`（総ステップの1%）
- ✅ `BATCH_SIZE = 1`（VRAMの制約）

### 3. チェックポイント管理

- 定期的に音声生成をテスト（loss値だけでなく音質を確認）
- 複数のチェックポイントを保持（5k、10k、20k、50k、100k）
- ベストチェックポイントを選択（最終ステップとは限らない）

### 4. 学習の監視

- `loss`が0.1-0.3に収束することを確認
- 過学習の兆候（loss減少が止まる）に注意
- 定期的に音声品質をリスニングチェック

## パフォーマンス最適化

### メモリ使用量

**VRAM**:
- BATCH_SIZE=1: 約15GB
- BATCH_SIZE=2: 約16GB（ほぼ100%）

**RAM**:
- データローダー: 約5-10GB
- WSL2（Windows）: 約20GB

### 速度改善（将来の実装）

- [ ] Latentの事前計算とキャッシュ
- [ ] DataLoaderのマルチプロセス対応（ONNX encoder改善）
- [ ] Mixed precision training（FP16）
- [ ] Gradient accumulation（仮想的にバッチサイズを増やす）

## よくある質問（FAQ）

### Q1: ファインチューニングと一から学習の違いは？

**ファインチューニング**:
- 既存の英語モデルから開始
- 学習率低め（1e-5）
- 少ないステップ数（10k-50k）
- 少ないデータでも可能

**一から学習（このガイド）**:
- ゼロから初期化
- 学習率高め（1e-4）
- 多いステップ数（100k）
- 大量のデータが必要（全100話者）

### Q2: 1話者だけで学習できますか？

可能ですが、推奨しません。理由:
- データ量不足（100発話は少ない）
- 過学習しやすい
- 汎化性能が低い

**代替案**:
- 複数話者で学習後、特定話者でファインチューニング
- 最低10-20話者を使用

### Q3: 学習時間を短縮できますか？

**現実的な方法**:
- より高速なGPU（A100、H100など）
- Multi-GPU訓練
- BATCH_SIZEを増やす（VRAMが許せば）

**将来の実装**:
- Latentの事前計算（10倍高速化の可能性）

### Q4: multilingual と ja モードの違いは？

**jaモード（推奨）**:
- 日本語専用（64トークン）
- 英語非対応
- このガイドで使用

**multilingualモード**:
- 日本語+英語（約205トークン）
- 両言語対応
- 学習に時間がかかる

## 参考資料

### データセット
- [JVS Corpus公式サイト](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- [JVS論文](https://arxiv.org/abs/1908.06248)

### ツール
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
- [Accelerate (Hugging Face)](https://huggingface.co/docs/accelerate/)

### SmallTTS関連
- [SmallTTS README](../README.md)
- [Japanese Phonemization Tasks](japanese-phonemization-tasks.md)
- [Training Japanese (Fine-tuning)](training_japanese.md)

## まとめ

このガイドで実現できること:
- ✅ JVS corpus（全100話者、約10,000サンプル）を使用
- ✅ 一から日本語TTSモデルを学習
- ✅ マルチスピーカー対応
- ✅ 推定訓練時間: 約28時間（単一GPU、BATCH_SIZE=1）
- ✅ 出力: 日本語専用Teacherモデル（128ステップ推論）

**学習開始コマンド**:
```bash
# Single GPU
uv run --no-sync accelerate launch scripts/train/teacher_japanese.py

# Multi-GPU
uv run --no-sync accelerate launch --multi-gpu scripts/train/teacher_japanese.py
```

質問や問題がある場合は、GitHubのIssuesで報告してください。

---

**最終更新**: 2025-10-12
**ブランチ**: `japanese-phonemization`
**対応バージョン**: SmallTTS v0.1.0+
