# CLAUDE.md

このファイルは、このリポジトリで作業する際にClaude Code (claude.ai/code)にガイダンスを提供します。

## プロジェクト概要

**smalltts**は、音声クローニング機能を持つ表現豊かなテキスト音声合成システムです。Distribution Matching Distillation (DMD2)を使用して128ステップの拡散モデルを4ステップに削減し、CPUとGPUでリアルタイム性能を実現しています。モデルは感情表現が豊かで、キャラクターボイスに適しています。

## 主要な技術的アプローチ

1. **教師モデル**: 128サンプリングステップで音声を生成する拡散TTSモデル
2. **潜在空間**: 生の音声/メルスペクトログラムではなく、エンコードされた潜在表現で動作。Microsoft VibeVoiceのエンコーダー/デコーダーを使用し、約3200倍の圧縮率
3. **DMD2蒸留**: Distribution Matching Distillationを使用して教師モデルを4ステップに蒸留
4. **補助モデル**: 音素とスタイルのアライメントを維持するために、蒸留中にASR（自動音声認識）とスピーカー検証モデルを訓練

## アーキテクチャコンポーネント

### コアモデル (src/smalltts/models/)

- **Backbone (backbone/model.py)**: 以下を含むメイン拡散モデル:
  - TimeEmbedding: 拡散タイムステップ用の正弦波時間埋め込み
  - PhonemeEmbedding: 音素トークンの埋め込み
  - DiT (Diffusion Transformer): セルフアテンションとクロスアテンションブロックを持つトランスフォーマーアーキテクチャ
  - 拡散用の速度予測ヘッド

- **DiT (backbone/dit.py)**: 以下を含むDiffusion Transformer実装:
  - AdaLayerNormZero: 時間で条件付けされた適応的レイヤー正規化
  - Attention: RoPE（回転位置埋め込み）を使用したセルフアテンション
  - CrossAttention: 潜在表現と音素埋め込み間のクロスアテンション
  - ConvPositionEmbedding: 畳み込み位置埋め込み
  - DiTBlock: セルフアテンション、クロスアテンション（一部のレイヤー）、フィードフォワードを組み合わせたトランスフォーマーブロック

- **ASR (asr.py)**: 蒸留中に音素アライメントを維持するために使用される自動音声認識モデル

- **SV (sv/model.py)**: 音声特性を保持するために蒸留中に使用されるスピーカー検証モデル

- **Discriminator (discriminator.py)**: DMD訓練で実際のサンプルと生成されたサンプルを区別するために使用

### 推論 (src/smalltts/infer/)

- **onnx.py**: 以下を行うメイン推論インターフェース`SmallTTS`クラス:
  - ONNXモデル（e2eモデル + 長さ予測器）をロード
  - 条件付け音声潜在表現、転写、ターゲットテキストを受け取る
  - 出力長を予測し、4ステップ生成を実行
  - 生成された音声テンソルを返す

### コーデック (src/smalltts/codec/)

- **onnx.py**: VibeVoice ONNX エンコーダー/デコーダーのラッパー:
  - Encoder: 24kHz音声 → 潜在表現（形状: batch × T × 64）
  - Decoder: 潜在表現 → 音声（約3200倍の圧縮率）

### データ (src/smalltts/data/)

- **dummy.py**: ダミーデータローダー（訓練には実データに置き換える必要あり）
- **japanese.py**: JVS (Japanese Versatile Speech corpus) データローダー（マルチスピーカー対応）
- **phonemization/**: テキスト正規化と音素トークン化
  - **phonemes.py**: 日本語専用音素化システム（64トークン、pyopenjtalk-plus使用）
  - **normalizer_ja.py**: 日本語テキスト正規化

## 訓練パイプライン

`accelerate`を使用して、以下の特定の順序で訓練を実行する必要があります:

1. **教師モデル** (128ステップ):
   ```bash
   uv run accelerate launch scripts/train/teacher.py
   ```
   - 完全な拡散モデルを訓練
   - `assets/teacher_checkpoints/`に保存

2. **DMD2蒸留** (4ステップ):
   ```bash
   uv run accelerate launch scripts/train/dmd2/distill.py
   ```
   - 教師を4ステップの生徒モデルに蒸留
   - discriminator、student、student_scorerモデルを訓練
   - ASRとSVの補助損失を使用（ステップ5000と7000の後に有効化）
   - `assets/dmd_checkpoints/`に保存

3. **スピーカー検証**:
   ```bash
   uv run accelerate launch scripts/train/dmd2/sv.py
   ```
   - SVモデルを訓練
   - `assets/sv_checkpoints/`に保存

4. **ASR**:
   ```bash
   uv run accelerate launch scripts/train/dmd2/asr.py
   ```
   - ASRモデルを訓練
   - `assets/asr_checkpoints/`に保存

## 推論の使い方

### クイックテスト
```bash
uv run python scripts/tryme.py "Hello from smallTTS!"
```
出力は`out/tryme.wav`に保存されます

### インタラクティブ（CPUでリアルタイム）
```bash
uv run python scripts/infer/interactive.py
```

### バッチ推論
```bash
uv run python scripts/infer/batch.py
```

### 音声クローニング
```bash
uv run python scripts/infer/clone.py \
  --wav assets/test_audio/1.wav \
  --transcription "参照音声の転写テキスト" \
  --text "生成したいテキスト"
```

### Python API
```python
from smalltts import SmallTTS
import torch

tts = SmallTTS()
# conditionings: 潜在テンソルのリスト（形状: T × 64）
# transcriptions: 音素文字列またはトークンリストのリスト
# texts: ターゲットテキスト文字列またはトークンリストのリスト
audios = tts(conditionings, transcriptions, texts)
```

## 主要な技術詳細

### 拡散プロセス
- 速度パラメータ化を使用: `v = α * noise - σ * x`
- 蒸留用のタイムステップスケジュール: `[1.0, 1.0, 0.75, 0.50, 0.25]`（4ステップ）
- 生成時にCFG（Classifier-Free Guidance）スケール3.0を使用

### 条件付け
- ランダム条件付け戦略: グラウンドトゥルース音声の一部がコンテキストとして提供される
- 条件付けマスクは、既知の音声の予測をモデルに防止

### マスク
- `phonemes_mask`: 有効な音素位置でTrue
- `mask`: 有効な潜在シーケンス位置でTrue
- `cond_mask`: 条件付け（既知）音声でTrue、生成する位置でFalse

### モデルの次元
- 潜在次元: 64
- 隠れ層次元: 896
- 音素次元: 512
- トランスフォーマーブロック数: 18
- 約半分のレイヤーでクロスアテンション（インデックス: 0, 1, 3, 5, 7, 11, 13, 15, 17）

## インストール

依存関係管理には[uv](https://github.com/astral-sh/uv)を使用:
```bash
uv pip install "git+https://github.com/smallbraineng/smalltts"
```

依存関係には以下が含まれます: accelerate, torch, torchaudio, onnxruntime, phonemizer, soundfile, speechbrain, huggingface_hub など

## アセットとチェックポイント

すべてのモデルはHuggingFaceでホスト: [smallbraineng/smalltts](https://huggingface.co/smallbraineng/smalltts)

- `assets/teacher_checkpoints/`: 教師モデルの重み
- `assets/dmd_checkpoints/`: 蒸留された生徒モデルの重み
- `assets/asr_checkpoints/`: ASRモデルの重み
- `assets/sv_checkpoints/`: スピーカー検証の重み
- `assets/e2e/`: ONNXエンドツーエンド推論モデル
- `assets/length/`: ONNX長さ予測器
- `assets/codec/`: VibeVoiceエンコーダー/デコーダーONNXモデル

アセットは必要に応じて`smalltts.assets.ensure.ensure_assets()`により自動的にダウンロードされます。

## データ要件

- デフォルトのデータローダーはダミーデータ（`data/dummy.py`）
- 訓練には以下を含む実際の音声データが必要:
  - 音声ファイル（24kHz、モノラル推奨）
  - 対応する転写テキスト
  - 音素化されたテキスト
- プロジェクトはWebDataset形式でテスト済み

## 日本語対応

SmallTTSは日本語専用音素語彙（64トークン）により、日本語音声合成をサポートします。

### 音素化システム

**日本語専用モード**:
- 日本語音素のみ（64トークン）
- pyopenjtalk-plusによる日本語音素化
- 英語・espeak-ng依存を完全に削除

```python
from smalltts.data.phonemization.phonemes import set_language, get_token_ids

# 日本語モードに設定
set_language("ja")

# 日本語テキストの音素化
tokens_ja = get_token_ids("こんにちは、世界！")
```

### 日本語データセット

**JVS (Japanese Versatile Speech corpus)**:
- 30時間、100話者
- 24kHz、高品質
- データローダー: `src/smalltts/data/japanese.py`

```python
from smalltts.data.japanese import get_jvs_dataloader

# 全100話者を使用するマルチスピーカー学習
loader = get_jvs_dataloader(
    root_dir="data/jvs_ver1",
    speaker_ids=None,  # None = 全話者を使用
    subset="parallel100",
    batch_size=1,
    num_workers=0  # ONNX encoder使用時は0必須
)
```

### 日本語モデルの学習

JVS corpusを使用して、日本語専用TTSモデルを一から学習:

```bash
# Python 3.12.7 + CUDA 12.4のセットアップ後
uv run --no-sync accelerate launch scripts/train/teacher_japanese.py
```

**特徴**:
- 一から学習（training from scratch）
- 全100話者のマルチスピーカー対応
- 学習率: 1e-4（標準訓練レート）
- 推奨ステップ数: 100,000ステップ（約28時間、単一GPU）

**詳細ガイド**: `docs/japanese_training_guide.md`を参照

## ライセンス

- コード: CC-BY-NC（非商用）
- モデルの重み: CC-BY-NA（非商用、帰属不要）

## 参考文献

以下から着想を得ています:
- DMDSpeech (arXiv:2410.11097)
- DMD2 (arXiv:2405.14867)
- F5-TTS (arXiv:2410.06885)
- VibeVoice (arXiv:2508.19205)
- Nanospeech
