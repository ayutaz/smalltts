# Japanese Fine-tuning Guide for SmallTTS

このドキュメントでは、SmallTTSモデルを日本語データセット（JVS corpus）でファインチューニングする方法を説明します。

## 概要

SmallTTSは元々英語音声合成用に訓練されたモデルですが、多言語音素語彙（205トークン）に対応しているため、日本語への拡張が可能です。このガイドでは、既存の英語Teacherモデルを日本語JVSデータセットでファインチューニングする手順を説明します。

## 前提条件

### 1. JVSデータセットの準備

JVS (Japanese Versatile Speech) corpusをダウンロードして配置します。

**ダウンロード**: [JVS corpus公式サイト](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)

**推奨ディレクトリ構造**:
```
data/
  jvs_ver1/
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
          ...
        transcripts_utf8.txt
    ...
```

**転写ファイル形式** (`transcripts_utf8.txt`):
```
VOICEACTRESS100_001:それは確かにそうです。
VOICEACTRESS100_002:今日はいい天気ですね。
VOICEACTRESS100_003:音声合成システムを作っています。
...
```

### 2. 依存関係のインストール

日本語音素化のために`pyopenjtalk-plus`が必要です:

```bash
uv pip install pyopenjtalk-plus
```

### 3. ベースモデルの確認

英語Teacherモデルは初回実行時に自動的にダウンロードされます。手動でダウンロードする場合:

```bash
uv run python -c "from smalltts.assets.ensure import ensure_assets; ensure_assets(['teacher_checkpoints'])"
```

## ファインチューニング手順

### Step 1: データセットパスの設定

`scripts/train/teacher_japanese.py`を編集し、JVSデータセットのパスを設定:

```python
# Dataset paths
AUDIO_DIR = "data/jvs_ver1/jvs001/parallel100/wav24kHz16bit"
TRANSCRIPT_FILE = "data/jvs_ver1/jvs001/parallel100/transcripts_utf8.txt"
```

**複数話者を使用する場合**:
複数のJVS話者データを結合するには、データローダーを修正するか、シンボリックリンクで統合ディレクトリを作成します。

### Step 2: ハイパーパラメータの調整（オプション）

必要に応じて、`teacher_japanese.py`の設定を調整:

```python
# Training parameters
BATCH_SIZE = 2          # GPUメモリに応じて調整（4, 8など）
NUM_WORKERS = 0         # データローダーのワーカー数
NUM_STEPS = 50_000      # ファインチューニングステップ数
NUM_SAVE_STEPS = 1_000  # チェックポイント保存間隔

# Fine-tuning learning rate
LEARNING_RATE = 1e-5    # 英語モデルより低い学習率
WARMUP_STEPS = 1_000    # ウォームアップステップ数
```

**推奨設定**:
- **単一話者**: 20,000-50,000ステップ
- **複数話者**: 50,000-100,000ステップ
- **学習率**: 1e-5（英語訓練の1e-4より低く設定）

### Step 3: ファインチューニングの実行

#### Single GPU:
```bash
uv run --no-sync accelerate launch scripts/train/teacher_japanese.py
```

#### Multi-GPU:
```bash
uv run --no-sync accelerate launch --multi-gpu scripts/train/teacher_japanese.py
```

**注意**: `--no-sync`オプションは必須です。これにより、CPU版PyTorchの再インストールを防ぎます。

### Step 4: 訓練の監視

訓練中は以下の情報が表示されます:

```
Fine-tuning: 100%|████████| 50000/50000 [10:23:15<00:00, loss=0.245, lr=9.87e-06]
```

- **loss**: 速度予測のMSE損失（0.1-0.3が良好）
- **lr**: 現在の学習率

TensorBoard（オプション）:
```bash
tensorboard --logdir=assets/teacher_checkpoints_ja
```

## チェックポイントの管理

### 保存されるチェックポイント

ファインチューニング中、以下のチェックポイントが保存されます:

```
assets/teacher_checkpoints_ja/
  checkpoint_step_1000/      # 訓練再開用（オプティマイザ状態含む）
  checkpoint_step_1000.pt    # モデルの重みのみ
  checkpoint_step_2000/
  checkpoint_step_2000.pt
  ...
  checkpoint_latest.pt       # 最新のチェックポイント
  checkpoint_final.pt        # 最終チェックポイント
```

### 訓練の再開

訓練を中断した場合、最後のチェックポイントから再開できます:

```python
# teacher_japanese.pyで設定
LOAD_FROM_CHECKPOINT = "assets/teacher_checkpoints_ja/checkpoint_step_10000"
```

## 推論

### 音素埋め込みの確認

ファインチューニングされたモデルは自動的に205トークンの多言語音素語彙を使用します。

### テスト推論

```python
from smalltts.data.phonemization.phonemes import set_language, get_token_ids

# 多言語モードに設定
set_language("multilingual")

# 日本語テキストの音素化
text = "こんにちは、世界！"
phoneme_tokens = get_token_ids(text)
print(f"Phoneme tokens: {phoneme_tokens}")
```

### モデルのロード

```python
import torch
from smalltts.models.backbone.model import Backbone

# モデルの初期化
model = Backbone(latent_dim=64)

# ファインチューニングされたチェックポイントのロード
checkpoint = torch.load("assets/teacher_checkpoints_ja/checkpoint_latest.pt")
model.load_state_dict(checkpoint["model"])
model.eval()
```

## トラブルシューティング

### 問題 1: pyopenjtalk not found

**エラー**: `pyopenjtalk-plus not installed`

**解決策**:
```bash
uv pip install pyopenjtalk-plus
```

### 問題 2: 音素語彙サイズの不一致

**エラー**: `size mismatch for phoneme_embedding.phoneme_embed.weight`

**原因**: モデルが多言語モード（205トークン）に設定されていない

**解決策**:
```python
from smalltts.data.phonemization.phonemes import set_language
set_language("multilingual")  # 必ず最初に実行
```

### 問題 3: CUDA out of memory

**解決策**:
- `BATCH_SIZE`を減らす（2 → 1）
- `max_audio_length_sec`を減らす（30 → 20）
- より小さなGPUの場合、gradient checkpointingを有効化

### 問題 4: 損失が減少しない

**原因**: 学習率が高すぎる、または低すぎる

**解決策**:
- `LEARNING_RATE`を調整（1e-6 ～ 1e-4の範囲で試す）
- `WARMUP_STEPS`を増やす（1000 → 2000）

## ベストプラクティス

### 1. データの品質

- **音声品質**: 24kHz、モノラル、ノイズが少ない
- **転写精度**: 正確な日本語転写が重要
- **データ量**: 最低30分、推奨1時間以上

### 2. ハイパーパラメータ

- **学習率**: 1e-5から開始（英語訓練より低く）
- **ステップ数**: データ量に応じて20k-100k
- **バッチサイズ**: GPUメモリに応じて調整

### 3. 検証

- 定期的に生成音声をリスニングチェック
- 損失値だけでなく、音声品質を確認
- 過学習を避けるため、早期停止を検討

## 次のステップ: DMD2蒸留（オプション）

ファインチューニングされたTeacherモデルの推論速度を向上させるには、DMD2蒸留を実行します:

### 前提条件
1. 日本語でファインチューニングされたTeacherモデル
2. 日本語データでトレーニングされたASRモデル
3. 日本語データでトレーニングされたSVモデル

### 蒸留手順
```bash
# ASRモデルのトレーニング（日本語データ）
uv run accelerate launch scripts/train/dmd2/asr.py

# SVモデルのトレーニング（日本語データ）
uv run accelerate launch scripts/train/dmd2/sv.py

# DMD2蒸留（128ステップ → 4ステップ）
uv run accelerate launch scripts/train/dmd2/distill.py
```

詳細は`scripts/train/dmd2/`のスクリプトを参照してください。

## 参考資料

- [JVS Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
- [SmallTTS README](../README.md)
- [Japanese Phonemization Tasks](japanese-phonemization-tasks.md)

## まとめ

このガイドでは、SmallTTSモデルを日本語JVSデータセットでファインチューニングする方法を説明しました。主なステップは:

1. ✅ JVSデータセットの準備
2. ✅ `teacher_japanese.py`でパスとパラメータを設定
3. ✅ ファインチューニングの実行
4. ✅ チェックポイントの検証

質問や問題がある場合は、GitHubのIssuesで報告してください。
