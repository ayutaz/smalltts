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
  - **phonemes.py**: 日本語専用音素化システム（92トークン語彙、pyopenjtalk-plus使用）
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

SmallTTSは日本語専用音素語彙（92トークン）により、日本語音声合成をサポートします。

### 音素化システム

**日本語専用モード**:
- 日本語音素: 64トークン（基本音素）
- 韻律マーカー: 28トークン（アクセント核、句境界、品詞、モーラ数、ポーズ）
- 合計: 92トークン語彙
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
    codec_encoder=None,  # 潜在表現キャッシュ使用時はNone
    cache_dir="data/jvs_ver1_latents",  # 事前キャッシュされた潜在表現
    batch_size=60,  # RTX 4070 Ti SUPER最適化（キャッシュ使用時）
    num_workers=4,  # キャッシュ使用時は複数ワーカー可能
)
```

### 日本語モデルの学習

JVS corpusを使用して、日本語専用TTSモデルを一から学習:

```bash
# Python 3.12.7 + CUDA 12.4のセットアップ後
uv run --no-sync accelerate launch scripts/train/teacher_japanese.py
```

**訓練設定**:
- 一から学習（training from scratch）
- 全100話者のマルチスピーカー対応
- 学習率: 1e-4（10k warmup + cosine annealing）
- バッチサイズ: 80（10 × 2 gradient accumulation × 4 GPUs）
- GPU: Tesla T4 × 4（各15GB VRAM）
- 推奨ステップ数: 600,000ステップ（英語モデルと同等）

**実測訓練時間** (Tesla T4 × 4):
- 300,000ステップ: 約56時間（2025-10-22実測）
- 600,000ステップ: 約112時間（4.7日、推定）
- 平均速度: ~1.6 it/s

**注意**: より高性能なGPU（A100、H100など）では大幅に高速化されます。

**詳細ガイド**: `docs/japanese_training_guide.md`を参照

### 日本語教師モデルの推論

300,000ステップ訓練後の教師モデルでの推論:

```bash
uv run python scripts/infer/test_teacher_japanese.py \
  --checkpoint assets/teacher_checkpoints_ja_accent_corrected/checkpoint_final.pt \
  --reference data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav \
  --transcription "参照音声の転写テキスト" \
  --text "生成したいテキスト" \
  --output "out/output.wav" \
  --steps 128 \
  --cfg-scale 3.0
```

**推奨パラメータ**:
- 拡散ステップ数: 128 steps
- CFG scale: 3.0（最も安定した結果）
- 条件付け長さ: ターゲット長さの30%
- Codec FPS: 7.54 fps（実測値）

**音質に関する注意**:
- 300,000ステップモデルは基本的な日本語発音を生成可能
- ところどころ発音が不安定（無声母音情報の損失が原因 - 後述）
- 実用的な音質には600,000ステップの訓練が推奨される
- 英語モデルは600,000ステップ訓練されている（参考）

---

## 日本語モデル開発の現状と課題（2025-10-23更新）

### 現在のモデル状態

**完了した訓練**:
- **訓練ステップ**: 300,000 steps（2025-10-22完了）
- **訓練時間**: 約56時間（Tesla T4 × 4 GPU）
- **データセット**: JVS corpus（100話者、parallel100サブセット）
- **チェックポイント**: `assets/teacher_checkpoints_ja_accent_corrected/`
- **Hugging Face**: [ayousanz/smalltts-ja](https://huggingface.co/ayousanz/smalltts-ja)（アップロード予定）

**現在の能力**:
- ✅ 日本語の基本的な発音生成が可能
- ✅ 100話者の音声クローニング対応
- ✅ アクセント核、句境界、韻律情報を考慮
- ⚠️ 発音が部分的に不安定（詳細は後述）

---

### 発見・修正された問題

#### 1. アクセント核位置バグ（修正済み）

**発見日**: 2025-10-19
**修正日**: 2025-10-19
**状態**: ✅ 修正済み

**問題**:
```python
# src/smalltts/data/phonemization/phonemes.py:289-290（修正前）
if i + 1 < len(mora_positions) and mora_positions[i + 1] != mora_pos:
    result['accent_nuclei'].append(i + 1)  # ← バグ: i+1ではなくiが正しい
```

アクセント核マーカー（↓）の位置が1つずれていた。例:
- 「こんにちは」(ko N ni chi wa): `s ↓`（間違い）→ `a ↓`（正しい）
- 「世界」(se ka i): `k ↓`（間違い）→ `e ↓`（正しい）

**修正内容**:
```python
# 修正後
if i + 1 < len(mora_positions) and mora_positions[i + 1] != mora_pos:
    result['accent_nuclei'].append(i)  # i+1 → i
```

**影響**: 300kステップ訓練は修正後のコードで実施されているため、**この問題は解決済み**。

**詳細**: `docs/accent_nucleus_bug_fix.md`参照

---

#### 2. 推論時の条件付けバグ（修正済み）

**発見日**: 2025-10-22
**修正日**: 2025-10-22
**状態**: ✅ 修正済み

**問題1: 時間計算の誤り**
```python
# scripts/infer/test_teacher_japanese.py（修正前）
reference_length = reference_latents.shape[1]  # 65 frames（0.87秒相当）
# ← 実際の音声は8.62秒なのに、潜在フレーム数を使用していた
```

**修正**:
```python
# 修正後
reference_audio_duration_sec = x.shape[1] / 24000  # 実際の秒数を計算
codec_fps = reference_latents.shape[1] / reference_audio_duration_sec  # 7.54 fps
```

**問題2: 条件付け比率の誤り**
```python
# 修正前
cond_length = int(reference_latents.shape[1] * 0.5)  # 50% of reference
# ← 短いターゲットで93%以上の条件付けになり、全て参照音声と同じ発音に
```

**修正**:
```python
# 修正後
cond_length = int(target_length * 0.3)  # 30% of target
# ← 訓練時の分布（0-50%、平均~25%）と一致
```

**影響**: 推論スクリプトが修正され、正常に異なるテキストの発音が生成可能に。

**ファイル**: `scripts/infer/test_teacher_japanese.py:161-177, 21-62`

---

#### 3. 無声母音情報の損失（**修正済み**）

**発見日**: 2025-10-23
**修正日**: 2025-10-23
**状態**: ✅ **修正済み（再学習準備完了）**

**問題**:
```python
# src/smalltts/data/phonemization/phonemes.py:209-212（修正前）
if curr_phoneme in ['A', 'I', 'U', 'E', 'O']:
    curr_phoneme = curr_phoneme.lower()  # ← 無声母音情報を削除
```

**詳細**:
- pyopenjtalkは無声母音を大文字（A, I, U, E, O）で出力
- コードがこれを小文字に変換していた
- 結果: モデルが有声/無声の区別を学習できなかった

**検証例**:
```
テキスト: "です"
修正前: d e s u    （u = 有声）← 情報損失
修正後: d e s U  （U = 無声）✓
```

**影響（300kモデル）**:
- 日本語では「です」「ます」「した」などで母音が無声化する
- 300kモデルは全て有声で発音を学習
- これが**発音不安定の主要因**だった
- 語彙92トークン中、実質64トークンしか使用されていなかった

**修正内容**:
```python
# phonemes.py:209-212 を削除（大文字のまま保持）
# Note: Unvoiced vowels (A, I, U, E, O) are now preserved as uppercase
# to maintain voiced/unvoiced distinction for natural Japanese pronunciation
```

**修正後の検証結果**:
```python
from smalltts.data.phonemization.phonemes import get_token_ids
# "です" → 'd', 'e', 's', 'U'  ✓ 無声母音が保持される
# 語彙サイズ: 92トークン（全て使用可能）✓
```

**次のステップ**: ゼロから600kステップまで再学習
- ブランチ: `fix-unvoiced-vowels-phonemization`
- 保存先: `assets/teacher_checkpoints_ja_unvoiced`
- 推定時間: 約112時間（Tesla T4 × 4）

**期待される効果**:
- 無声母音（A, I, U, E, O）が適切に学習される
- 「です」「ます」「した」などの発音が自然に
- 語彙全92トークンを適切に活用
- 発音の安定性が大幅に向上

---

### その他の潜在的な問題

#### 4. 品詞（POS）コードのマッピング不足（優先度: 中）

**問題**:
```python
# phonemes.py:266-274
pos_map = {
    '02': '[N]',    # 名詞
    '10': '[V]',    # 動詞
    '20': '[ADJ]',  # 形容詞
    '24': '[PART]', # 助詞
    '14': '[AUX]',  # 助動詞
    '01': '[SYM]',  # 記号
}
result['pos_tags'][phoneme_idx] = pos_map.get(pos_code, '[N]')  # デフォルト: [N]
```

**影響**:
- 6種類のPOSタグのみマッピング
- 未知のPOSコードは全て`[N]`（名詞）扱い
- 副詞、連体詞、接続詞などが誤った品詞タグを持つ可能性

**pyopenjtalk警告**:
```
WARNING: convert_pos() in njd2jpcommon.c: 助動詞 非自立 助動詞語幹 * are not appropriate POS.
```
→ pyopenjtalk自体が一部の助動詞を適切に処理できていない

#### 5. モーラ数の上限（優先度: 低）

**問題**: `phonemes.py:275`で10モーラに制限
```python
result['mora_counts'][phoneme_idx] = min(phrase_len, 10)  # 10で打ち切り
```

**影響**: 10モーラ超のフレーズの長さ情報が失われる（実用上は稀）

#### 6. 単独長音符のエッジケース（優先度: 低）

**問題**: テキストに単独で'ー'が現れた場合、音素が生成されない
```python
入力: "ー"
出力: "^ $"  （空）
```

**影響**: 実用上ほとんど発生しない

---

### 実行済みアクション

#### ✅ 無声母音バグ修正完了（2025-10-23）

**完了したステップ**:

1. ✅ **300kモデルをHugging Faceに保存**
   - 保存先: [ayousanz/smalltts-ja](https://huggingface.co/ayousanz/smalltts-ja)
   - ラベル: "300k steps, voiced vowels only, accent corrected"
   - ローカル: `assets/teacher_checkpoints_ja_accent_corrected/`

2. ✅ **音素化コードを修正**
   - ブランチ作成: `fix-unvoiced-vowels-phonemization`
   - 修正内容: `phonemes.py:209-212` 削除
   - コミット: e3d92e8
   - テスト: 無声母音（A,I,U,E,O）が正しく保持されることを確認

3. ✅ **訓練設定を更新**
   - OUTPUT_DIR: `assets/teacher_checkpoints_ja_unvoiced`（300kモデルと分離）
   - NUM_STEPS: 600,000（英語モデルと同等）
   - コミット: 3db9739

**現在の状態**: 🔄 **600kステップ訓練の準備完了**

---

### 次のアクション

#### 優先度1: 600kステップ訓練の実行（**準備完了**）

**訓練開始コマンド**:
```bash
# ブランチ: fix-unvoiced-vowels-phonemization（現在のブランチ）
uv run --no-sync accelerate launch scripts/train/teacher_japanese.py
```

**訓練設定**:
- ステップ数: 600,000
- 保存先: `assets/teacher_checkpoints_ja_unvoiced/`
- チェックポイント: 50kステップごと
- 推定時間: 約112時間（4.7日、Tesla T4 × 4）

**期待される効果**:
- 無声母音（A, I, U, E, O）が適切に学習される
- 「です」「ます」「した」などの発音が自然に
- 語彙全92トークンを適切に活用
- 発音の安定性が大幅に向上
- 300kモデル（有声のみ）との比較検証が可能

**モデル比較**:

| モデル | ステップ数 | 語彙 | 特徴 | ブランチ | 状態 |
|--------|-----------|------|------|---------|------|
| 300k (旧) | 300,000 | 64トークン実使用 | 有声のみ | japanese-phonemization | ✅ 完了・HF公開 |
| **600k (新)** | **600,000** | **92トークン全使用** | **有声+無声** | **fix-unvoiced-vowels-phonemization** | 🔄 **準備完了** |

---

#### 優先度2: POSコードマッピングの拡充（オプション）

- 副詞、連体詞、接続詞などのPOSコードを追加
- より正確な韻律情報の提供
- 600k訓練完了後に検討

---

#### 優先度3: DMD2蒸留（600k教師モデル完成後）

無声母音修正版の600k教師モデルが完成したら:
1. DMD2蒸留で128ステップ → 4ステップに削減
2. リアルタイム生成が可能に
3. CPU推論対応

---

### 参考リソース

**ドキュメント**:
- `docs/accent_nucleus_bug_fix.md`: アクセント核バグの詳細
- `docs/japanese_training_guide.md`: 訓練ガイド
- `assets/teacher_checkpoints_ja_accent_corrected/README.md`: モデルカード

**主要ファイル**:
- `src/smalltts/data/phonemization/phonemes.py`: 音素化実装
- `scripts/train/teacher_japanese.py`: 訓練スクリプト
- `scripts/infer/test_teacher_japanese.py`: 推論スクリプト
- `scripts/debug/debug_phonemes.py`: 音素化デバッグツール

**GitHub**: [ayutaz/smalltts/tree/japanese-phonemization](https://github.com/ayutaz/smalltts/tree/japanese-phonemization)

**Hugging Face**: [ayousanz/smalltts-ja](https://huggingface.co/ayousanz/smalltts-ja)（近日公開）

---

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
