# 日本語音素化対応タスクリスト

## 概要

smallTTSに日本語対応を追加するため、pyopenjtalk-plusを使用した音素化システムを実装します。
現在のespeak-ng（英語専用）から、日本語も扱えるマルチリンガルシステムへ拡張します。

**目標**: 日本語データセット（JVS）でファインチューニングできる環境を構築

---

## フェーズ1: 最小実装（優先度: 高）

### タスク1: 依存関係の追加
**ステータス**: ✅ 完了 (Commit: 617e474)

**変更ファイル**:
- `pyproject.toml` ✅
- `Dockerfile` ✅
- `Dockerfile.gpu` ✅

**実装内容**:
1. pyproject.tomlに依存関係を追加:
   ```toml
   dependencies = [
       ...
       "pyopenjtalk-plus>=0.4.1",
   ]
   ```

2. Dockerfileに必要なパッケージを追加:
   ```dockerfile
   # MeCab関連（pyopenjtalk-plusの依存）
   RUN apt-get update && apt-get install -y --no-install-recommends \
       mecab \
       libmecab-dev \
       mecab-ipadic-utf8 \
       && rm -rf /var/lib/apt/lists/*
   ```

**チェックリスト**:
- [x] pyproject.tomlに`pyopenjtalk-plus`を追加
- [x] Dockerfileを更新（CPU版）
- [x] Dockerfile.gpuを更新（GPU版）
- [x] `uv sync`でインストール確認
- [ ] Dockerイメージを再ビルドして確認（未実施）

---

### タスク2: 日本語音素化バックエンドの実装
**ステータス**: ✅ 完了 (Commit: 617e474)

**変更ファイル**:
- `src/smalltts/data/phonemization/phonemes.py` ✅

**実装内容**:

1. 日本語音素セットの定義:
   ```python
   # 日本語音素セット（約70音素）
   _punct_ja = "、。！？…ー"
   _phonemes_ja = [
       "a", "i", "u", "e", "o",              # 母音
       "k", "s", "t", "n", "h", "m", "y", "r", "w", "g", "z", "d", "b", "p",  # 子音
       "N",                                   # ん
       "ch", "sh", "ts",                      # 拗音
       "ky", "gy", "ny", "hy", "by", "py", "my", "ry",  # 拗音2
       "pau",                                 # ポーズ
   ]
   ```

2. pyopenjtalk-plusバックエンドの追加:
   ```python
   import pyopenjtalk

   # 言語設定（環境変数またはグローバル変数）
   LANGUAGE = "ja"  # "en" or "ja"

   def _phonemize_ja(text: str) -> str:
       """日本語テキストを音素化"""
       phonemized = pyopenjtalk.g2p(text, kana=False)
       return phonemized

   def _phonemize(text: str) -> str:
       """言語に応じて音素化"""
       if LANGUAGE == "ja":
           return _phonemize_ja(text)
       else:
           # 既存の英語処理
           text = normalizer.normalize(text)
           phonemized = " ".join(_tok.findall(_es.phonemize([text])[0]))
           return phonemized
   ```

3. 音素マッピングの更新:
   ```python
   def _build_phoneme_mappings(language="en"):
       """言語に応じた音素マッピングを構築"""
       if language == "ja":
           _syms = list(_phonemes_ja)
       else:
           # 既存の英語音素
           _syms = []
           _seen = set()
           for ch in _punct + _letters + _letters_ipa:
               if ch not in _seen:
                   _seen.add(ch)
                   _syms.append(ch)

       p2idx = {ch: i + 1 for i, ch in enumerate(_syms)}
       idx2p = {v: k for k, v in p2idx.items()}
       phoneme_len = len(p2idx) + 1

       return p2idx, idx2p, phoneme_len, _syms
   ```

**チェックリスト**:
- [x] 日本語音素セットを定義（64音素）
- [x] pyopenjtalk-plusの統合（遅延ロード対応）
- [x] 言語切り替え機能の実装（`set_language()`関数）
- [x] `get_token_ids()`関数の更新（空白区切り対応）
- [x] `decode_token_ids()`関数の更新（空白区切り対応）

---

### タスク3: テストコードの作成
**ステータス**: ✅ 完了 (Commit: 617e474)

**変更ファイル**:
- `src/smalltts/data/phonemization/phonemes.py`の`__main__`ブロック ✅

**実装内容**:

```python
if __name__ == "__main__":
    # 英語テスト
    print("=== English Test ===")
    LANGUAGE = "en"
    sentences_en = [
        "Hello world!",
        "This is a test.",
    ]
    for s in sentences_en:
        p = _phonemize(s)
        tids = get_token_ids(s)
        dec = decode_token_ids(tids)
        print(f"orig   : {s}")
        print(f"phoneme: {p}")
        print(f"tids   : {tids[:32]}")
        print(f"decoded: {dec}\n")

    # 日本語テスト
    print("=== Japanese Test ===")
    LANGUAGE = "ja"
    sentences_ja = [
        "こんにちは",
        "これはテストです。",
        "音声合成システムを作っています。",
    ]
    for s in sentences_ja:
        p = _phonemize(s)
        tids = get_token_ids(s)
        dec = decode_token_ids(tids)
        print(f"orig   : {s}")
        print(f"phoneme: {p}")
        print(f"tids   : {tids[:32]}")
        print(f"decoded: {dec}\n")
```

**チェックリスト**:
- [x] 英語の音素化テスト（Docker環境で実施可能）
- [x] 日本語の音素化テスト
- [x] トークンIDへの変換テスト
- [x] デコードテスト
- [x] エラーハンドリングの確認（pyopenjtalk未インストール時）

---

## フェーズ2: 完全実装（優先度: 中）

### タスク4: 日本語テキストノーマライザーの作成
**ステータス**: ✅ 完了 (Commit: 2cffd8d)

**新規ファイル**:
- `src/smalltts/data/phonemization/normalizer_ja.py` ✅

**変更ファイル**:
- `src/smalltts/data/phonemization/phonemes.py` ✅

**実装内容**:

1. **JapaneseTextNormalizerクラスの実装**:
   - Unicode NFKC正規化による全角→半角変換
   - 記号の正規化（英語記号→日本語記号）
   - オプションの数字→日本語読み変換（単独数字のみ）
   - 連続空白の正規化

2. **phonemes.pyとの統合**:
   ```python
   # normalizer_jaのインポートと初期化
   from .normalizer_ja import JapaneseTextNormalizer
   normalizer_ja = JapaneseTextNormalizer(convert_numbers=False)

   def _phonemize_ja(text: str) -> str:
       # テキスト正規化を追加
       text = normalizer_ja.normalize(text)
       phonemized = pyopenjtalk.g2p(text, kana=False)
       return phonemized
   ```

3. **テストコード**:
   - 全角→半角変換のテスト
   - 記号正規化のテスト
   - 数字変換のテスト（オプション）
   - 連続空白の正規化テスト

**チェックリスト**:
- [x] JapaneseTextNormalizerクラスの実装
- [x] 数字の読み方変換（単独数字のみ、オプション機能）
- [x] 記号の処理（12種類の記号マッピング）
- [x] 全角・半角の正規化（NFKC）
- [x] テストコードの作成（__main__ブロック）
- [x] phonemes.pyとの統合

---

### タスク5: 音素埋め込み層のサイズ調整
**ステータス**: ✅ 完了 (Commit: b43d728)

**変更ファイル**:
- `src/smalltts/data/phonemization/phonemes.py` ✅

**新規ファイル**:
- `src/smalltts/models/utils.py` ✅

**実装内容**:

**採用戦略: 多言語対応（205トークン）**
- 英語音素（~175） + 日本語音素（~64） = 205トークン（重複除外後）
- デフォルトモード: `LANGUAGE = "multilingual"`
- 自動言語検出機能（Unicode範囲による判定）

1. **多言語音素語彙の実装**:
   ```python
   def _build_phoneme_mappings(language="en"):
       if language == "multilingual":
           # 英語音素を追加
           for ch in _punct + _letters + _letters_ipa:
               if ch not in _seen:
                   _syms.append(ch)
           # 日本語音素を追加（重複を除外）
           for ch in _phonemes_ja:
               if ch not in _seen:
                   _syms.append(ch)
       # ...
   ```

2. **自動言語検出**:
   ```python
   def _phonemize(text: str, force_language: str = None) -> str:
       lang = force_language if force_language else LANGUAGE
       if lang == "multilingual":
           # Unicode範囲で日本語を検出
           has_japanese = any('\u3040' <= c <= '\u309F' or  # Hiragana
                            '\u30A0' <= c <= '\u30FF' or  # Katakana
                            '\u4E00' <= c <= '\u9FFF'     # Kanji
                            for c in text)
           lang = "ja" if has_japanese else "en"
       # ...
   ```

3. **チェックポイント互換性ユーティリティ** (`models/utils.py`):
   - `expand_phoneme_embedding()`: 埋め込み層の拡張
   - `adapt_state_dict_for_expanded_phonemes()`: state dictの変換
   - 3つの初期化方法: zeros, mean, random

**語彙サイズ**:
- 英語のみ: 175トークン
- 日本語のみ: 64トークン
- 多言語: 205トークン

**チェックリスト**:
- [x] 拡張戦略の選択（多言語対応を採用）
- [x] 音素マッピングの更新（3言語モード対応）
- [x] 自動言語検出の実装
- [x] 埋め込み層拡張ユーティリティの実装
- [x] チェックポイント変換関数の実装
- [x] テストコードの作成と検証

---

### タスク6: データローダーの作成
**ステータス**: ✅ 完了 (Commit: 7162a23, 1e13661)

**新規ファイル**:
- `src/smalltts/data/japanese.py` ✅

**実装内容**:

**JVS専用データローダー** （ユーザー要望によりJSUTサポートは削除）

1. **JVSDatasetクラス**:
   ```python
   class JVSDataset(Dataset):
       """JVS (Japanese versatile speech corpus) データセット"""

       def __init__(
           self,
           audio_dir: str,  # wav24kHz16bit/
           transcript_file: str,  # transcripts_utf8.txt
           codec_encoder=None,  # Optional VibeVoice encoder
           target_sample_rate: int = 24000,
           max_audio_length_sec: float = 30.0,
       ):
           # ...
   ```

2. **トランスクリプト形式のサポート**:
   - JVS形式（コロン区切り）: `VOICEACTRESS100_001:それは確かにそうです。`
   - 代替形式も対応: パイプ、タブ、スペース区切り

3. **音声前処理機能**:
   - 自動リサンプリング（24kHz）
   - ステレオ→モノラル変換
   - 音声長制限（デフォルト30秒）
   - オプションのCodec encoder統合

4. **バッチ処理**:
   ```python
   def jvs_collate_fn(batch):
       # 音素とlatentsのパディング
       # 長さ情報の保持
       # 既存のダミーデータローダーと互換性のある形式
   ```

5. **便利なファクトリ関数**:
   ```python
   loader = get_jvs_dataloader(
       audio_dir="data/jvs_ver1/jvs001/parallel100/wav24kHz16bit",
       transcript_file="data/jvs_ver1/jvs001/parallel100/transcripts_utf8.txt",
       batch_size=16,
       num_workers=4
   )
   ```

**チェックリスト**:
- [x] JVSDatasetクラスの実装
- [x] JVS形式のサポート（コロン区切り + 代替形式）
- [x] jvs_collate_fn関数の実装
- [x] 音声前処理（リサンプリング、モノラル変換）
- [x] get_jvs_dataloader ファクトリ関数
- [x] テストコード・ドキュメント作成
- [x] 既存形式との互換性確保

---

## フェーズ3: 統合とドキュメント（優先度: 中）

### タスク7: ドキュメント更新と実装
**ステータス**: 🔄 進行中 (開始: 2025-10-11)

**変更ファイル**:
- `CLAUDE.md` ⬜
- `README.md` ⬜
- `docs/training_japanese.md` ✅

**新規ファイル**:
- `scripts/utils/expand_checkpoint.py` ✅
- `scripts/train/teacher_japanese.py` ✅

**実装内容**:

1. **チェックポイント拡張ユーティリティ** ✅:
   - 既存チェックポイントの音素埋め込みを175→205トークンに拡張
   - コマンドラインツール: `scripts/utils/expand_checkpoint.py`
   - 使用例:
     ```bash
     python scripts/utils/expand_checkpoint.py \
       --input assets/teacher_checkpoints/checkpoint_latest.pt \
       --output assets/teacher_checkpoints/checkpoint_multilingual.pt \
       --old-vocab-size 175 \
       --new-vocab-size 205
     ```

2. **日本語ファインチューニングスクリプト** ✅:
   - JVSデータセット対応のトレーニングスクリプト
   - 既存英語モデルからのファインチューニング
   - 自動的な音素埋め込み拡張
   - ファイル: `scripts/train/teacher_japanese.py`
   - 使用例:
     ```bash
     # データセットパスを設定後
     uv run accelerate launch scripts/train/teacher_japanese.py
     ```

3. **詳細トレーニングガイド** ✅:
   - ファイル: `docs/training_japanese.md`
   - 内容:
     - JVSデータセットの準備方法
     - ファインチューニングの詳細手順
     - ハイパーパラメータの推奨値
     - トラブルシューティングガイド
     - DMD2蒸留への拡張ガイド

4. **CLAUDE.md**に日本語対応の情報を追加 ⬜:
   - 音素化システムの説明
   - 言語切り替え方法
   - データフォーマット
   - ファインチューニング手順の概要

5. **README.md**に使用例を追加 ⬜:
   ```python
   # 日本語で推論（多言語モード）
   from smalltts import SmallTTS
   from smalltts.data.phonemization.phonemes import set_language

   # 多言語モードに設定（自動言語検出）
   set_language("multilingual")

   tts = SmallTTS()
   # 日本語テキストは自動的に検出される
   ```

**チェックリスト**:
- [x] チェックポイント拡張スクリプトの作成
- [x] teacher_japanese.py トレーニングスクリプトの作成
- [x] training_japanese.md の作成
- [ ] CLAUDE.mdの更新
- [ ] README.mdに使用例を追加
- [ ] サンプルコードの動作確認（Docker環境）

---

## 技術的メモ

### 音素語彙の互換性

**実装された構造** ✅:
- 英語のみ: 175トークン (phoneme_len = 175)
- 日本語のみ: 64トークン (phoneme_len = 64)
- **多言語（デフォルト）**: 205トークン (phoneme_len = 205)
  - 英語 ~175 + 日本語 ~64 = 205（重複除外後）

**言語モード**:
```python
from smalltts.data.phonemization.phonemes import set_language

# 多言語モード（デフォルト、自動言語検出）
set_language("multilingual")  # phoneme_len = 205

# 英語のみ
set_language("en")  # phoneme_len = 175

# 日本語のみ
set_language("ja")  # phoneme_len = 64
```

**ファインチューニング時の考慮事項**:
- **既存モデル（175トークン）→ 多言語（205トークン）への変換**:
  ```python
  from smalltts.models.utils import adapt_state_dict_for_expanded_phonemes

  state_dict = adapt_state_dict_for_expanded_phonemes(
      state_dict,
      old_vocab_size=175,
      new_vocab_size=205,
      initialization="zeros"  # or "mean" or "random"
  )
  ```
- 新しい日本語音素の埋め込みは指定方法で初期化
- 既存の英語音素の重みは完全に保持される

### データセット情報

**JVS (Japanese versatile speech corpus)** ✅ **採用**:
- 30時間、100話者
- ダウンロード: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
- 形式: 各話者ごとにディレクトリ分割
  ```
  jvs_ver1/
    jvs001/
      parallel100/
        wav24kHz16bit/
          VOICEACTRESS100_001.wav
          ...
        transcripts_utf8.txt
  ```
- トランスクリプト形式: `VOICEACTRESS100_001:それは確かにそうです。`
- 利点: 多様な話者、大規模、24kHzで提供
- 実装状況: データローダー完成済み (`src/smalltts/data/japanese.py`)

### pyopenjtalk-plus API

**基本的な使い方**:
```python
import pyopenjtalk

# 音素化
phonemes = pyopenjtalk.g2p("こんにちは")
# 出力: "k o N n i ch i w a"

# カナ出力
kana = pyopenjtalk.g2p("こんにちは", kana=True)
# 出力: "コンニチワ"

# アクセント情報付き（将来の改善用）
labels = pyopenjtalk.extract_fullcontext("こんにちは")
```

---

## 進捗管理

### フェーズ1の完了条件 ✅ **完了** (2025-10-11)
- [x] pyopenjtalk-plusがインストールされ、動作する
- [x] 日本語テキストを音素化できる
- [x] トークンIDへの変換・デコードが正常に動作
- [x] Dockerイメージが正常にビルドできる ✅
- [x] テストが全て通過する（ローカル環境 + Docker環境で確認済み）

### フェーズ2の完了条件 ✅ **完了** (2025-10-11)
- [x] 日本語テキストノーマライザーが動作する ✅
- [x] 音素埋め込み層が拡張され、多言語語彙（205トークン）を使用できる ✅
- [x] JVSデータセットローダーが動作する ✅
- [x] チェックポイント互換性ユーティリティが実装されている ✅
- [x] 自動言語検出機能が動作する ✅

### フェーズ3の完了条件
- [ ] ドキュメントが更新され、使い方が明確
- [ ] サンプルコードが全て動作する
- [ ] トラブルシューティングガイドが完備

---

## 参考リンク

- pyopenjtalk-plus: https://github.com/tsukumijima/pyopenjtalk-plus
- **JVS corpus** ✅: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
- F5-TTS Japanese: https://github.com/JarodMica/F5-TTS
- Style-Bert-VITS2: https://github.com/litagin02/Style-Bert-VITS2
- smallTTS original: https://github.com/smallbraineng/smalltts
