# 日本語音素化対応タスクリスト

## 概要

smallTTSに日本語対応を追加するため、pyopenjtalk-plusを使用した音素化システムを実装します。
現在のespeak-ng（英語専用）から、日本語も扱えるマルチリンガルシステムへ拡張します。

**目標**: 日本語データセット（JSUT/JVS）でファインチューニングできる環境を構築

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
**ステータス**: ⬜ 未着手

**新規ファイル**:
- `src/smalltts/data/phonemization/normalizer_ja.py`

**実装内容**:

```python
import re
from typing import Optional

class JapaneseTextNormalizer:
    """日本語テキストの正規化"""

    def __init__(self):
        # 数字の読み方マッピング
        self._digit_map = {
            "0": "ぜろ", "1": "いち", "2": "に", "3": "さん", "4": "よん",
            "5": "ご", "6": "ろく", "7": "なな", "8": "はち", "9": "きゅう"
        }

        # 記号の処理
        self._symbol_map = {
            "!": "！",
            "?": "？",
            ",": "、",
            ".": "。",
        }

    def normalize(self, text: str) -> str:
        """テキストの正規化"""
        # 全角英数字を半角に
        text = self._normalize_ascii(text)

        # 数字を日本語読みに変換（オプション）
        # text = self._normalize_numbers(text)

        # 記号の正規化
        text = self._normalize_symbols(text)

        return text

    def _normalize_ascii(self, text: str) -> str:
        """全角ASCII文字を半角に変換"""
        # 実装
        return text

    def _normalize_symbols(self, text: str) -> str:
        """記号の正規化"""
        for en, ja in self._symbol_map.items():
            text = text.replace(en, ja)
        return text
```

**チェックリスト**:
- [ ] JapaneseTextNormalizerクラスの実装
- [ ] 数字の読み方変換（オプション）
- [ ] 記号の処理
- [ ] 全角・半角の正規化
- [ ] ユニットテストの作成

---

### タスク5: 音素埋め込み層のサイズ調整
**ステータス**: ⬜ 未着手

**変更ファイル**:
- `src/smalltts/models/backbone/phonemes.py`
- `src/smalltts/models/backbone/model.py`

**実装内容**:

**オプション1: 多言語対応（推奨）**
- 英語音素（約180）+ 日本語音素（約70）= 約250トークン
- 既存モデルの埋め込みを拡張
- ファインチューニング時に新しい日本語音素のみを学習

```python
def expand_phoneme_embeddings(model, old_vocab_size, new_vocab_size):
    """音素埋め込み層を拡張"""
    old_embeddings = model.phoneme_embedding.phoneme_embed.weight.data
    new_embeddings = nn.Embedding(new_vocab_size, old_embeddings.shape[1])

    # 既存の英語音素をコピー
    new_embeddings.weight.data[:old_vocab_size] = old_embeddings

    # 新しい日本語音素はランダム初期化（小さい値）
    nn.init.normal_(new_embeddings.weight.data[old_vocab_size:], std=0.02)

    model.phoneme_embedding.phoneme_embed = new_embeddings
    return model
```

**オプション2: 日本語のみ**
- 日本語音素のみ（約70トークン）
- 新規訓練が必要
- モデルサイズは小さくなる

**チェックリスト**:
- [ ] 拡張戦略の選択（オプション1 or 2）
- [ ] PhonemeEmbeddingクラスの更新
- [ ] 埋め込み層拡張の実装
- [ ] 既存チェックポイントからのロード処理
- [ ] テストコードの作成

---

### タスク6: データローダーの作成
**ステータス**: ⬜ 未着手

**新規ファイル**:
- `src/smalltts/data/japanese_dataset.py`

**実装内容**:

```python
import torch
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import soundfile as sf

class JapaneseDataset:
    """日本語TTSデータセット（JSUT/JVS形式）"""

    def __init__(self, data_dir: str, encoder):
        self.data_dir = Path(data_dir)
        self.encoder = encoder  # VibeVoice encoder

        # メタデータの読み込み
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """メタデータ（音声ファイルパスとテキスト）を読み込み"""
        # JSUT形式: transcript_utf8.txt
        # JVS形式: 各話者ディレクトリ内
        pass

    def __getitem__(self, idx):
        audio_path, text = self.metadata[idx]

        # 音声を読み込み
        audio, sr = sf.read(audio_path)

        # 24kHzにリサンプリング（必要に応じて）
        if sr != 24000:
            # リサンプリング処理
            pass

        # エンコーダーで潜在表現に変換
        latents = self.encoder.encode(audio)

        # テキストを音素化
        from smalltts.data.phonemization.phonemes import get_token_ids
        phonemes = get_token_ids(text)

        return {
            "text": text,
            "phonemes": torch.tensor(phonemes),
            "latents": latents,
        }

    def __len__(self):
        return len(self.metadata)

def collate_fn(batch):
    """バッチ処理用のcollate関数"""
    texts = [item["text"] for item in batch]
    phonemes = [item["phonemes"] for item in batch]
    latents = [item["latents"] for item in batch]

    # パディング
    phonemes_padded = pad_sequence(phonemes, batch_first=True, padding_value=0)
    latents_padded = pad_sequence(latents, batch_first=True, padding_value=0.0)

    # 長さ情報
    phonemes_lengths = torch.tensor([len(p) for p in phonemes])
    latents_lengths = torch.tensor([len(l) for l in latents])

    return {
        "texts": texts,
        "phonemes": phonemes_padded,
        "phonemes_lengths": phonemes_lengths,
        "latents": latents_padded,
        "latents_lengths": latents_lengths,
    }
```

**チェックリスト**:
- [ ] JapaneseDatasetクラスの実装
- [ ] JSUT形式のサポート
- [ ] JVS形式のサポート
- [ ] collate_fn関数の実装
- [ ] データローダーのテスト

---

## フェーズ3: 統合とドキュメント（優先度: 中）

### タスク7: ドキュメント更新
**ステータス**: ⬜ 未着手

**変更ファイル**:
- `CLAUDE.md`
- `README.md`
- `docs/training_japanese.md`（新規）

**実装内容**:

1. **CLAUDE.md**に日本語対応の情報を追加:
   - 音素化システムの説明
   - 言語切り替え方法
   - データフォーマット

2. **README.md**に使用例を追加:
   ```python
   # 日本語で推論
   from smalltts import SmallTTS
   from smalltts.data.phonemization import phonemes

   # 言語を日本語に設定
   phonemes.LANGUAGE = "ja"

   tts = SmallTTS()
   # ... 以下同様
   ```

3. **docs/training_japanese.md**（新規作成）:
   - 日本語データセットの準備方法
   - ファインチューニングの手順
   - ハイパーパラメータの推奨値
   - トラブルシューティング

**チェックリスト**:
- [ ] CLAUDE.mdの更新
- [ ] README.mdに使用例を追加
- [ ] training_japanese.mdの作成
- [ ] サンプルコードの動作確認

---

## 技術的メモ

### 音素語彙の互換性

**現在の構造**:
- 英語音素: 約180トークン
- phoneme_len = 181（0はpadding）

**日本語追加後**:
- オプション1（多言語）: 約250トークン
- オプション2（日本語のみ）: 約70トークン

**ファインチューニング時の考慮事項**:
- モデルのphoneme_embeddingレイヤーを拡張する場合、既存の重みは保持
- 新しい日本語音素の埋め込みは小さい値でランダム初期化
- 学習率を既存部分よりも新規部分で高く設定することを検討

### データセット情報

**JSUT（推奨・初期実装）**:
- 10時間、単一話者（女性）
- ダウンロード: https://sites.google.com/site/shinnosuketakamichi/publication/jsut
- 形式: WAV + transcript_utf8.txt
- 利点: シンプル、扱いやすい

**JVS（本格実装）**:
- 30時間、100話者
- ダウンロード: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
- 形式: 各話者ごとにディレクトリ分割
- 利点: 多様な話者、大規模

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

### フェーズ2の完了条件
- [ ] 日本語テキストノーマライザーが動作する
- [ ] 音素埋め込み層が拡張され、既存モデルをロードできる
- [ ] 日本語データセットローダーが動作する
- [ ] サンプルデータで訓練が開始できる

### フェーズ3の完了条件
- [ ] ドキュメントが更新され、使い方が明確
- [ ] サンプルコードが全て動作する
- [ ] トラブルシューティングガイドが完備

---

## 参考リンク

- pyopenjtalk-plus: https://github.com/tsukumijima/pyopenjtalk-plus
- JSUT dataset: https://sites.google.com/site/shinnosuketakamichi/publication/jsut
- JVS corpus: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
- F5-TTS Japanese: https://github.com/JarodMica/F5-TTS
- Style-Bert-VITS2: https://github.com/litagin02/Style-Bert-VITS2
