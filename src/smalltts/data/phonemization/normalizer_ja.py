"""Japanese text normalizer for TTS preprocessing"""

import re
import unicodedata
from typing import Optional


class JapaneseTextNormalizer:
    """日本語テキストの正規化

    主な機能：
    - 全角ASCII文字を半角に変換
    - 記号の正規化
    - 数字の日本語読みへの変換（オプション）
    - 英数字の処理
    """

    def __init__(self, convert_numbers: bool = False):
        """
        Args:
            convert_numbers: 数字を日本語読みに変換するかどうか
        """
        self.convert_numbers = convert_numbers

        # 数字の読み方マッピング（単独の数字用）
        self._digit_map = {
            "0": "ぜろ", "1": "いち", "2": "に", "3": "さん", "4": "よん",
            "5": "ご", "6": "ろく", "7": "なな", "8": "はち", "9": "きゅう"
        }

        # 記号の正規化マッピング（英語記号→日本語記号）
        self._symbol_map = {
            "!": "！",
            "?": "？",
            ",": "、",
            ".": "。",
            ":": "：",
            ";": "；",
            "(": "（",
            ")": "）",
            "[": "［",
            "]": "］",
            "{": "｛",
            "}": "｝",
        }

    def normalize(self, text: str) -> str:
        """テキストの正規化を実行

        Args:
            text: 正規化するテキスト

        Returns:
            正規化されたテキスト
        """
        if not text:
            return text

        # 全角ASCII文字を半角に
        text = self._normalize_ascii(text)

        # 数字を日本語読みに変換（オプション）
        if self.convert_numbers:
            text = self._normalize_numbers(text)

        # 記号の正規化（日本語文脈では日本語記号を使用）
        text = self._normalize_symbols(text)

        # 連続する空白を1つに
        text = re.sub(r'\s+', ' ', text)

        # 先頭・末尾の空白を削除
        text = text.strip()

        return text

    def _normalize_ascii(self, text: str) -> str:
        """全角ASCII文字を半角に変換

        Args:
            text: 変換するテキスト

        Returns:
            変換されたテキスト
        """
        # NFKCで正規化（全角→半角など）
        text = unicodedata.normalize('NFKC', text)
        return text

    def _normalize_symbols(self, text: str) -> str:
        """記号の正規化

        Args:
            text: 正規化するテキスト

        Returns:
            正規化されたテキスト
        """
        for en_symbol, ja_symbol in self._symbol_map.items():
            text = text.replace(en_symbol, ja_symbol)
        return text

    def _normalize_numbers(self, text: str) -> str:
        """数字を日本語読みに変換

        Args:
            text: 変換するテキスト

        Returns:
            変換されたテキスト
        """
        # 単独の数字（0-9）を日本語読みに変換
        def replace_digit(match):
            digit = match.group(0)
            return self._digit_map.get(digit, digit)

        # 1桁の数字のみを変換（複数桁の数字は複雑なので現在は対象外）
        text = re.sub(r'(?<!\d)(\d)(?!\d)', replace_digit, text)

        return text

    def _number_to_japanese(self, num_str: str) -> str:
        """数値文字列を日本語読みに変換（将来の拡張用）

        Args:
            num_str: 数値文字列

        Returns:
            日本語読み
        """
        # TODO: 複数桁の数字の読み方を実装
        # 例: "123" -> "ひゃくにじゅうさん"
        # 現在は単純な実装のみ
        return num_str


if __name__ == "__main__":
    # テストコード
    print("=" * 80)
    print("JAPANESE TEXT NORMALIZER TEST")
    print("=" * 80)

    normalizer = JapaneseTextNormalizer(convert_numbers=False)
    normalizer_with_numbers = JapaneseTextNormalizer(convert_numbers=True)

    test_cases = [
        ("こんにちは、世界！", "全角記号のテスト"),
        ("Hello, World!", "英語記号→日本語記号"),
        ("これは　　　テストです", "連続空白の正規化"),
        ("１２３４５", "全角数字→半角数字"),
        ("ＡＢＣａｂｃ", "全角アルファベット→半角"),
        ("今日は1月1日です", "数字混在（変換なし）"),
        ("今日は1月1日です", "数字混在（変換あり）"),
    ]

    print("\n[数字変換なし]")
    print("-" * 80)
    for text, description in test_cases[:-1]:
        normalized = normalizer.normalize(text)
        print(f"Input : {text}")
        print(f"Output: {normalized}")
        print(f"Note  : {description}")
        print()

    print("\n[数字変換あり]")
    print("-" * 80)
    text, description = test_cases[-1]
    normalized = normalizer_with_numbers.normalize(text)
    print(f"Input : {text}")
    print(f"Output: {normalized}")
    print(f"Note  : {description}")
    print()

    print("=" * 80)
    print("[SUCCESS] Japanese text normalizer test completed!")
    print("=" * 80)
