# CPU版Dockerfile - 軽量で高速起動
FROM python:3.10-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムパッケージの更新と必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1-dev \
    ffmpeg \
    git \
    curl \
    build-essential \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    && rm -rf /var/lib/apt/lists/*

# uvをインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# プロジェクトファイルをコピー
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

# 依存関係をインストール
RUN uv sync --no-dev

# outディレクトリを作成
RUN mkdir -p /app/out

# assetsディレクトリを作成（後でボリュームマウントまたは自動ダウンロード）
RUN mkdir -p /app/assets

# デフォルトのコマンド
CMD ["bash"]
