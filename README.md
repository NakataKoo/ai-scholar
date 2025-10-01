# AI-SCHOLAR

AI研究論文の自動収集・要約生成システム

## 概要

AI-SCHOLARは、AI研究論文を自動的に収集し、日本語で分かりやすく要約するPythonベースのシステムです。Hugging Faceの最新論文を収集し、GPT-4やマルチモーダルAIモデルを活用して、研究者や技術者が効率的に論文の内容を把握できるような解説記事を生成します。

## 主な機能

- **自動論文収集**: Hugging Face APIから最新のAI論文を自動取得
- **Google Sheets連携**: 論文メタデータの管理と処理状況の追跡
- **マルチモーダル論文解析**: PDF全体の画像解析による包括的な理解
- **日本語要約生成**: 詳細解説と3点要約の自動生成
- **Jupyter Notebook対応**: Google Colabでのインタラクティブな処理

## システム構成

### 1. 論文収集システム (`collect_paper.py`)
- Hugging Face APIから最新論文を取得
- Google Sheetsにメタデータを保存
- 処理状況の管理

### 2. 記事生成システム (`generate_article.py`)
- arXivからPDFダウンロード
- PDF全体を画像化してマルチモーダル解析
- Azure OpenAI/OpenAI APIを使用した日本語要約生成
- Google Sheetsへの結果更新

### 3. Jupyter Notebook (`AI_SCHOLAR記事生成ツールv2.ipynb`)
- Google Colab環境での対話的処理
- 論文収集から要約生成までの一連の処理
- リアルタイムでの結果確認

## 前提条件

### 必要なアカウント・認証情報
- **Google Cloud Platform**: Service Account with Sheets API access
- **Azure OpenAI** または **OpenAI API**: GPT-4o access
- **Google Sheets**: 管理用スプレッドシート「AI-SCHOLAR運用管理システム」

### 必要な依存関係
- Python 3.8+
- poppler-utils (PDF処理用)
- 各種Pythonパッケージ (requirements.txt参照)

## セットアップ

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd ai-scholar
```

### 2. 依存関係のインストール
```bash
pip install -r requirements.txt

# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler
```

### 3. 環境変数の設定
`.env`ファイルを作成し、以下の設定を追加：

```bash
# Google Service Account JSON file path
GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/your/service-account.json

# API Selection: 'azure' or 'openai'
SELECT_API=azure

# Azure OpenAI Configuration (SELECT_API=azure の場合)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-01-preview
AZURE_OPENAI_API_KEY=your-azure-api-key

# OpenAI Configuration (SELECT_API=openai の場合)
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o
```

### 4. Google Sheets の準備
1. Google Cloud Consoleでプロジェクトを作成
2. Google Sheets APIを有効化
3. Service Accountを作成し、JSONキーをダウンロード
4. 「AI-SCHOLAR運用管理システム」という名前のスプレッドシートを作成
5. 「LLM-Papers」ワークシートを作成
6. Service AccountにSpreadsheetの編集権限を付与

## クイックスタート

### 方法1: スクリプトを個別実行

#### 1. 論文収集
```bash
python collect_paper.py
```

このスクリプトは以下の処理を実行します：
- Hugging Face APIから当日の最新論文を取得
- 論文のタイトル、URL、日付をGoogle Sheetsに保存
- 処理済みの論文はスキップして重複を防止

#### 2. 論文解説記事の生成
```bash
python generate_article.py
```

このスクリプトは以下の処理を実行します：
- Google Sheetsから未処理の論文を取得
- arXivからPDFをダウンロード
- PDF全体を画像化してマルチモーダル解析
- 「概要」「研究手法」「まとめ」の3セクションで詳細解説を生成
- 3点要約を生成
- 結果をGoogle Sheetsに保存

### 方法2: Jupyter Notebook使用（推奨）

Google Colabで`AI_SCHOLAR記事生成ツールv2.ipynb`を開いて実行：

#### 1. Google Driveをマウント
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### 2. 論文収集セクションを実行
- Hugging Face APIから最新論文を自動取得
- Google Sheetsへの保存を確認

#### 3. 論文解説セクションを実行
- 必要なライブラリをインストール
- Azure OpenAI/OpenAI APIの設定
- 未処理論文の自動解析と要約生成

### 方法3: 一括処理
論文収集から解説生成まで一括で実行：

```bash
# 1. 論文収集
python collect_paper.py

# 2. 記事生成
python generate_article.py
```

## プロジェクト構造

```
ai-scholar/
├── README.md                           # このファイル
├── requirements.txt                   # Python依存関係
├── collect_paper.py                   # 論文収集スクリプト
├── generate_article.py               # 記事生成スクリプト
├── AI_SCHOLAR記事生成ツールv2.ipynb    # Jupyter Notebook
├── ai-scholar-article-tool-*.json    # Google Service Account credentials
└── content/                          # ダウンロードした論文保存ディレクトリ
    └── [paper-title]/
        └── [paper-id].pdf
```

## Google Sheets構造

管理用スプレッドシート「AI-SCHOLAR運用管理システム」の「LLM-Papers」ワークシートには以下の列があります：

| 列 | 項目 | 説明 |
|---|---|---|
| A | 日付 | 論文収集日（YYYYMMDD形式） |
| B | タイトル | 論文タイトル |
| C | URL | arXiv URL |
| D | ステータス | 収集ステータス |
| E | 処理日 | 解説生成日 |
| F | 詳細要約 | 生成された詳細解説 |
| G | 3点要約 | 生成された3点要約 |
| H | 処理ステータス | 解説生成ステータス（完了/エラー） |

## API統合詳細

### Azure OpenAI / OpenAI API
- **モデル**: GPT-4o
- **機能**: マルチモーダル解析（PDF画像 + テキスト）
- **レート制限**: 60 requests/minute
- **リトライ機能**: 指数バックオフで最大3回

### Google Sheets API
- **認証**: Service Account
- **スコープ**: Sheets API, Drive API
- **レート制限**: 1.1秒間隔で更新

### Hugging Face API
- **エンドポイント**: daily_papers API
- **データ**: 日次公開論文リスト
- **フォーマット**: JSON形式

## 生成される要約の形式

### 詳細要約
各論文について以下の3セクションで構成：
- **概要**: 研究の背景と目的
- **研究手法**: 提案手法の詳細説明
- **まとめ**: 結果と今後の展望

### 3点要約
論文の要点を以下の形式で要約：
- 研究で解決しようとした課題
- 提案した解決手法
- 達成した成果や可能になったこと

## トラブルシューティング

### よくある問題

**1. Azure OpenAI Content Filter エラー**
```
ResponsibleAIPolicyViolation: content_filter
```
- 解決策: プロンプトの内容を確認し、ポリシー違反となる表現を修正

**2. Google Sheets認証エラー**
```
gspread.exceptions.APIError: 403 Forbidden
```
- 解決策: Service Accountにスプレッドシートの編集権限があることを確認

**3. PDF変換エラー**
```
PDFInfoNotInstalledError
```
- 解決策: poppler-utilsがインストールされていることを確認

**4. メモリ不足エラー**
- 解決策: PDFページ数が多い場合は画像解像度を下げる

## 今後の改善予定

- ColQwen2を使用したマルチモーダルRAG機能の実装
- 図表の詳細解析機能の追加
- Webインターフェースの開発
- バッチ処理の最適化
