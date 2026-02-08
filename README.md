# musubi-eval

Rust側の `musubi` を評価するための Python 評価基盤（ローカル実行）。

## 目的
- E2E 評価（text -> search results）
- パラメータ探索（k/ef/alpha/filter）
- 指標: Recall@k / MRR / nDCG / latency(p50/p95)

## ディレクトリ構成
- `musubi_eval/domain/`: ドメインモデルと指標計算
- `musubi_eval/application/`: ユースケースとポート定義
- `musubi_eval/infrastructure/`: HTTP/ファイル/外部ツールのアダプタ
- `musubi_eval/cli.py`: CLI エントリ
- `musubi_eval/dataset.py`: JSONL データ読み込み（インフラ）
- `musubi_eval/reporting.py`: Evidently/MLflow レポート（インフラ）
- `examples/`: 例シナリオとサンプルデータ
- `outputs/`: 結果出力先
- `tests/`: テスト

## シナリオ YAML スキーマ（MVP）
```yaml
base_url: "http://localhost:8080"
log_level: "INFO"
timeout_sec: 30

retry:
  max_attempts: 3
  base_backoff_sec: 0.5
  max_backoff_sec: 5.0

ingestion:
  poll_interval_sec: 1.0
  timeout_sec: 600

datasets:
  documents: "path/to/documents.jsonl"
  queries: "path/to/queries.jsonl"

search:
  params:
    - name: "k10_default"
      k: 10
      ef: 100
      alpha: 0.5
      filter: {"category": "tech"}

output:
  dir: "outputs"
  save_json: true
  save_csv: true
  save_prefix: "run"
```

### データ形式
- `documents.jsonl`
  - 必須: `id`, `text`
  - 任意: `metadata`
- `queries.jsonl`
  - 必須: `id`, `query`, `positive_ids`
  - 任意: `filter`

## セットアップ
### 前提
- Python 3.10 以上
- `uv`（推奨。ない場合は `pip` でも可）
- `musubi` API が `http://localhost:8080` で起動していること

### 依存関係のインストール
```bash
uv sync
```

### musubi の起動（Docker Compose 例）
```bash
docker compose up
```

## 最小実行コマンド（uv）
```bash
uv sync
uv run -m musubi_eval.cli run -c examples/scenario.yaml
```

## uv セットアップ例
```bash
uv sync
```

## テスト
```bash
uv run pytest
```

## ダッシュボード（Evidently / MLflow）
### 依存関係
`evidently` と `mlflow` を使う場合は追加依存が必要です。

```bash
uv sync
```

### Evidently レポート
`examples/scenario.yaml` の `evidently.enabled: true` で、実行のたびに
`outputs/evidently/` に HTML/JSON レポートを出力します。

HTMLの確認方法:
```bash
# 例: 生成されたHTMLをブラウザで開く
outputs/evidently/evidently_*.html
```

### MLflow
`examples/scenario.yaml` の `mlflow.enabled: true` で、実行のたびに
`mlruns/` に実験結果を記録します（ローカルのファイルベース）。

ローカルUI:
```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

`bash: mlflow: command not found` が出る場合も、`uv run mlflow ...` なら起動できます。

## MVP 実装ステップ（小さいPR単位）
1. 最小骨格: `client/dataset/metrics/pipeline/cli` と YAML 読み込み
2. リトライ/タイムアウト/ログ
3. 指標集計と JSON/CSV 出力
4. 検索パラメータ複数実行
5. 追加指標/可視化の拡張
