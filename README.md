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

## 実行スクリプト
`docker compose` の起動・ヘルスチェック待機・評価実行をまとめて行うスクリプトです。

```bash
# 初回やクリーン起動
./scripts/run_eval.sh

# 既存コンテナを使って高速実行
./scripts/run_eval.sh --reuse

# シナリオ切り替え
./scripts/run_eval.sh --reuse -c examples/scenario.yaml
```

## Makefile
よく使う操作は `make` で実行できます。

```bash
# ターゲット一覧
make help

# フル実行（コンテナ再起動あり）
make run

# 高速実行（既存コンテナ再利用）
make run-reuse

# lint / format / test
make lint
make format
make test
make check
```

## uv セットアップ例
```bash
uv sync
```

## テスト
```bash
uv run pytest
```

## Lint / Format (Ruff)
```bash
# lint
uv run ruff check .

# lintを自動修正
uv run ruff check . --fix

# format
uv run ruff format .
```

## Natural Questions (NQ) 1000件データセット
NQ 由来の 1000 クエリ評価セットを生成・実行できます。

### データ準備
NQ の元データ（Google 提供 JSONL）が必要です。

```bash
# ビルド + バリデーション
make nq-prepare NQ_INPUT_PATH=/path/to/nq-train.jsonl

# 個別実行
make nq-build NQ_INPUT_PATH=/path/to/nq-train.jsonl
make nq-validate
```

オプション:
- `NQ_OUT_DIR`: 出力先（default: `examples/data/nq_1000`）
- `--num-queries`: 抽出数（default: `1000`）
- `--seed`: ランダムシード（default: `42`）
- `--max-passages-per-page`: ページあたり最大パッセージ数（default: `20`）

### 評価実行
```bash
# シナリオ実行
uv run -m musubi_eval.cli run -c examples/scenario_nq_1000.yaml

# パラメータチューニング
uv run -m musubi_eval.cli tune -c examples/tuning_nq_1000.yaml
```

### データバリデーション
```bash
uv run python scripts/validate_dataset.py --dataset-dir examples/data/nq_1000
```

## Optuna パラメータチューニング
Optuna を使って `k`, `ef`, `alpha` の最適な組み合わせを自動探索できます。

### tuning.yaml の例
```yaml
base_scenario: "examples/scenario.yaml"

study:
  name: "musubi-tuning"
  direction: "maximize"
  n_trials: 20
  timeout_sec: 600
  sampler_seed: 42

search_space:
  k:
    low: 5
    high: 50
    step: 5
  ef:
    low: 50
    high: 300
    step: 50
  alpha:
    low: 0.0
    high: 1.0
    step: 0.1

constraints:
  max_latency_p95_ms: 500.0

objective:
  metric: "recall_at_k"
  latency_penalty: 0.0001

output:
  dir: "outputs"
  save_history_csv: true
  save_history_json: true
  save_best_yaml: true

mlflow:
  enabled: true
  experiment_name: "musubi-tuning"
  run_name_prefix: "musubi-tuning"
```

### 実行
```bash
# 基本実行
make tune

# 設定ファイルを指定
make tune TUNING_CONFIG=examples/tuning.yaml

# uv で直接実行
uv run -m musubi_eval.cli tune -c examples/tuning.yaml
```

### 出力
- `outputs/tuning_<timestamp>.json` — 全 trial の履歴
- `outputs/tuning_<timestamp>.csv` — 履歴 CSV
- `outputs/best_params_<timestamp>.yaml` — 推奨パラメータ
- `outputs/best_scenario_<timestamp>.yaml` — best params を反映したシナリオ YAML

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
