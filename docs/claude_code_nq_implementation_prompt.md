# Claude Code向け実装指示（Natural Questions 1000件）

以下をそのまま実装してください。  
目的は、`musubi-eval` に Natural Questions（NQ）由来の `1000` クエリ評価セットを追加し、既存の `run/tune` フローで実行可能にすることです。

## ゴール

- `examples/data/nq_1000/` に `documents.jsonl` / `queries.jsonl` を生成できる。
- 生成データが `musubi_eval` の既存フォーマットに完全準拠している。
- `examples/scenario_nq_1000.yaml` と `examples/tuning_nq_1000.yaml` でそのまま実行できる。
- 生成物の妥当性を検証するスクリプトとテストがある。

## 実装タスク

1. `scripts/build_nq_subset.py` を追加
- 役割:
  - NQデータソースから `num_queries=1000` を抽出し、評価形式へ変換して保存。
- CLI:
  - `--num-queries`（default `1000`）
  - `--seed`（default `42`）
  - `--out-dir`（default `examples/data/nq_1000`）
  - `--input-path`（NQ元データのパス）
  - `--max-passages-per-page`（default `20`）
- 出力:
  - `<out-dir>/documents.jsonl`
  - `<out-dir>/queries.jsonl`
- JSONLスキーマ:
  - documents:
    - `id: str`
    - `text: str`
    - `metadata: dict`（`source`, `page_id`, `title` など）
  - queries:
    - `id: str`
    - `query: str`
    - `positive_ids: List[str]`
    - `filter: dict` は任意（初期は省略可）

2. サンプリング仕様を実装
- `seed` 固定で再現可能にする。
- 明らかな不正行を除外:
  - 質問文なし
  - 正例 passage を作れない
- バランス改善（軽量でよい）:
  - 同一 page_id の過剰集中を避ける（`--max-passages-per-page`）

3. `scripts/validate_dataset.py` を追加
- CLI:
  - `--dataset-dir`（`documents.jsonl` と `queries.jsonl` を読む）
- 検証内容:
  - `positive_ids` が documents 側の `id` に存在
  - 必須キー欠損チェック
  - 重複 `id` チェック
  - 件数と基本統計（クエリ長、positive_ids数）を標準出力
- 不正時は非ゼロ終了コードにする。

4. サンプル設定を追加
- `examples/scenario_nq_1000.yaml`
  - `datasets` を `examples/data/nq_1000/*.jsonl` に向ける
  - その他は `examples/scenario.yaml` を踏襲
- `examples/tuning_nq_1000.yaml`
  - `base_scenario: examples/scenario_nq_1000.yaml`
  - その他は既存 `examples/tuning.yaml` を踏襲

5. Makefile ターゲット追加
- `nq-build`:
  - `uv run python scripts/build_nq_subset.py ...`
- `nq-validate`:
  - `uv run python scripts/validate_dataset.py --dataset-dir examples/data/nq_1000`
- `nq-prepare`:
  - `nq-build` + `nq-validate`

6. README 更新
- NQ 1000件の準備手順を追加:
  - `make nq-prepare`
  - `uv run -m musubi_eval.cli run -c examples/scenario_nq_1000.yaml`
  - `uv run -m musubi_eval.cli tune -c examples/tuning_nq_1000.yaml`

7. pytest テスト追加
- `tests/test_nq_build.py`
  - 小さなダミーNQ入力から `documents/queries` が正しく生成される
  - seed固定で再現される
- `tests/test_validate_dataset.py`
  - 正常系（pass）
  - 異常系（missing keys / broken positive_ids / duplicate ids）で fail

## 非機能要件

- 既存コードスタイルに合わせる（Ruff / pytest）。
- 可能な限り関数を小さく分離し、型注釈を付与。
- 既存機能を壊さない（`tests` 全体が通ること）。

## 受け入れ基準

- 以下が通ること:
  - `uv run ruff check .`
  - `uv run pytest -q`
- 以下が成功すること:
  - `make nq-prepare`
  - `uv run -m musubi_eval.cli run -c examples/scenario_nq_1000.yaml`

## 実装後に提出してほしい内容

- 変更ファイル一覧
- 実行したコマンドと結果要約
- 既知の制約（NQ入力フォーマット依存点など）
