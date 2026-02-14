SHELL := /bin/bash

SCENARIO ?= examples/scenario.yaml
TUNING_CONFIG ?= examples/tuning.yaml
NQ_INPUT_PATH ?= /path/to/nq-train.jsonl
NQ_OUT_DIR ?= examples/data/nq_1000
MLFLOW_BACKEND ?= ./mlruns
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  sync            - Install/update dependencies with uv"
	@echo "  compose-up      - Start docker compose services"
	@echo "  compose-down    - Stop docker compose services"
	@echo "  run             - Run full eval flow (restart containers)"
	@echo "  run-reuse       - Run eval flow without restarting containers"
	@echo "  run-scenario    - Run evaluator only (expects services up)"
	@echo "  tune            - Run Optuna parameter tuning"
	@echo "  nq-build        - Build NQ 1000 dataset from source"
	@echo "  nq-validate     - Validate NQ 1000 dataset"
	@echo "  nq-prepare      - Build + validate NQ 1000 dataset"
	@echo "  test            - Run pytest"
	@echo "  lint            - Run ruff check"
	@echo "  lint-fix        - Run ruff check with auto-fix"
	@echo "  format          - Run ruff format"
	@echo "  format-check    - Check ruff formatting"
	@echo "  check           - Run lint + format-check + test"
	@echo "  mlflow-ui       - Start MLflow UI"
	@echo ""
	@echo "Examples:"
	@echo "  make run"
	@echo "  make run-reuse SCENARIO=examples/scenario.yaml"
	@echo "  make tune TUNING_CONFIG=examples/tuning.yaml"
	@echo "  make nq-prepare NQ_INPUT_PATH=/path/to/nq-train.jsonl"
	@echo "  make mlflow-ui MLFLOW_PORT=5001"

.PHONY: sync
sync:
	uv sync

.PHONY: compose-up
compose-up:
	docker compose up -d

.PHONY: compose-down
compose-down:
	docker compose down

.PHONY: run
run:
	./scripts/run_eval.sh -c "$(SCENARIO)"

.PHONY: run-reuse
run-reuse:
	./scripts/run_eval.sh --reuse -c "$(SCENARIO)"

.PHONY: run-scenario
run-scenario:
	uv run -m musubi_eval.cli run -c "$(SCENARIO)"

.PHONY: tune
tune:
	uv run -m musubi_eval.cli tune -c "$(TUNING_CONFIG)"

.PHONY: nq-build
nq-build:
	uv run python scripts/build_nq_subset.py --input-path "$(NQ_INPUT_PATH)" --out-dir "$(NQ_OUT_DIR)"

.PHONY: nq-validate
nq-validate:
	uv run python scripts/validate_dataset.py --dataset-dir "$(NQ_OUT_DIR)"

.PHONY: nq-prepare
nq-prepare: nq-build nq-validate

.PHONY: test
test:
	uv run pytest

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: lint-fix
lint-fix:
	uv run ruff check . --fix

.PHONY: format
format:
	uv run ruff format .

.PHONY: format-check
format-check:
	uv run ruff format --check .

.PHONY: check
check: lint format-check test

.PHONY: mlflow-ui
mlflow-ui:
	uv run mlflow ui --backend-store-uri "$(MLFLOW_BACKEND)" --host "$(MLFLOW_HOST)" --port "$(MLFLOW_PORT)"
