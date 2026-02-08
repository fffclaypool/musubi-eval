#!/usr/bin/env bash
set -euo pipefail

REUSE_MODE=0
SCENARIO_PATH="examples/scenario.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reuse)
      REUSE_MODE=1
      shift
      ;;
    -c|--config)
      SCENARIO_PATH="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--reuse] [-c|--config <scenario.yaml>]" >&2
      exit 1
      ;;
  esac
done

wait_healthy() {
  local service="$1"
  local waited=0
  local timeout=600

  echo "Waiting for ${service} to become healthy..."
  until docker compose ps "${service}" | grep -q "healthy"; do
    sleep 2
    waited=$((waited + 2))
    if [[ "${waited}" -ge "${timeout}" ]]; then
      echo "Timed out waiting for ${service} to become healthy." >&2
      docker compose ps -a >&2
      docker compose logs --tail=120 "${service}" >&2 || true
      exit 1
    fi
  done
}

if [[ "${REUSE_MODE}" -eq 1 ]]; then
  docker compose up -d
else
  docker compose down
  docker compose up -d
fi

wait_healthy musubi-embed
wait_healthy musubi-api

docker compose ps -a
uv run -m musubi_eval.cli run -c "${SCENARIO_PATH}"
