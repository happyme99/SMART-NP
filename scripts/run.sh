#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT=naturalRAG
cd "$ROOT"
exec uv run naturalrag run "$@"
