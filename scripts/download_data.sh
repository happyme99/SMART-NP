#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$ROOT/data/dataset"
ARCHIVE=smart-np-dataset-v1.tar.gz

if ! command -v gh >/dev/null 2>&1; then
    echo "Install GitHub CLI first: https://cli.github.com/" >&2
    exit 1
fi
if find "$DATASET" -mindepth 1 -maxdepth 1 ! -name .gitkeep -print -quit | grep -q .; then
    echo "Data already exists in $DATASET; refusing to overwrite it" >&2
    exit 1
fi

gh auth status --hostname github.com >/dev/null 2>&1 || gh auth login --hostname github.com --web
DOWNLOAD_DIR=$(mktemp -d "$DATASET/.download.XXXXXX")
trap 'rm -rf -- "$DOWNLOAD_DIR"' EXIT
ASSETS=$(gh api repos/happyme99/SMART-NP/releases/tags/data-v1 --jq \
    '.assets[] | select(.name == "smart-np-dataset-v1.tar.gz" or .name == "smart-np-dataset-v1.tar.gz.sha256") | [.id, .size, .name] | @tsv')
while IFS=$'\t' read -r asset_id asset_size asset_name; do
    gh api "repos/happyme99/SMART-NP/releases/assets/$asset_id" -H 'Accept: application/octet-stream' |
        uv run python -m tqdm --bytes --total "$asset_size" --buf-size 1048576 --desc "$asset_name" > "$DOWNLOAD_DIR/$asset_name"
done <<< "$ASSETS"
printf 'Verifying SHA-256 checksum...\n'
(cd "$DOWNLOAD_DIR" && sha256sum -c "${ARCHIVE}.sha256")
uv run python -m tqdm --bytes --total "$(wc -c < "$DOWNLOAD_DIR/$ARCHIVE")" --buf-size 1048576 --desc 'Extracting dataset' \
    < "$DOWNLOAD_DIR/$ARCHIVE" | tar -xz -C "$ROOT/data"
printf 'Dataset ready: %s\n' "$DATASET"
