#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo $ROOT
GSET_DIR="${ROOT}/data/GSet"

mkdir -p "${GSET_DIR}"

echo "=== Downloading GSet instances ==="

GSET_BASE="https://web.stanford.edu/~yyye/yyye/Gset"

for instance in G38 G55 G60 G70; do
    echo "Downloading ${instance}..."
    curl -fL --retry 3 \
        "${GSET_BASE}/${instance}" \
        -o "${GSET_DIR}/${instance}"
done

