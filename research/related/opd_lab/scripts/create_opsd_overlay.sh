#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OPDLAB_BASE_PYTHON="${1:?Usage: bash scripts/create_opsd_overlay.sh /absolute/path/to/OPD/python}"
if [[ -e .venv-opsd ]]; then
  echo '.venv-opsd already exists; inspect it before creating another environment.' >&2
  exit 1
fi
"$OPDLAB_BASE_PYTHON" -m venv --system-site-packages .venv-opsd
.venv-opsd/bin/python -m pip install --no-deps \
  'diffusers==0.36.0' 'transformers==4.51.3' 'huggingface-hub==0.34.4' \
  'tokenizers==0.21.4' 'peft==0.17.1' 'timm==1.0.19'
.venv-opsd/bin/python -c 'import diffusers, transformers, peft; from timm.data import ImageNetInfo; print(diffusers.__version__, transformers.__version__, peft.__version__)'
