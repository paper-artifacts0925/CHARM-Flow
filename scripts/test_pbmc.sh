#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${CHARM_TRAIN_PYTHON:-python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUTPUT_DIR:-$ROOT/runs/test/pbmc_$STAMP}"
[[ ! -e "$OUT" ]] || { echo "refusing existing output: $OUT" >&2; exit 4; }
if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
  "$PY" "$ROOT/scripts/verify_package.py" --fast >/dev/null
  "$PY" -c 'import sys; from pathlib import Path; [compile(Path(p).read_text(), p, "exec") for p in sys.argv[1:]]' \
    "$ROOT/scripts/pbmc/run_full.py" "$ROOT/scripts/pbmc/sample_shard.py" \
    "$ROOT/scripts/pbmc/sample_model.py" "$ROOT/scripts/pbmc/shard_runtime.py" \
    "$ROOT/scripts/pbmc/shard_model.py" "$ROOT/scripts/pbmc/shard_sampler_core.py" \
    "$ROOT/scripts/pbmc/evaluate_by_celltype.py"
  echo "CONFIG_ONLY_PASS pbmc"; exit 0
fi
"$PY" "$ROOT/scripts/pbmc/run_full.py" --run-root "$OUT" \
  --gpu-ids "${GPU_IDS:-0,1,2,3,4,5}" --eval-threads "${EVAL_THREADS:-32}"
"$PY" "$ROOT/scripts/extract_pbmc_core.py" \
  --completion "$OUT/pooled_celleval_exact_v1/completion.json" --outdir "$OUT/metrics" \
  | tee "$OUT/core_metrics.log"
echo "COMPLETE output=$OUT metrics=$OUT/metrics/core_metrics.json"
