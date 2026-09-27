#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PY="${CHARM_TRAIN_PYTHON:-python}"
EVAL_PY="${CHARM_EVAL_PYTHON:-python}"
GPU="${GPU:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUTPUT_DIR:-$ROOT/runs/test/replogle_$STAMP}"
WEIGHT="$ROOT/weights/replogle.pt"
EXPECTED=f681d8671751456fb054129bbf150fa723ec17faffe7c4f885fa0236feb79e22
if [[ "${CONFIG_ONLY:-0}" != 1 ]]; then
  [[ "$(sha256sum "$WEIGHT" | awk '{print $1}')" == "$EXPECTED" ]] || { echo 'replogle weight hash mismatch' >&2; exit 3; }
fi
[[ ! -e "$OUT" ]] || { echo "refusing existing output: $OUT" >&2; exit 4; }
mkdir -p "$OUT/samples" "$OUT/celleval_input" "$OUT/metrics" "$OUT/cache" "$OUT/logs"
RESOURCE_ROOT="${CHARM_RESOURCE_ROOT:-$ROOT/resources}"
export CELL_DETR_DATA_ROOT="${CHARM_DATA_ROOT:-$RESOURCE_ROOT/data/PerturbDiff_data/perturb_data}"
export CELL_DETR_ARTIFACT_ROOT="${CHARM_ARTIFACT_ROOT:-$RESOURCE_ROOT/artifacts}"
export CELL_DETR_OUTPUT_ROOT="${CELL_DETR_OUTPUT_ROOT:-$ROOT/runs}"
export CELL_DETR_CACHE_ROOT="${CELL_DETR_CACHE_ROOT:-$ROOT/runs/cache}"
export CELL_DETR_CHECKPOINT_SHA256="$EXPECTED"
OVERRIDES=(
  "run_name=charm_replogle_main_s42"
  "++model_checkpoint_path=$WEIGHT"
  "++device=auto"
  "++sampling.sample_unperturbed=false"
  "++sampling.split=test"
  "++sampling.fixed_cells_per_perturbation=null"
  "++sampling.cells_per_source=null"
  "++sampling.max_perturbations=null"
  "++sampling.num_sampled_batches=null"
  "++sampling.batch_size=512"
  "++sampling.use_ddim=true"
  "++sampling.clip_denoised=true"
  "++sampling.progress=false"
  "++sampling.guidance_strength=0.0"
  "++sampling.initial_state=parent_residual"
  "++sampling.start_time=100"
  "++sampling.nw=0.5"
  "++sampling.start_guide_steps=500"
  "++sampling.eta=0.0"
  "++sampling.output_format=npz"
  "++sampling.output_dir=$OUT/samples"
  "data.dataset_path=$CELL_DETR_DATA_ROOT/finetune_data/nadig_processed_data/replogle.h5ad"
  "data.selected_gene_file=$CELL_DETR_DATA_ROOT/selected_genes/replogle_real_selected_genes.pkl"
  "data.indices_cache_dir=$OUT/cache"
  "data.evaluation_splits=[test]"
)
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$ROOT" "$TRAIN_PY" "$ROOT/scripts/replogle/sample.py" \
  --config-path "$ROOT/configs/frozen" --config-name replogle_train \
  --cfg job --resolve "${OVERRIDES[@]}" > "$OUT/logs/resolved_test.yaml"
if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then echo "CONFIG_ONLY_PASS $OUT/logs/resolved_test.yaml"; exit 0; fi
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$TRAIN_PY" "$ROOT/scripts/replogle/sample.py" --config-path "$OUT/logs" \
  --config-name resolved_test > "$OUT/logs/sample.log" 2>&1
mapfile -t PREDS < <(find "$OUT/samples" -maxdepth 1 -type f -name 'diffusion_predict_*.h5ad')
mapfile -t REALS < <(find "$OUT/samples" -maxdepth 1 -type f -name 'diffusion_true_*.h5ad')
(( ${#PREDS[@]} == 1 && ${#REALS[@]} == 1 )) || { echo 'expected one prediction/truth pair' >&2; exit 3; }
"$TRAIN_PY" "$ROOT/scripts/internal/prepare_parent_locked_celleval_input.py" \
  --input "${PREDS[0]}" --output "$OUT/celleval_input/pred.h5ad" \
  --cell-eval-threshold 15 --max-singleton-fraction 0.01 --max-endpoint-mean-drift 1e-5 \
  > "$OUT/logs/celleval_input.json"
EVAL_INPUT="$("$TRAIN_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation_h5ad"])' "$OUT/logs/celleval_input.json")"
CUDA_VISIBLE_DEVICES='' "$EVAL_PY" "$ROOT/scripts/evaluate_core_metrics.py" \
  --pred "$EVAL_INPUT" --real "${REALS[0]}" --outdir "$OUT/metrics" \
  --dataset replogle --pert-col gene --control non-targeting --num-threads "${EVAL_THREADS:-32}" \
  | tee "$OUT/logs/core_metrics.log"
echo "COMPLETE output=$OUT metrics=$OUT/metrics/core_metrics.json"
