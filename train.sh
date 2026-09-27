#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
DATASET="${1:-}"; MODE="${2:-formal14000}"
case "$DATASET" in
  replogle) CONFIG=replogle_train; DEVICES=1; DEFAULT_GPU_SET=0; ACCUM=8; WEIGHT=replogle.pt ;;
  pbmc) CONFIG=pbmc_train; DEVICES=2; DEFAULT_GPU_SET=0,1; ACCUM=4; WEIGHT=pbmc.pt ;;
  tahoe|tahoe100m) DATASET=tahoe; CONFIG=tahoe_train; DEVICES=4; DEFAULT_GPU_SET=0,1,2,3; ACCUM=2; WEIGHT=tahoe.pt ;;
  *) echo "usage: $0 {replogle|pbmc|tahoe} {formal14000|smoke200}" >&2; exit 2 ;;
esac
case "$MODE" in formal14000) STEPS=14000; SAVE_EVERY=2000 ;; smoke200) STEPS=200; SAVE_EVERY=200 ;; *) echo 'mode must be formal14000 or smoke200' >&2; exit 2 ;; esac
PY="${CHARM_TRAIN_PYTHON:-python}"
RESOURCE_ROOT="${CHARM_RESOURCE_ROOT:-$ROOT/resources}"
DATA_ROOT="${CHARM_DATA_ROOT:-$RESOURCE_ROOT/data/PerturbDiff_data/perturb_data}"
ARTIFACT_ROOT="${CHARM_ARTIFACT_ROOT:-$RESOURCE_ROOT/artifacts}"
PBMC_SUPERVISION_ROOT="${CHARM_PBMC_SUPERVISION_ROOT:-$ARTIFACT_ROOT/parent_residual_gene_dit/pbmc_donor_celltype_train_supervision}"
DEFAULT_PREPROCESSING_MANIFEST="$ARTIFACT_ROOT/preprocessing_manifests/$DATASET.json"
MANIFEST_ARGS=()
if [[ -n "${CHARM_PREPROCESSING_MANIFEST:-}" ]]; then
  [[ -f "$CHARM_PREPROCESSING_MANIFEST" ]] || {
    echo "missing CHARM_PREPROCESSING_MANIFEST: $CHARM_PREPROCESSING_MANIFEST" >&2; exit 2;
  }
  MANIFEST_ARGS=(--preprocessing-manifest "$CHARM_PREPROCESSING_MANIFEST")
elif [[ -f "$DEFAULT_PREPROCESSING_MANIFEST" ]]; then
  MANIFEST_ARGS=(--preprocessing-manifest "$DEFAULT_PREPROCESSING_MANIFEST")
fi
GPU_SET="${GPU_SET:-$DEFAULT_GPU_SET}"
IFS=, read -r -a GPU_ARRAY <<< "$GPU_SET"
(( ${#GPU_ARRAY[@]} == DEVICES )) || { echo "$DATASET needs $DEVICES visible GPUs; got $GPU_SET" >&2; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"; NAME="${RUN_NAME:-charm_${DATASET}_${MODE}_s42_$STAMP}"
OUT="${OUTPUT_DIR:-$ROOT/runs/train/$NAME}"
[[ ! -e "$OUT" ]] || { echo "refusing existing output: $OUT" >&2; exit 4; }
mkdir -p "$OUT/checkpoints" "$OUT/cache" "$OUT/logs" "$OUT/trainer" "$OUT/export" "$OUT/csv"
"$PY" "$ROOT/scripts/rebase_frozen_config.py" \
  --input "$ROOT/configs/frozen/$CONFIG.yaml" \
  --output "$OUT/logs/frozen_rebased.yaml" \
  --data-root "$DATA_ROOT" --artifact-root "$ARTIFACT_ROOT" \
  --pbmc-supervision-root "$PBMC_SUPERVISION_ROOT" \
  "${MANIFEST_ARGS[@]}"
OVERRIDES=(
  "run_name=$NAME" "save_dir_path=$OUT/export/$NAME"
  "trainer.default_root_dir=$OUT/trainer" "trainer.devices=$DEVICES"
  "trainer.max_steps=$STEPS" trainer.max_epochs=null trainer.limit_val_batches=0
  trainer.num_sanity_val_steps=0 trainer.use_distributed_sampler=false
  "trainer.accumulate_grad_batches=$ACCUM" "lightning.callbacks.checkpoint.dirpath=$OUT/checkpoints"
  "lightning.callbacks.checkpoint.every_n_train_steps=$SAVE_EVERY"
  lightning.callbacks.checkpoint.every_n_epochs=null lightning.callbacks.checkpoint.save_top_k=-1
  "data.indices_cache_dir=$OUT/cache" optimization.seed=42 optimization.micro_batch_size=64
  data.optimizer_batch_size=64 model.ckpt_path=null model.model_weight_ckpt_path=null
  model.partial_weight_ckpt_path=null model.reinitial_all=false model.reinitial_all_from_scratch=false
)
if [[ "$DATASET" == replogle ]]; then OVERRIDES+=("lightning.logger.save_dir=$OUT/csv"); fi
if [[ "${INIT_FROM_PACKAGED_WEIGHT:-0}" == 1 ]]; then
  OVERRIDES+=("model.model_weight_ckpt_path=$ROOT/weights/$WEIGHT")
fi
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$ROOT" "$PY" "$ROOT/src/apps/run/rawdata_diffusion_training.py" \
  --config-path "$OUT/logs" --config-name frozen_rebased --cfg job --resolve \
  "${OVERRIDES[@]}" > "$OUT/logs/resolved.yaml"
if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then echo "CONFIG_ONLY_PASS config=$OUT/logs/resolved.yaml"; exit 0; fi
CUDA_VISIBLE_DEVICES="$GPU_SET" PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" "$ROOT/src/apps/run/rawdata_diffusion_training.py" --config-path "$OUT/logs" \
  --config-name resolved 2>&1 | tee "$OUT/logs/train.log"
echo "COMPLETE output=$OUT export=$OUT/export/$NAME"
