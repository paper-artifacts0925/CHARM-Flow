#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PY="${CHARM_TRAIN_PYTHON:-python}"
EVAL_PY="${CHARM_EVAL_PYTHON:-python}"
GPU="${GPU:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUTPUT_DIR:-$ROOT/runs/test/tahoe_$STAMP}"
WEIGHT="$ROOT/weights/tahoe.pt"
EXPECTED=2c31ef6c936d7447252082406e2e0b64c3fafc91a2f059271d041f2927a2aee4
if [[ "${CONFIG_ONLY:-0}" != 1 ]]; then
  [[ "$(sha256sum "$WEIGHT" | awk '{print $1}')" == "$EXPECTED" ]] || { echo 'tahoe weight hash mismatch' >&2; exit 3; }
fi
[[ ! -e "$OUT" ]] || { echo "refusing existing output: $OUT" >&2; exit 4; }
mkdir -p "$OUT" "$OUT/logs"
RESOURCE_ROOT="${CHARM_RESOURCE_ROOT:-$ROOT/resources}"
DATA_ROOT="${CHARM_DATA_ROOT:-$RESOURCE_ROOT/data/PerturbDiff_data/perturb_data}"
ARTIFACT_ROOT="${CHARM_ARTIFACT_ROOT:-$RESOURCE_ROOT/artifacts}"
export CHARM_RESOURCE_ROOT="$RESOURCE_ROOT" CHARM_DATA_ROOT="$DATA_ROOT" CHARM_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export WORK_ROOT="$RESOURCE_ROOT" OUTPUT_ROOT="$ROOT/runs" PERTURB_DATA_ROOT="$DATA_ROOT"
export PYTHONUNBUFFERED=1 TAHOE_SCREEN_AUDIT_ONLY=false CUDA_VISIBLE_DEVICES="$GPU"
ENTRY="$ROOT/scripts/tahoe/sample.py"
HYDRA_MODE=()
if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then HYDRA_MODE=(--cfg job --resolve); fi
"$TRAIN_PY" "$ENTRY" "${HYDRA_MODE[@]}" \
  "model_checkpoint_path=$WEIGHT" data=tahoe100m_finetune model=parent_residual_ocoot_u2_core \
  path=local_path cov_encoding=trixie_onehot cov_encoding.celltype_encoding=llm \
  cov_encoding.replogle_gene_encoding=onehot data.use_cell_set=1 '+data.evaluation_splits=[test]' \
  data.normalize_counts=10 data.num_workers=2 data.prefetch_factor=2 data.persistent_workers=false \
  data.skip_cached_indices=false \
  "data.control_distribution_context_path=$ARTIFACT_ROOT/ocoot/tahoe100m_recursive128_minleaf20_context_v2.runtime.pkl" \
  data.control_context_mode=latent_cluster_mixture data.control_bank_path=null \
  +data.control_members_per_token=1 +data.latent_control_candidate_strategy=all \
  data.latent_control_top_k=0 +data.latent_control_fixed_max_k=true \
  data.latent_control_use_cluster_sets=false +data.cell_detr_unique_control_bank=true \
  +data.hungarian_flow_combination_sampler=false model.parent_residual_context_hidden_dim=256 \
  model.parent_residual_positive_multiplicity_cap_to_active_children=true \
  "model.parent_residual_artifact_path=$ARTIFACT_ROOT/parent_residual_gene_dit/tahoe100m_parent_residual_ocoot_v1.npz" \
  model.parent_residual_artifact_sha256=d12b79060c60f2c475b0bb1c1f041e28fb9ecc140f9b7cf2644bdad825fbb467 \
  "model.parent_residual_set_pca_path=$ARTIFACT_ROOT/ocoot/tahoe100m_control_pca64_div10_v2.npz" \
  model.parent_residual_set_pca_sha256=989aa9bb8488567b653016d07da5cc388e1969eb75359f974715801b6ce61d63 \
  "model.parent_residual_real_control_reservoir_path=$ARTIFACT_ROOT/ocoot/tahoe100m_recursive128_minleaf20_real_control_div10_v2.npz" \
  model.parent_residual_real_control_reservoir_sha256=48a5b5fe5706e8d38b60ef380703766fa478d064f5c9997dadcf3dcbb853d9bd \
  optimization.micro_batch_size=512 optimization.seed=42 sampling.batch_size=512 \
  sampling.split=test sampling.num_sampled_batches=null sampling.max_perturbations=null \
  sampling.fixed_cells_per_perturbation=null sampling.guidance_strength=0.0 sampling.progress=false \
  sampling.initial_state=parent_residual "sampling.output_dir=$OUT" \
  ++sampling.parent_locked_max_singleton_fraction=0.0 \
  ++sampling.parent_locked_endpoint_mean_drift_tolerance=1e-5 \
  +sampling.tahoe_groups_per_scenario=245 +sampling.tahoe_cells_per_group=64 \
  run_name=charm_tahoe_main_s42 device=cuda:0 \
  lightning.logger._target_=pytorch_lightning.loggers.logger.DummyLogger \
  '~lightning.logger.project' '~lightning.logger.save_dir' '~lightning.logger.name' \
  2>&1 | tee "$OUT/logs/sample.log"
if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then echo "CONFIG_ONLY_PASS $OUT/logs/sample.log"; exit 0; fi
"$EVAL_PY" "$ROOT/scripts/tahoe/prepare_evaluation.py" \
  --test-dir "$OUT" --output-dir "$OUT/celleval_input" 2>&1 | tee "$OUT/logs/prepare.log"
"$EVAL_PY" "$ROOT/scripts/evaluate_core_metrics.py" \
  --pred "$OUT/celleval_input/prediction_with_canonical_controls.h5ad" \
  --real "$OUT/celleval_input/truth_with_canonical_controls.h5ad" \
  --outdir "$OUT/metrics" --dataset tahoe --pert-col drugname_drugconc \
  --control "[('DMSO_TF', 0.0, 'uM')]" --num-threads "${EVAL_THREADS:-32}" \
  | tee "$OUT/logs/core_metrics.log"
echo "COMPLETE output=$OUT metrics=$OUT/metrics/core_metrics.json"
