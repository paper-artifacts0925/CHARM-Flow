#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-}"
case "$DATASET" in replogle|pbmc|tahoe|all) ;; *) echo "usage: $0 {replogle|pbmc|tahoe|all}" >&2; exit 2 ;; esac
command -v hf >/dev/null || { echo "missing Hugging Face 'hf' CLI" >&2; exit 2; }
command -v zstd >/dev/null || { echo "missing zstd CLI" >&2; exit 2; }
RESOURCE_ROOT="${CHARM_RESOURCE_ROOT:-$ROOT/resources}"
LOCAL_ROOT="${CHARM_HF_LOCAL_ROOT:-$RESOURCE_ROOT/data/PerturbDiff_data}"
CACHE_ROOT="${CHARM_CACHE_ROOT:-$RESOURCE_ROOT/cache/huggingface}"
export HF_HOME="$CACHE_ROOT"
mkdir -p "$LOCAL_ROOT" "$CACHE_ROOT"

fetch_replogle() {
  hf download katarinayuan/PerturbDiff_data --repo-type dataset --local-dir "$LOCAL_ROOT" \
    --include 'perturb_data/finetune_data/nadig_processed_data/replogle.h5ad.zst' \
    --include 'perturb_data/selected_genes/replogle_real_selected_genes.pkl' \
    --include 'perturb_data/selected_genes/vocab_id2name.csv' \
    --include 'perturb_data/gene_names/replogle_gene_emb_dict_perturbation_emb_dict.pkl' \
    --include 'perturb_data/meta_data/new_all_emb.pkl' \
    --include 'perturb_data/indices_cache/grouped_pert_*replogle*.pkl'
}

fetch_pbmc() {
  hf download katarinayuan/PerturbDiff_data --repo-type dataset --local-dir "$LOCAL_ROOT" \
    --include 'perturb_data/finetune_data/pbmc_new/*.h5ad.zst' \
    --include 'perturb_data/selected_genes/pbmc_real_selected_genes.pkl' \
    --include 'perturb_data/selected_genes/vocab_id2name.csv' \
    --include 'perturb_data/meta_data/new_all_emb.pkl' \
    --include 'perturb_data/indices_cache/grouped_pert_*Parse_10M_PBMC_cytokines_processed_Xselected*.pkl'
}

fetch_tahoe() {
  hf download katarinayuan/PerturbDiff_data --repo-type dataset --local-dir "$LOCAL_ROOT" \
    --include 'perturb_data/finetune_data/tahoe100m_full_selected_processed_new/*.h5ad.zst' \
    --include 'perturb_data/selected_genes/tahoe100m_real_selected_genes.pkl' \
    --include 'perturb_data/selected_genes/vocab_id2name.csv' \
    --include 'perturb_data/meta_data/new_all_emb.pkl' \
    --include 'perturb_data/indices_cache/grouped_pert_*Tahoe100M_WServicesFrom_ParseGigalab_processed_final_processed*.pkl'
}

if [[ "$DATASET" == replogle || "$DATASET" == all ]]; then fetch_replogle; fi
if [[ "$DATASET" == pbmc || "$DATASET" == all ]]; then fetch_pbmc; fi
if [[ "$DATASET" == tahoe || "$DATASET" == all ]]; then fetch_tahoe; fi

mapfile -t archives < <(find "$LOCAL_ROOT/perturb_data/finetune_data" -type f -name '*.h5ad.zst' -print | sort)
for archive in "${archives[@]}"; do
  output="${archive%.zst}"
  if [[ ! -s "$output" ]]; then
    zstd -q -t "$archive"
    zstd -T0 -q -d "$archive" -o "$output.part"
    mv "$output.part" "$output"
  fi
done

echo "FETCH_READY dataset=$DATASET data_root=$LOCAL_ROOT/perturb_data"
