import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cell_eval import MetricsEvaluator

SKIP_METRICS = [
    "mse_delta", "mae_delta", "pearson_edistance", "clustering_agreement",
    "de_sig_genes_recall", "de_nsig_counts", "overlap_at_50", "overlap_at_100",
    "overlap_at_200", "overlap_at_500", "precision_at_50", "precision_at_100",
    "precision_at_200", "precision_at_500", "mse", "mae",
    "discrimination_score_l1", "discrimination_score_l2",
    "discrimination_score_cosine", "de_spearman_sig", "de_direction_match",
    "de_spearman_lfc_sig", "pr_auc", "roc_auc",
]
RENAME = {"pearson_delta": "PDCorr", "overlap_at_N": "DEOver", "precision_at_N": "DEPrec"}
CORE = ("PDCorr", "DEOver", "DEPrec")

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pert-col", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--dataset", required=True, choices=("replogle","pbmc","tahoe"))
    args=parser.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    evaluator=MetricsEvaluator(
        adata_pred=args.pred, adata_real=args.real, control_pert=args.control,
        pert_col=args.pert_col, num_threads=args.num_threads, batch_size=64,
        outdir=str(out/"celleval_work"),
    )
    results,_=evaluator.compute(profile="full",skip_metrics=SKIP_METRICS,write_csv=False,break_on_error=True)
    frame=results.to_pandas().rename(columns=RENAME)
    missing=[key for key in CORE if key not in frame]
    if missing: raise RuntimeError(f"Cell-Eval omitted required metrics: {missing}")
    columns=["perturbation",*CORE] if "perturbation" in frame else list(CORE)
    core=frame[columns].copy()
    core.to_csv(out/"core_metrics_per_perturbation.csv",index=False)
    summary={}
    for name in CORE:
        values=pd.to_numeric(core[name],errors="coerce").to_numpy(dtype=float)
        values=values[np.isfinite(values)]
        summary[name]={"mean":float(values.mean()),"std":float(values.std(ddof=1)) if len(values)>1 else None,"count":int(len(values))}
    payload={"dataset":args.dataset,"cell_eval":importlib.metadata.version("cell-eval"),"aggregation":"perturbation-level macro mean","metrics":summary}
    (out/"core_metrics.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    pd.DataFrame([{"metric":k,**v} for k,v in summary.items()]).to_csv(out/"core_metrics.csv",index=False)
    print(json.dumps({k:v["mean"] for k,v in summary.items()},sort_keys=True))
if __name__=="__main__": main()
