import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--completion',required=True); p.add_argument('--outdir',required=True); a=p.parse_args()
d=json.loads(Path(a.completion).read_text()); h=d['headline_means']; vals={'PDCorr':h['pearson_delta'],'DEOver':h['overlap_at_N'],'DEPrec':h['precision_at_N']}
out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
(out/'core_metrics.json').write_text(json.dumps({'dataset':'pbmc','aggregation':'perturbation-level macro mean','metrics':{k:{'mean':float(v),'count':int(d['perturbations'])} for k,v in vals.items()}},indent=2,sort_keys=True)+'\n')
with (out/'core_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['metric','mean','count']); w.writeheader(); [w.writerow({'metric':k,'mean':v,'count':d['perturbations']}) for k,v in vals.items()]
print(json.dumps(vals,sort_keys=True))
