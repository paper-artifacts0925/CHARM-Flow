import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
 'weights/replogle.pt':'f681d8671751456fb054129bbf150fa723ec17faffe7c4f885fa0236feb79e22',
 'weights/pbmc.pt':'a083cff84245b644ebf0e67d7cf0da6963ac77467edb2f7d511ee0d6bed610ab',
 'weights/tahoe.pt':'2c31ef6c936d7447252082406e2e0b64c3fafc91a2f059271d041f2927a2aee4',
 'configs/frozen/replogle_train.yaml':'a6935d413d1d94d03a5a150a32ea499e0b5eca16aeb2f188e38a12e2e3ce4ac7',
 'configs/frozen/pbmc_train.yaml':'be8f0c6fa869a3daa58288ff0aa81ed627e78b338c805e3b9565d161e35e6c58',
 'configs/frozen/tahoe_train.yaml':'4f55f0046992feb27ee28bb198bf932314ba2944595ee50100b21859313f53eb',
}
REQUIRED_RUNTIME_FILES = (
 'train.sh',
 'test.sh',
 'fetch_data.sh',
 'prepare_data.sh',
 'requirements.txt',
 'configs/data/replogle_finetune.yaml',
 'configs/data/perturb_data/replogle.yaml',
 'scripts/rebase_frozen_config.py',
 'scripts/preprocess/prepare.py',
 'scripts/preprocess/build_replogle_pca.py',
 'scripts/test_replogle.sh',
 'scripts/test_pbmc.sh',
 'scripts/test_tahoe.sh',
'scripts/replogle/sample.py', 'scripts/internal/prepare_parent_locked_celleval_input.py', 'scripts/pbmc/run_full.py', 'scripts/pbmc/sample_shard.py', 'scripts/pbmc/sample_model.py', 'scripts/pbmc/shard_runtime.py', 'scripts/pbmc/shard_model.py', 'scripts/pbmc/shard_sampler_core.py', 'scripts/pbmc/evaluate_by_celltype.py', 'scripts/tahoe/sample.py', 'scripts/tahoe/prepare_evaluation.py',
 'scripts/parent_residual_gene_dit/postprocess_parent_locked_celleval_bound.py',
 'scripts/parent_residual_gene_dit/verify_parent_locked_celleval_bound.py',
 'src/models/parent_residual_gene_dit/wrapper.py',
 'src/models/donor_aware_parent_delta/artifact_model.py',
 'src/models/rectified_flow.py',
)
FORBIDDEN_RELEASE_PATHS = (
 'configs/experiments',
 'src/models/anchor_bridge',
 'src/models/cross_dit',
 'src/models/diffusion',
 'src/models/response_parent_gate',
 'src/apps/training/response_parent_gate_training.py',
 'src/models/parent_residual_gene_dit/response_parent_gate_only_training.py',
 'src/models/parent_locked_residual/no_flow.py',
 'src/models/parent_locked_residual/direct_endpoint.py',
 'src/models/parent_mean_direct_flow.py',
)

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''): h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--fast',action='store_true',help='skip hashing the 2.5 GB of weights'); a=p.parse_args()
 rows={}; ok=True
 for rel,want in EXPECTED.items():
  path=ROOT/rel; exists=path.is_file() and path.stat().st_size>0
  required=not rel.startswith('weights/')
  actual=None if (a.fast and rel.startswith('weights/')) or not exists else sha(path)
  passed=(not required and not exists) or (exists and (actual is None or actual==want)); ok &= passed
  rows[rel]={'exists':exists,'required':required,'size_bytes':path.stat().st_size if exists else None,'sha256':actual,'expected_sha256':want,'passed':passed}
 expected_weights={'pbmc.pt','replogle.pt','tahoe.pt'}
 actual_weights={path.name for path in (ROOT/'weights').glob('*.pt')}
 other_model_files=sorted(
  str(path.relative_to(ROOT))
  for pattern in ('*.ckpt','*.pth','*.safetensors')
  for path in ROOT.rglob(pattern)
 )
 weights_exact=actual_weights in (set(),expected_weights) and not other_model_files
 rows['packaged_weight_set']={
  'actual':sorted(actual_weights),
  'expected':'empty for Git, or all three published weights',
  'required':False,
  'unexpected_model_files':other_model_files,
  'passed':weights_exact,
 }
 ok &= weights_exact
 expected_configs={'pbmc_train.yaml','replogle_train.yaml','tahoe_train.yaml'}
 actual_configs={path.name for path in (ROOT/'configs'/'frozen').glob('*_train.yaml')}
 configs_exact=actual_configs==expected_configs
 rows['frozen_training_config_set']={
  'actual':sorted(actual_configs),
  'expected':sorted(expected_configs),
  'passed':configs_exact,
 }
 ok &= configs_exact
 required={rel:(ROOT/rel).is_file() for rel in REQUIRED_RUNTIME_FILES}
 forbidden={rel:(ROOT/rel).exists() for rel in FORBIDDEN_RELEASE_PATHS}
 cache_files=sorted(
  str(path.relative_to(ROOT)) for path in ROOT.rglob('*')
  if path.name == '__pycache__' or path.suffix == '.pyc'
 )
 editor_backups=sorted(
  str(path.relative_to(ROOT)) for path in ROOT.rglob('*.orig')
 )
 scope_passed=(
  all(required.values()) and not any(forbidden.values())
  and not cache_files and not editor_backups
 )
 rows['release_scope']={
  'required_runtime_files':required,
  'forbidden_paths_present':forbidden,
  'cache_files':cache_files,
  'editor_backup_files':editor_backups,
  'passed':scope_passed,
 }
 ok &= scope_passed
 resource=Path(os.environ.get('CHARM_RESOURCE_ROOT',str(ROOT/'resources')))
 external={
  'external_data':Path(os.environ.get('CHARM_DATA_ROOT',str(resource/'data/PerturbDiff_data/perturb_data'))),
  'external_artifacts':Path(os.environ.get('CHARM_ARTIFACT_ROOT',str(resource/'artifacts'))),
 }
 for label,path in external.items():
  rows[label]={'exists':path.exists(),'resolved':str(path.resolve()) if path.exists() else None,'required':False,'passed':True}
 payload={'passed':bool(ok),'root':str(ROOT),'checks':rows}
 print(json.dumps(payload,indent=2,sort_keys=True)); raise SystemExit(0 if ok else 1)
if __name__=='__main__': main()
