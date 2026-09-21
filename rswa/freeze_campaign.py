"""Freeze reviewed source and receipts only after all required real gates finish."""
import argparse,shutil
from pathlib import Path
from common import ROOT,read,write,sha
from freeze_inputs import main as freeze_inputs
TASK=ROOT/'dkucc/rswa_20260920'
RESULTS=ROOT/'results'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--pilot-finish',type=Path,required=True)
    ap.add_argument('--order-gate',type=Path,required=True)
    ap.add_argument('--full-equivalence',type=Path,required=True)
    ap.add_argument('--review',type=Path,required=True);a=ap.parse_args()
    assert not a.root.exists(),'Never replace an existing campaign'
    assert a.root.resolve().parent==RESULTS.resolve()
    review=read(a.review)
    assert review['passed'] and review['full_equivalence_run']==str(a.full_equivalence)
    assert review['original_failures_preserved'] and review['quality_claim'] is False
    proofs=[a.review]
    def require(path,key='passed'):
        value=read(path);assert value[key],(path,key);proofs.append(path);return value
    for run in [a.pilot_finish,a.order_gate,a.full_equivalence]:
        receipt=read(run/'job_exit.json');assert receipt['exit_code']==0,run
        proofs.append(run/'job_exit.json')
    require(RESULTS/'rswa-core-64084/contracts.json')
    require(TASK/'evidence/dev_length_check.json')
    require(TASK/'evidence/audio_content_audit.json')
    require(TASK/'evidence/recording_identity_audit.json')
    require(RESULTS/'rswa-train-gate-64086/sp2_contract.json')
    require(RESULTS/'rswa-train-gate-64086/adapter_check.json')
    for run in [RESULTS/'rswa-train-gate-64087/sp1',RESULTS/'rswa-train-gate-64087/sp4',
                RESULTS/'rswa-train-gate-64087/longest-sp4',RESULTS/'rswa-long-gate-64098/typical-sp4']:
        state=run/'checkpoint-1/trainer_state.json'
        assert read(state)['global_step']==1,run
        proofs += [state,run/'rswa_runtime.json']
    require(RESULTS/'rswa-eval-pilot-64279/dynamic_contract.json')
    require(RESULTS/'rswa-eval-pilot-64279/sp4_dynamic_contract.json')
    require(RESULTS/'rswa-infer-gate-64088/vllm_contract.json')
    require(RESULTS/'rswa-restore-gate-64097/restored/restore_exact.json')
    for rank in range(4):require(RESULTS/f'rswa-restore-gate-64097/restored/rng_restore_rank{rank}.json')
    order=require(a.order_gate/'training/order_audit.json')
    assert order['optimizer_steps']==2 and order['actual_meetings']==8 and order['all_four_rank_orders_equal']
    require(a.pilot_finish/'pilot_completion.json')
    for sample in ['smoke','typical']:
        path=a.full_equivalence/sample/'equivalence.json';rows=read(path)['records']
        assert [r['mode'] for r in rows]==['native','native_repeat','custom_full','nonbinding_window']
        assert all(len(r['gradient_norms'])==683 for r in rows)
        assert review['full_equivalence_sha256'][sample]==sha(path)
        proofs.append(path)
    for path in [RESULTS/'rswa-triton-gate-64276/compact_audit.json',
                 RESULTS/'rswa-triton-gate-64276/distribution_review.json',
                 RESULTS/'rswa-eval-pilot-64279/distribution_review.json',
                 RESULTS/'rswa-cache-reset-64279/check/complete.json']:
        read(path);proofs.append(path)
    rows=read(RESULTS/'rswa-triton-gate-64276/compact_audit.json')['records']
    for sample in ['smoke','typical','longest']:
        for window in [128,256]:
            r=rows[f'{sample}-{window}/hf_alignment.json']
            assert r['reference_unchanged'] and r['hf_storage_bounded'] and r['vllm_pages_bounded']
            assert not r['high_margin_disagreements']
            assert r['tokens']>8*window if sample!='smoke' else r['tokens']>=2*window
    # These are executable source contracts, not merely receipts from similar code.
    source_contracts={
        'attention.py':RESULTS/'rswa-eval-pilot-64279/source',
        'cache.py':RESULTS/'rswa-triton-gate-64276/source',
        'kv_audit.py':RESULTS/'rswa-triton-gate-64276/source',
        'model_adapter.py':a.order_gate/'source',
        'moss_rswa_plugin.py':a.order_gate/'source',
        'runtime_callback.py':a.order_gate/'source',
        'native_sft.sh':a.order_gate/'source',
        'evaluate_ce.py':a.pilot_finish/'source',
        'input_receipt.py':a.pilot_finish/'source',
        'full_equivalence.py':a.full_equivalence/'source',
        'performance.py':RESULTS/'rswa-cache-reset-64279/source',
        'moss_vllm_rswa.py':RESULTS/'rswa-cache-reset-64279/source',
    }
    for name,source in source_contracts.items():assert sha(TASK/name)==sha(source/name),(name,source)
    freeze_inputs()
    proofs += [TASK/'evidence/run_contract.json',TASK/'evidence/additional_inputs_contract.json',
               TASK/'evidence/audio_content_audit.json',TASK/'evidence/environment.json']
    manifest=read(TASK/'source_manifest.json')
    for name,digest in manifest.items():
        assert Path(name).name==name and sha(TASK/name)==digest,(name,'Changed source')
    source=a.root/'source';source.mkdir(parents=True)
    for name in list(manifest)+['source_manifest.json']:shutil.copy2(TASK/name,source/name)
    receipts={str(path):sha(path) for path in proofs}
    write(a.root/'qualification.json',dict(passed=True,source_manifest_sha256=sha(source/'source_manifest.json'),
        proofs=receipts,source_contracts={k:str(v) for k,v in source_contracts.items()},
        numerical_review=review,quality_claim=False,training_lr=1e-7,inference_penalty=1.02,
        meaning='authorized for the pre-registered experiment; no recognition-quality success asserted'))
    print(str(a.root),flush=True)

if __name__=='__main__':main()
