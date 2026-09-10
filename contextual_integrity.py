"""Audit retained protocol/state/position/shuffle evidence without model inference."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from real_model_coupling_study import read_snapshot, digest_file


def verify(root):
    protocol=json.loads((root/'protocol.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    trials=protocol['trials'];free=[t for t in trials if t['kind']=='free'];fixed=[t for t in trials if t['kind']=='fixed']
    assert len(free)==24 and len(fixed)==32 and len({t['id'] for t in trials})==56
    assert {t['arm'] for t in free}=={'lookup','contextual','none'}
    assert {t['state'] for t in fixed}=={'zero','persisted'}
    assert {t['seed'] for t in free}=={91,193}
    assert digest_file(root/'protocol.json')['sha256']==manifest['protocol_sha256']
    assert digest_file(root/'state.npz')['sha256']==manifest['snapshot_sha256']
    assert digest_file(root/'dispatcher.rs')['sha256']==manifest['source_sha256']
    for name,digest in manifest['sources'].items():assert digest_file(Path(name))['sha256']==digest,name
    _,layers,_=read_snapshot(root/'state.npz')
    state_hash={name:hashlib.sha256(b''.join(h.tobytes() for h in values)).hexdigest() for name,values in
        [('persisted',layers),('zero',tuple(np.zeros_like(h) for h in layers))]}
    completed=[];failures=[];missing=[]
    for spec in trials:
        path=root/(spec['id']+'.json')
        if not path.exists():missing.append(spec['id']);continue
        cell=json.loads(path.read_text());assert cell['spec']==spec
        value=cell.get('result')
        if value is None:failures.append(dict(id=spec['id'],outcome=cell['outcome'],error=cell.get('error')));continue
        assert value['initial_state_sha256']==state_hash[spec['state']]
        assert len(value['trace'])<=128 and len(value['checkpoints'])<=4
        calls=value['capture_calls'];expected=0
        for call in calls:
            assert call['first']==expected;expected=call['last']+1
        assert expected==value['prompt_tokens']+value['completion_tokens']
        assert value['end_of_prompt']['position']==value['prompt_tokens']-1
        assert value['end_of_prompt']['feedback_applied'] is False
        for row in value['trace']:
            assert row['absolute_position']==value['prompt_tokens']+row['output_index']
            assert len(row['contextual_projected'])==32
        if spec['arm']=='shuffled':
            assert spec['kind']=='fixed'
            assert value['shuffle_permutation']==np.random.default_rng(137).permutation(value['completion_tokens']).tolist()
        if spec['arm']=='none':
            expected_norm=[float(np.linalg.norm(h if spec['state']=='persisted' else np.zeros_like(h))) for h in layers]
            np.testing.assert_allclose(value['final_state_norms'],expected_norm,rtol=1e-5,atol=1e-5)
        completed.append(spec['id'])
    raw=root/'raw-logit-qualification.json'
    qualification=json.loads(raw.read_text()) if raw.exists() else None
    return dict(schema='contextual_integrity_v1',protocol_valid=True,sources_unchanged=True,snapshots_unchanged=True,
        completed=completed,failures=failures,missing=missing,raw_logit_qualification=qualification,
        all_required_evidence_present=not missing and qualification is not None)

if __name__=='__main__':
    result=verify(Path(sys.argv[1]));print(json.dumps(result,indent=2))
