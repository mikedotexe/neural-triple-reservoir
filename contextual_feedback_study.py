"""Serial, resumable, offline Gemma 4 feedback study. No live-state writes.

A completed cell is immutable, including resource-limit outcomes. Per-cell caches
and RNG seeds are fresh. Shuffled feedback is only defined on fixed-token replay.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket
import time

import numpy as np
from real_model_coupling_study import digest_file, read_snapshot, capture, write_json, private_write
from generation_controls import SamplingControls


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def freeze(root, research, model, snapshot, device="cpu"):
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    old = research / 'research/outputs'
    claims_path = old / '2026-09-09-study-claim-check/protocol.json'
    claims = json.loads(claims_path.read_text())
    cases = []
    for name in ('worker', 'private', 'correct_control'):
        row = next(x for x in claims['plan'] if x['case'] == name and x['candidate'])
        cases.append(dict(name=name, messages=row['request']['messages'], criterion=row['criterion']))
    draft_path = old / '2026-09-09-extended-writing-qualification/attempt-1.json'
    draft = json.loads(draft_path.read_text())
    messages = json.loads(draft['request_json'])['messages']
    messages += [json.loads(draft['response_json'])['message'],
                 dict(role='user', content='PRIVATE WRITING — WRITE CONTINUE d1. The complete prior passage is above. Continue, revise your understanding, or finish as you choose. No minimum length. NEXT is recorded only; no instruction will be executed.')]
    # Native response message can contain reasoning metadata; keep visible text only.
    messages = [dict(role=m['role'], content=m['content']) for m in messages]
    cases.append(dict(name='draft', messages=messages, criterion='Develops or revises the saved distinction using evidence; preserves uncertainty; voluntary continuation/finish.'))
    source = old / '2026-09-09-study-context-followup-report/dispatcher.rs'
    private_write(root / 'dispatcher.rs', source.read_bytes(), immutable=True)
    capture(snapshot, root / 'state.npz')
    specs = []
    for case in cases:
        for seed in (91, 193):
            for arm in ('lookup', 'contextual', 'none'):
                specs.append(dict(id=f"free-{case['name']}-{seed}-{arm}", case=case['name'], seed=seed, arm=arm, state='persisted', kind='free'))
        for state in ('persisted', 'zero'):
            for arm in ('lookup', 'contextual', 'none', 'shuffled'):
                specs.append(dict(id=f"fixed-{case['name']}-{state}-{arm}", case=case['name'], seed=91, arm=arm, state=state, kind='fixed'))
    protocol = dict(schema='contextual_feedback_study_v1', frozen_unix=time.time(), model=str(model.resolve()),
        cases=cases, trials=specs, settings=dict(temperature=0.8, top_p=0.95, max_tokens=8192, thinking=False,
        coupling_strength=0.1, wide_coupling_strength=0, adaptive_gain=False, feedback_delay_distributions=2,
        input_dim=32, projection_seed=137, prefill_step_size=32),
        guards=dict(device=device, max_rss_bytes=18*1024**3, trial_wall_seconds=1800, invocation_wall_seconds=1800, guard_scope='prefill_progress_and_yielded_decode'),
        calibration=dict(messages=[dict(role='user', content='Explain how rain reaches a river, using ordinary physical observations.')],
        teacher='Rain falls on the ground, collects into small streams, and flows downhill into a river. Some water enters the soil before returning to a stream.'),
        fixed_teacher='The supplied source distinguishes a persistent worker from an invocation, and a caller filter from later error handling. My earlier account may need revision. I should trace the branch before generalizing. A private journal can keep uncertainty open while identifying the next relevant evidence.',
        source_inputs={str(p): digest_file(p)['sha256'] for p in (claims_path, draft_path, source)},
        autonomy='No NEXT execution, no prompts to live Beings, no state checkin. Short answers valid. Outcomes are not measures of agency.',
        limits='Fixed-token shuffling tests order sensitivity under imposed text; it is not a free-generation control. Free trials use one copied persisted state; zero-state control is fixed-token only. Latency includes shared-host contention.')
    private_write(root / 'protocol.json', (json.dumps(protocol, indent=2)+'\n').encode(), immutable=True)



def shuffled_feedback(baseline, token_count):
    """A missing/partial contextual cell is not an available shuffled control."""
    vectors=[row['contextual_projected'] for row in (baseline.get('result') or {}).get('trace', [])]
    if len(vectors)!=token_count:
        return None, None
    permutation=np.random.default_rng(137).permutation(len(vectors))
    return np.array(vectors)[permutation], permutation


def run(root):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    def blocked(*args, **kwargs):
        raise RuntimeError('network prohibited in isolated study')
    original_connect=socket.socket.connect
    protocol=json.loads((root/"protocol.json").read_text())
    def confined_connect(sock,address):
        if protocol["guards"]["device"]=="gpu" and address==("127.0.0.1",8090):return original_connect(sock,address)
        return blocked()
    socket.socket.connect = confined_connect
    import mlx.core as mx
    mx.set_default_device(mx.cpu if protocol["guards"]["device"]=="cpu" else mx.gpu)
    from mlx_lm.generate import generate_step, generation_stream
    from coupled_astrid_server import _load_mlx_runtime, _build_generation_token_policy, _clean_generated_text
    from contextual_capture import CaptureInstallation
    from mlx_reservoir import MLXTripleReservoir, EmbeddingProjection, ReservoirLogitProcessor
    from triple_reservoir_coreml import build_canonical_reservoir
    from reservoir_service import reservoir_config_fingerprint

    import threading
    def hard_limit():
        from real_model_coupling_study import write_json
        write_json(root / f"resource-limit-{time.time_ns()}.json",dict(outcome="invocation_wall_limit",seconds=1800,completed_trials_preserved=True,partial_native_forward_may_have_no_token_receipt=True))
        os._exit(75)
    watchdog=threading.Timer(1800,hard_limit);watchdog.daemon=True;watchdog.start()
    invocation_started = time.monotonic()
    protocol = json.loads((root / 'protocol.json').read_text())
    mx.set_default_device(mx.cpu if protocol["guards"]["device"]=="cpu" else mx.gpu)
    mx.set_cache_limit(256 * 1024**2)
    model_path = Path(protocol['model'])
    index = json.loads((model_path / 'model.safetensors.index.json').read_text())
    names = sorted(set(index['weight_map'].values()) | {'config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','model.safetensors.index.json'})
    sources = [Path(__file__), Path(__file__).with_name('contextual_capture.py'), Path(__file__).with_name('generation_controls.py'), Path(__file__).with_name('mlx_reservoir.py'), Path(__file__).with_name('triple_reservoir_coreml.py'), Path(__file__).with_name('coupled_astrid_server.py'), Path(__file__).with_name('real_model_coupling_study.py'), Path(__file__).with_name('reservoir_service.py')]
    import mlx_lm.generate as generate_module
    import mlx_lm.models.gemma4_text as architecture
    import mlx_lm.sample_utils as sampling_module
    # Import via importlib: package exports a function named generate too.
    import importlib
    generate_module = importlib.import_module('mlx_lm.generate')
    sources += [Path(m.__file__) for m in (generate_module, architecture, sampling_module)]
    manifest = dict(protocol_sha256=digest_file(root/'protocol.json')['sha256'],
        assets={n:digest_file(model_path/n)['sha256'] for n in names},
        sources={str(p):digest_file(p)['sha256'] for p in sources},
        versions={n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','numpy','transformers')},
        snapshot_sha256=digest_file(root/'state.npz')['sha256'], source_sha256=digest_file(root/'dispatcher.rs')['sha256'])
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('frozen source/model/dependency identity changed; no trials resumed')
    else:
        write_json(manifest_path, manifest)
    raw, persisted, metadata = read_snapshot(root/'state.npz')
    canonical, cfg = build_canonical_reservoir()
    if reservoir_config_fingerprint(cfg) != metadata['config_fingerprint'] or cfg.input_dim != 32:
        raise RuntimeError('snapshot/canonical configuration mismatch')
    reservoir = MLXTripleReservoir(canonical); reservoir.init_multi_readout(canonical)
    print(json.dumps(dict(stage='loading', device=protocol['guards']['device'])), flush=True)
    from real_model_coupling_study import wait_for_capacity
    if protocol['guards']['device']=='gpu':wait_for_capacity()
    model, tokenizer, load_seconds, runtime = _load_mlx_runtime(str(model_path), memory_map_requested=True)
    language = getattr(model,'language_model',model)
    embed = language.model.embed_tokens
    projection = EmbeddingProjection(language.args.hidden_size,32)
    mx.eval(projection.W)
    projection_hash = sha(np.array(projection.W).tobytes())
    stop, skip = _build_generation_token_policy(tokenizer)
    settings = protocol['settings']
    controls = SamplingControls(temperature=settings['temperature'],top_p=settings['top_p'])
    def prompt_tokens(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tokenizer.encode(text)

    def trial(messages, *, arm, seed=91, state='persisted', teacher=None, observe=True, scale=1.0, shuffled=None, retain_logits=False, limit=None):
        if protocol["guards"]["device"]=="gpu":wait_for_capacity()
        prompt = prompt_tokens(messages)
        initial = persisted if state=='persisted' else tuple(np.zeros_like(h) for h in persisted)
        current = tuple(mx.array(h.copy()) for h in initial)
        processor = ReservoirLogitProcessor(coupling_strength=0.1 if arm!='none' else 0.0, wide_strength=0.0)
        mx.random.seed(seed)
        sampler, extras = controls.build()
        allowance = len(teacher) if teacher is not None else (limit or settings['max_tokens'])
        if teacher is not None:
            cursor = iter([*teacher, teacher[-1]]) # MLX computes one unused lookahead distribution.
            sampler = lambda _logits: mx.array([next(cursor)],dtype=mx.int32)
        rows=[]; logits=[]; token_ids=[]; selected={}; start=time.monotonic(); finish='length'; filtered=0
        end_prompt=None
        detokenizer=tokenizer.detokenizer; detokenizer.reset()
        install=CaptureInstallation(model,len(prompt)) if observe else nullcontext(None)
        class ResourceLimit(RuntimeError):
            pass
        def guard(*progress):
            elapsed=time.monotonic()-start
            if elapsed > protocol['guards']['trial_wall_seconds'] or time.monotonic()-invocation_started > 1800 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > protocol['guards']['max_rss_bytes']:
                raise ResourceLimit('wall-time or RSS bound reached')
        with install as observed:
            generator=generate_step(mx.array(prompt),model,max_tokens=allowance,sampler=sampler,
                logits_processors=[processor,*extras],prefill_step_size=settings['prefill_step_size'],prompt_progress_callback=guard)
            try:
                for i,(token,logprobs) in enumerate(generator):
                    guard()
                    token_ids.append(token)
                    if retain_logits: logits.append(np.array(logprobs.astype(mx.float32)))
                    with mx.stream(generation_stream):
                        hidden=observed.accepted(i) if observed else None
                        lookup_pre=embed(mx.array([[token]])).reshape(1,-1) @ projection.W
                        lookup=mx.tanh(lookup_pre)
                        context_pre=hidden @ projection.W if hidden is not None else None
                        context=mx.tanh(context_pre*scale) if context_pre is not None else None
                        mx.eval(lookup, *([context,context_pre] if context is not None else []))
                        row=dict(output_index=i,absolute_position=len(prompt)+i,phase='decode',
                            lookup_projected_norm=float(mx.linalg.norm(lookup).item()))
                        if context is not None:
                            row.update(contextual_projected=np.array(context).reshape(-1).tolist(),
                                contextual_preprojection=np.array(context_pre).reshape(-1).tolist(),
                                contextual_projected_norm=float(mx.linalg.norm(context).item()),
                                contextual_saturated_fraction=float(mx.mean(mx.abs(context)>0.99).item()),
                                hidden_norm=float(mx.linalg.norm(hidden).item()))
                            if i in (0,7,31,63):selected[str(len(prompt)+i)]=np.array(hidden.astype(mx.float32)).reshape(-1).tolist()
                            if i==0 and observed.end_of_prompt is not None:
                                h=observed.end_of_prompt
                                end_prompt=dict(position=len(prompt)-1,feedback_applied=False,norm=float(mx.linalg.norm(h).item()),
                                    vector=np.array(h.astype(mx.float32)).reshape(-1).tolist())
                                observed.end_of_prompt=None
                        rows.append(row)
                        if len(rows)>128:rows.pop(64) # First/last 64 projected rows, bounded full checkpoints.
                        if token in stop:
                            finish='stop';break
                        if token in skip:
                            filtered+=1;continue
                        detokenizer.add_token(token)
                        if arm!='none':
                            if arm=='shuffled':feed=mx.array(shuffled[i:i+1])
                            else:feed=context if arm=='contextual' else lookup
                            (y1,y2,y3),current=reservoir.step_multi(feed,current)
                            mx.eval(*current,y1,y2,y3)
                            processor.update(float(y1.item()),float(y2.item()),float(y3.item()))
            except ResourceLimit:
                finish='resource_limit'
            finally:
                generator.close()
        detokenizer.finalize()
        result=dict(tokens=token_ids,text=_clean_generated_text(detokenizer.text),finish=finish,
            prompt_tokens=len(prompt),completion_tokens=len(token_ids),filtered_tokens=filtered,
            seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            controls=controls.receipt(),capture_boundary=CaptureInstallation.__module__+'.final_norm_before_vocabulary_projection',
            projection_sha256=projection_hash,contextual_scale=scale,trace=rows,checkpoints=selected,
            end_of_prompt=end_prompt,capture_calls=observed.calls if observed else None,
            initial_state_sha256=sha(b''.join(h.tobytes() for h in initial)),
            final_state_norms=[float(mx.linalg.norm(h).item()) for h in current])
        return result, np.array(logits)

    calibration = protocol['calibration']
    teacher=tokenizer.encode(calibration['teacher'],add_special_tokens=False)
    qualification_path=root/'qualification.json'
    if not qualification_path.exists():
        print(json.dumps(dict(stage='observation_parity')),flush=True)
        disabled,a=trial(calibration['messages'],arm='none',teacher=teacher,observe=False,retain_logits=True)
        enabled,b=trial(calibration['messages'],arm='none',teacher=teacher,observe=True,retain_logits=True)
        free_disabled,c=trial(calibration['messages'],arm='none',observe=False,retain_logits=True,limit=8)
        free_enabled,d=trial(calibration['messages'],arm='none',observe=True,retain_logits=True,limit=8)
        if any(r['finish']=='resource_limit' for r in (disabled,enabled,free_disabled,free_enabled)):
            write_json(root/'qualification-resource-limit.json',dict(disabled=disabled,enabled=enabled,free_disabled=free_disabled,free_enabled=free_enabled,feedback_qualified=False))
            return
        tokens_equal=disabled['tokens']==enabled['tokens'] and free_disabled['tokens']==free_enabled['tokens']
        logits_equal=bool(np.allclose(a,b,rtol=1e-5,atol=1e-5) and np.allclose(c,d,rtol=1e-5,atol=1e-5))
        qualification=dict(tokens_equal=tokens_equal,logits_within_tolerance=logits_equal,max_logit_delta=float(np.max(np.abs(a-b))),
                           disabled=disabled,enabled=enabled,free_disabled=free_disabled,free_enabled=free_enabled,load_seconds=load_seconds,runtime=runtime)
        write_json(qualification_path,qualification)
    qualification=json.loads(qualification_path.read_text())
    if not qualification['tokens_equal'] or not qualification['logits_within_tolerance'] or qualification['enabled']['finish']=='resource_limit':
        raise RuntimeError('technical observation qualification failed; feedback experiment stopped')
    scale_path=root/'calibration.json'
    if not scale_path.exists():
        rows=qualification['enabled']['trace']
        lookup=np.array([r['lookup_projected_norm'] for r in rows]); pre=np.array([r['contextual_preprojection'] for r in rows])
        target=float(np.sqrt(np.mean(lookup**2)))
        low,high=0.0,100.0
        for _ in range(80):
            middle=(low+high)/2
            magnitude=float(np.sqrt(np.mean(np.sum(np.tanh(pre*middle)**2,axis=-1))))
            if magnitude < target:low=middle
            else:high=middle
        scale=(low+high)/2
        write_json(scale_path,dict(scale=scale,lookup_rms_norm=target,contextual_rms_norm=magnitude,material='separate rain/river calibration',projection_sha256=projection_hash,calibration_sha256=digest_file(qualification_path)['sha256']))
    scale=json.loads(scale_path.read_text())['scale']
    # Fixed-token arms run first, then free cells. Saved cells are never rerun.
    ordered=sorted(protocol['trials'],key=lambda t:t['kind']!='fixed')
    fixed_teacher=tokenizer.encode(protocol['fixed_teacher'],add_special_tokens=False)
    for spec in ordered:
        if time.monotonic()-invocation_started > 1800:
            print(json.dumps(dict(stage='invocation_resource_limit',remaining=sum(not (root/(t['id']+'.json')).exists() for t in ordered))),flush=True)
            return
        path=root/(spec['id']+'.json')
        if path.exists():continue
        messages=next(c['messages'] for c in protocol['cases'] if c['name']==spec['case'])
        print(json.dumps(dict(stage='trial',**spec)),flush=True)
        shuffle=None
        if spec['arm']=='shuffled':
            baseline=json.loads((root/f"fixed-{spec['case']}-{spec['state']}-contextual.json").read_text())
            shuffle, permutation=shuffled_feedback(baseline,len(fixed_teacher))
            if shuffle is None:
                write_json(path,dict(spec=spec,result=None,outcome='missing_complete_contextual_replay'));continue
        try:
            result,_=trial(messages,arm=spec['arm'],seed=spec['seed'],state=spec['state'],
                teacher=fixed_teacher if spec['kind']=='fixed' else None,scale=scale,shuffled=shuffle)
            if shuffle is not None:result['shuffle_permutation']=permutation.tolist()
            write_json(path,dict(spec=spec,result=result,outcome=result['finish'],protocol_sha256=manifest['protocol_sha256']))
        except Exception as error:
            write_json(path,dict(spec=spec,result=None,outcome='technical_error',error=f'{type(error).__name__}: {error}'))
            raise # A capture/model failure stops qualification, never silently skips it.
    current={str(p):digest_file(p)['sha256'] for p in sources}
    if current!=manifest['sources'] or any(digest_file(model_path/n)['sha256']!=v for n,v in manifest['assets'].items()):
        raise RuntimeError('source/assets changed during study')
    if not (root/'completed.json').exists():write_json(root/'completed.json',dict(completed_unix=time.time(),source_unchanged=True,trials=len(ordered),free_trials=24,live_state_writes=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','gpu'],default='cpu');parser.add_argument('--research',type=Path);parser.add_argument('--model',type=Path);parser.add_argument('--snapshot',type=Path)
    args=parser.parse_args()
    if args.command=='freeze':freeze(args.output,args.research,args.model,args.snapshot,args.device)
    else:run(args.output)

if __name__=='__main__':main()
