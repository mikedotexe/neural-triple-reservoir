"""Additional same-device raw-logit qualification, serial after the study worker.

The existing capture-off/on comparison retains normalized log probabilities. This
checks the vocabulary projection before normalization as well, so an additive
logit discrepancy cannot be hidden by log-softmax. It performs no feedback ticks.
"""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import numpy as np
from real_model_coupling_study import write_json, digest_file, wait_for_capacity


def main(root):
    protocol=json.loads((root/'protocol.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    if (root/'raw-logit-qualification.json').exists():raise ValueError('qualification already recorded')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    original=socket.socket.connect
    def confined(sock,address):
        if address==('127.0.0.1',8090):return original(sock,address)
        raise RuntimeError('offline qualification prohibits network')
    socket.socket.connect=confined
    watchdog=threading.Timer(1800,lambda:os._exit(75));watchdog.daemon=True;watchdog.start()
    import mlx.core as mx
    mx.set_default_device(mx.gpu if protocol['guards']['device']=='gpu' else mx.cpu)
    mx.set_cache_limit(256*1024**2)
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from coupled_astrid_server import _load_mlx_runtime
    from contextual_capture import CaptureInstallation
    for path,digest in manifest['sources'].items():
        if digest_file(Path(path))['sha256']!=digest:raise ValueError('frozen source changed')
    wait_for_capacity()
    model,tokenizer,_,_= _load_mlx_runtime(protocol['model'],memory_map_requested=True)
    prompt=tokenizer.encode(tokenizer.apply_chat_template(protocol['calibration']['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False))
    class RecordingModel:
        def __init__(self,body):self.body=body;self.rows=[];self.calls=[];self.position=0
        def __getattr__(self,name):return getattr(self.body,name)
        def __call__(self,inputs,*args,**kwargs):
            output=self.body(inputs,*args,**kwargs)
            last=output[:,-1,:].astype(mx.float32);mx.eval(last)
            self.rows.append(np.array(last));self.calls.append([self.position,self.position+inputs.shape[1]-1])
            self.position+=inputs.shape[1]
            return output
    def replay(observe):
        wait_for_capacity();mx.random.seed(91);wrapped=RecordingModel(model);cache=model.make_cache()
        installation=CaptureInstallation(model,len(prompt)) if observe else nullcontext(None)
        tokens=[];positions=[]
        with installation as capture:
            for i,(token,_) in enumerate(generate_step(mx.array(prompt),wrapped,max_tokens=8,prompt_cache=cache,sampler=make_sampler(temp=.8,top_p=.95),prefill_step_size=16)):
                tokens.append(token)
                if capture:
                    vector=capture.accepted(i);mx.eval(vector);positions.append(len(prompt)+i)
        return wrapped,tokens,[item.offset for item in cache],positions
    before,tokens_a,offsets_a,_=replay(False);after,tokens_b,offsets_b,positions=replay(True)
    a=np.array(before.rows);b=np.array(after.rows)
    parity=bool(np.allclose(a,b,rtol=1e-5,atol=1e-5))
    result=dict(schema='contextual_raw_logit_parity_v1',device=protocol['guards']['device'],
        source_manifest_sha256=digest_file(root/'manifest.json')['sha256'],
        raw_logits_within_tolerance=parity,max_absolute_delta=float(np.max(np.abs(a-b))),
        raw_logits_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in (a,b)],
        tokens_equal=tokens_a==tokens_b,tokens=tokens_a,cache_offsets_equal=offsets_a==offsets_b,
        cache_offsets=offsets_a,positions=positions,forward_ranges_equal=before.calls==after.calls,
        forward_ranges=before.calls,rtol=1e-5,atol=1e-5,no_extra_model_forward=True,live_state_writes=0,
        recorded_unix=time.time(),runner_sha256=digest_file(Path(__file__))['sha256'])
    write_json(root/'raw-logit-qualification.json',result)
    if not all([parity,result['tokens_equal'],result['cache_offsets_equal'],result['forward_ranges_equal']]):raise RuntimeError('raw-logit qualification failed; feedback interpretation invalid')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main(Path(sys.argv[1]))
