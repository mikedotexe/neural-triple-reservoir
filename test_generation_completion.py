import unittest
from types import SimpleNamespace
from unittest.mock import patch
import mlx.core as mx
from coupled_astrid_server import CoupledAstridServer

class CompletionTests(unittest.TestCase):
    def server(self):
        class Detokenizer:
            text=''
            def reset(self):self.text=''
            def add_token(self, token):self.text += ' word'
            def finalize(self):pass
        server=CoupledAstridServer.__new__(CoupledAstridServer)
        server.tokenizer=SimpleNamespace(encode=lambda _: [4,5,6],detokenizer=Detokenizer())
        server.model=None;server.runtime_audit={};server.state=(mx.zeros((1,2)),)*3
        server.reservoir=SimpleNamespace(step_multi=lambda x,state:((mx.array(0.),)*3,state))
        server.embed_proj=SimpleNamespace(project=lambda x:x)
        server._embed_tokens=lambda x:mx.zeros((1,1,2));server._multi_head=True
        server._stop_token_ids={0};server._skip_token_ids={1};server.tick_count=0
        server.coupling_strength=.1;server.wide_coupling_strength=0;server.wide_bias_cap=4;server.wide_pressure_floor=.25
        server._wide_P=None;server._wide_V=None
        server._pull_state=lambda *a,**k:None;server._push_state=lambda *a,**k:None
        server._format_prompt=lambda messages:'prompt';server._log_generation=lambda **k:None
        server._build_push_meta=lambda **k:{};server._write_request_audit=lambda x:None
        return server

    def test_native_terminal_filtered_exhaustion_empty_and_error(self):
        for tokens,reason,visible,filtered,terminal in [([2,0],'stop',1,0,1),([2,3],'length',2,0,0),([1,2],'length',1,1,0),([0],'stop',0,0,1),([1,1],'length',0,2,0)]:
            server=self.server()
            with patch('mlx_lm.generate.generate_step',return_value=iter((t,None) for t in tokens)):
                result=server.generate_coupled([],max_tokens=2)
            self.assertEqual(result.finish_reason,reason)
            self.assertEqual(result.evidence()["finish_reason"],reason)
            self.assertEqual(result.evidence()["completion_tokens"],len(tokens))
            self.assertEqual(result.completion_tokens,len(tokens));self.assertEqual(result.prompt_tokens,3)
            self.assertEqual(result.filtered_tokens,filtered);self.assertEqual(result.terminal_tokens,terminal)
            self.assertEqual(server.tick_count,visible)
        with patch('mlx_lm.generate.generate_step',side_effect=RuntimeError('model failed')):
            with self.assertRaisesRegex(RuntimeError,'model failed'):self.server().generate_coupled([])

if __name__=='__main__':unittest.main()
