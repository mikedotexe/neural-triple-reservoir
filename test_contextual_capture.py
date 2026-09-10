import unittest
import numpy as np
import mlx.core as mx
from mlx_lm.models.gemma4_text import Model, ModelArgs
from mlx_lm.generate import generate_step
from contextual_capture import CaptureInstallation

class CaptureTests(unittest.TestCase):
    def test_same_forward_cache_and_lookahead_parity_across_chunks(self):
        mx.random.seed(19)
        model=Model(ModelArgs(hidden_size=32,num_hidden_layers=2,intermediate_size=64,
            num_attention_heads=2,head_dim=16,global_head_dim=16,num_key_value_heads=1,
            vocab_size=64,vocab_size_per_layer_input=64,num_kv_shared_layers=0,
            hidden_size_per_layer_input=0,sliding_window=16))
        mx.eval(model.parameters())
        prompt=mx.array([1,2,3,4,5,6,7])
        def run(capture):
            cache=model.make_cache();tokens=[];logits=[];vectors=[]
            generator=generate_step(prompt,model,prompt_cache=cache,max_tokens=5,prefill_step_size=2)
            for i,(token,logprobs) in enumerate(generator):
                tokens.append(token);logits.append(np.array(logprobs))
                if capture:
                    row=capture.accepted(i);mx.eval(row);vectors.append(np.array(row))
                    self.assertEqual(capture.position, len(prompt)+i+1)
            return tokens,np.array(logits),[c.offset for c in cache],vectors
        a=run(None)
        body=model.model
        with CaptureInstallation(model,len(prompt)) as capture:
            b=run(capture)
            self.assertEqual(capture.calls[0]['first'],0)
            self.assertEqual(capture.calls[-1]['last'],11)
            self.assertEqual(capture.end_of_prompt.shape,(1,32))
        self.assertIs(model.model,body)
        self.assertEqual(a[0],b[0]);self.assertEqual(a[2],b[2])
        np.testing.assert_allclose(a[1],b[1],rtol=1e-5,atol=1e-5)
        self.assertEqual(len(b[3]),5)

if __name__=='__main__':unittest.main()
