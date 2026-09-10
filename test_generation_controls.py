import threading
import unittest
from unittest.mock import patch
import mlx.core as mx
from generation_controls import SamplingControls, validate_body, InvalidGenerationControls, GenerationResult
from coupled_http_gateway import ModelRuntimeCoordinator, GenerationRequest, ConflictingRetry
from test_coupled_http_gateway import GatewayTests, request_json, qos

class ControlDeliveryTests(GatewayTests):
    # Exercise actual HTTP JSON parsing, admission and worker invocation, then
    # the installed sampler construction, without a model or live service.
    def test_http_controls_reach_installed_sampler_and_repetition_processor(self):
        class Server:
            coupling_strength=.1
            vocab_size=8
            def health_snapshot(self):return {}
            def generate_coupled(self,messages,temperature,max_tokens,**kwargs):
                controls=kwargs['sampling'];sampler,processors=controls.build()
                self.controls=controls;self.processors=processors
                self.token=int(sampler(mx.log(mx.array([[.01,.02,.03,.04,.05,.06,.09,.7]]))).item())
                return GenerationResult('answer','length',17,5,1,0,6,controls.receipt())
        server=Server();runtime=ModelRuntimeCoordinator(enforce_main_thread=False);runtime.attach_server(server)
        _,url=self.start_gateway(runtime);stop=threading.Event();worker=threading.Thread(target=runtime.run_worker,args=(stop,));worker.start()
        self.addCleanup(lambda:(stop.set(),worker.join(2)))
        status,result,_=request_json(url+'/v1/chat/completions',body=dict(messages=[],top_p=.5,top_k=1,min_p=.1,repetition_penalty=1.2))
        self.assertEqual(status,200);self.assertEqual(server.token,7);self.assertEqual(len(server.processors),1)
        self.assertEqual(server.controls.repetition_context_size,20)
        self.assertEqual(result['choices'][0]['finish_reason'],'length');self.assertEqual(result['usage']['total_tokens'],22)
        self.assertEqual(result['coupled_generation_v1']['controls']['top_p'],.5)
        status,result,_=request_json(url+'/v1/chat/completions',body=dict(messages=[]))
        self.assertEqual(status,200);self.assertEqual(server.controls,SamplingControls());self.assertEqual(server.processors,[])
        invalid=[{'top_p':1.1},{'temperature':True},{'max_tokens':1.5},{'top_k':9},{'seed':3},{'stop':['x']},{'stream':True},{'repetition_penalty':0},{'repetition_context_size':4},{'min_p':float('nan')},[]]
        for body in invalid:
            status,_,_=request_json(url+'/v1/chat/completions',body=body);self.assertEqual(status,400,body)

    def test_idempotency_rejects_changed_controls_content_handle_aperture(self):
        class Server: coupling_strength=.1
        runtime=ModelRuntimeCoordinator(enforce_main_thread=False);runtime.attach_server(Server())
        base=dict(messages=[{'role':'user','content':'first'}],temperature=.8,max_tokens=12,handle_name='astrid',aperture=1.,qos=qos('normal','one'))
        first=runtime.submit(GenerationRequest(**base))
        self.assertIs(first,runtime.submit(GenerationRequest(**base)))
        for change in (dict(messages=[{'role':'user','content':'second'}]),dict(sampling=SamplingControls(top_p=.95)),dict(max_tokens=13),dict(handle_name='minime'),dict(aperture=.5)):
            with self.assertRaises(ConflictingRetry):runtime.submit(GenerationRequest(**(base|change)))
        base['messages'][0]['content']='mutated after enqueue'
        with self.assertRaises(ConflictingRetry):runtime.submit(GenerationRequest(**base))

class SamplingTests(unittest.TestCase):
    def test_invalid_values_and_defaults(self):
        self.assertEqual(validate_body({}),SamplingControls())
        for body in ({'temperature':-1},{'top_k':False},{'top_k':-1},{'top_p':None},{'repetition_penalty':float('inf')},{'messages':'oops'},{'aperture':2}):
            with self.assertRaises(InvalidGenerationControls):validate_body(body)
        self.assertEqual(SamplingControls(temperature=0,top_p=.5).receipt()['active_filters'],[])

if __name__=='__main__':unittest.main()
