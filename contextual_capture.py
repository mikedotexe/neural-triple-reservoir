"""Offline Gemma 4 observation at the normalized hidden/vocabulary boundary.

One wrapper call delegates once. Rows are indexed by absolute input position,
not by generator yield: MLX-LM processes the yielded token in lookahead first.
This module is deliberately not imported by the live coupled server.
"""
from __future__ import annotations

import time


class HiddenCapture:
    boundary = 'gemma4_text.model.final_norm_before_vocabulary_projection'

    def __init__(self, body, prompt_length):
        self.body = body
        self.prompt_length = prompt_length
        self.position = 0
        self.rows = {}
        self.calls = []
        self.end_of_prompt = None

    def __getattr__(self, name):
        return getattr(self.body, name)

    def __call__(self, inputs, *args, **kwargs):
        started = time.perf_counter()
        out = self.body(inputs, *args, **kwargs)
        size = out.shape[1]
        first, last = self.position, self.position + size - 1
        # Prefill chunks do not become reservoir input. Keep only the end row
        # for prompt observation, then a bounded window for accepted decode.
        if first <= self.prompt_length - 1 <= last:
            self.end_of_prompt = out[:, self.prompt_length - 1 - first, :]
        for pos in range(max(first, self.prompt_length), last + 1):
            self.rows[pos] = out[:, pos - first, :]
        self.position += size
        if len(self.rows) > 4:
            raise RuntimeError('capture consumer fell behind; refusing unbounded activation retention')
        self.calls.append(dict(first=first, last=last, phase='prefill' if last < self.prompt_length else 'decode',
                               graph_construction_seconds=time.perf_counter() - started))
        return out

    def accepted(self, output_index):
        absolute_position = self.prompt_length + output_index
        try:
            return self.rows.pop(absolute_position)
        except KeyError as exc:
            raise RuntimeError(f'no same-pass hidden row for output position {absolute_position}') from exc


class CaptureInstallation:
    def __init__(self, model, prompt_length):
        language = getattr(model, 'language_model', model)
        body = language.model
        if type(body).__module__ != 'mlx_lm.models.gemma4_text':
            raise ValueError('only installed Gemma 4 text architecture is qualified')
        self.language, self.body = language, body
        self.capture = HiddenCapture(body, prompt_length)

    def __enter__(self):
        self.language.model = self.capture
        return self.capture

    def __exit__(self, *unused):
        self.language.model = self.body
