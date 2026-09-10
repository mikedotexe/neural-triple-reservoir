"""Validated coupled-generation contract. No model loading or live state access."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


class InvalidGenerationControls(ValueError):
    pass


def number(name, value, low, high=None, *, integer=False):
    kind = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kind) or not math.isfinite(value):
        raise InvalidGenerationControls(f"{name} must be a finite {'integer' if integer else 'number'}")
    if value < low or (high is not None and value > high):
        raise InvalidGenerationControls(f"{name} must be in [{low}, {high if high is not None else 'infinity'})")
    return value


@dataclass(frozen=True)
class SamplingControls:
    temperature: float = 0.8
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float | None = None
    repetition_context_size: int = 20

    def __post_init__(self):
        number('temperature', self.temperature, 0)
        number('top_p', self.top_p, 0, 1)
        number('top_k', self.top_k, 0, integer=True)
        number('min_p', self.min_p, 0, 1)
        number('repetition_context_size', self.repetition_context_size, 1, integer=True)
        if self.repetition_penalty is not None:
            number('repetition_penalty', self.repetition_penalty, 0)
            if self.repetition_penalty == 0:
                raise InvalidGenerationControls('repetition_penalty must be greater than zero')

    @classmethod
    def from_payload(cls, body):
        fields = cls.__dataclass_fields__
        values = {name: body[name] for name in fields if name in body}
        if any(value is None for value in values.values()):
            raise InvalidGenerationControls('explicit sampling controls cannot be null; omit them for defaults')
        if 'repetition_context_size' in values and 'repetition_penalty' not in values:
            raise InvalidGenerationControls('repetition_context_size requires repetition_penalty')
        return cls(**values)

    def build(self):
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        sampler = make_sampler(temp=self.temperature, top_p=self.top_p,
                               top_k=self.top_k, min_p=self.min_p)
        processors = make_logits_processors(repetition_penalty=self.repetition_penalty,
                                            repetition_context_size=self.repetition_context_size)
        return sampler, processors

    def receipt(self):
        result = asdict(self)
        result['sampling_mode'] = 'greedy' if self.temperature == 0 else 'categorical'
        result['active_filters'] = ([name for name, active in (('top_p', 0 < self.top_p < 1), ('top_k', self.top_k > 0), ('min_p', self.min_p > 0)) if active] if self.temperature != 0 else [])
        result['repetition_active'] = self.repetition_penalty not in (None, 1.0)
        return result


def validate_body(body):
    if not isinstance(body, dict):
        raise InvalidGenerationControls('request must be a JSON object')
    supported = set(SamplingControls.__dataclass_fields__) | {
        'messages', 'max_tokens', 'model', 'stream', 'reservoir_handle',
        'handle_name', 'aperture', 'wide_coupling_strength', 'model_qos_v1',
    }
    unknown = sorted(set(body) - supported)
    if unknown:
        raise InvalidGenerationControls('unsupported request controls: ' + ', '.join(unknown))
    if body.get('stream', False) is not False:
        raise InvalidGenerationControls('streaming is not supported; stream must be false')
    messages = body.get('messages', [])
    if not isinstance(messages, list) or any(
        not isinstance(m, dict) or not isinstance(m.get('role'), str)
        or not isinstance(m.get('content'), str) for m in messages
    ):
        raise InvalidGenerationControls('messages must contain text role/content objects')
    number('max_tokens', body.get('max_tokens', 512), 1, integer=True)
    number('aperture', body.get('aperture', body.get('wide_coupling_strength', 1.0)), 0, 1)
    handle = body.get('reservoir_handle', body.get('handle_name', 'astrid'))
    if not isinstance(handle, str) or not handle.strip():
        raise InvalidGenerationControls('reservoir handle must be a nonempty string')
    for first, second in [('reservoir_handle', 'handle_name'), ('aperture', 'wide_coupling_strength')]:
        if first in body and second in body and body[first] != body[second]:
            raise InvalidGenerationControls(f'conflicting aliases: {first}, {second}')
    return SamplingControls.from_payload(body)


def identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class GenerationResult:
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    filtered_tokens: int
    terminal_tokens: int
    raw_content_chars: int
    controls: dict[str, Any]

    def usage(self):
        return dict(prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
                    total_tokens=self.prompt_tokens + self.completion_tokens)

    def evidence(self):
        return dict(schema='coupled_generation_v1', source='server_reported', controls=self.controls,
                    filtered_tokens=self.filtered_tokens, terminal_tokens=self.terminal_tokens,
                    visible_tokens=self.completion_tokens - self.filtered_tokens - self.terminal_tokens,
                    raw_content_chars=self.raw_content_chars, content_chars=len(self.content),
                    cleanup_removed_chars=self.raw_content_chars - len(self.content),
                    token_count_scope='yielded_tokens_including_filtered_and_terminal_excluding_unyielded_lookahead')
