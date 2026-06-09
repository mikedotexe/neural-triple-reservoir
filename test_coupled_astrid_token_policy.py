"""Tests for coupled Astrid generation token policy helpers."""

from __future__ import annotations

import unittest

from coupled_astrid_server import (
    _build_generation_token_policy,
    _clean_generated_text,
    _token_to_int,
)


class FakeTokenizer:
    eos_token_id = 1
    all_special_ids = [0, 1, 2, 3, 98, 100, 101, 105, 106]

    _ids = {
        "<eos>": [1],
        "<bos>": [2],
        "<pad>": [0],
        "<unk>": [3],
        "<|think|>": [98],
        "<|channel>": [100],
        "<channel|>": [101],
        "<|turn>": [105],
        "<turn|>": [106],
        "<|im_end|>": [107],
        "<|eot_id|>": [108],
        "<|endoftext|>": [109],
    }

    def encode(self, text: str, add_special_tokens: bool = False):
        return list(self._ids.get(text, [42, 43]))


class TokenPolicyTests(unittest.TestCase):
    def test_policy_stops_turn_and_channel_closers(self):
        stop_ids, skip_ids = _build_generation_token_policy(FakeTokenizer())

        self.assertIn(1, stop_ids)
        self.assertIn(101, stop_ids)
        self.assertIn(105, stop_ids)
        self.assertIn(106, stop_ids)
        self.assertIn(100, skip_ids)
        self.assertIn(98, skip_ids)
        self.assertNotIn(101, skip_ids)

    def test_clean_generated_text_removes_channel_artifacts(self):
        text = "thought\n<channel|>\nASTRID_CANARY_OK<turn|><eos>"

        self.assertEqual(_clean_generated_text(text), "ASTRID_CANARY_OK")

    def test_token_to_int_accepts_plain_int(self):
        self.assertEqual(_token_to_int(106), 106)


if __name__ == "__main__":
    unittest.main()
