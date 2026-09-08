import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from real_model_coupling_study import capture, capacity_ready, digest_file, private_write, read_snapshot


class StudyInputTests(unittest.TestCase):
    def test_capacity_requires_healthy_idle_empty_worker(self):
        self.assertTrue(capacity_ready({"ready": True, "worker": {"phase": "ready", "queue_depth": 0}}))
        for payload in ({}, {"ready": True}, {"ready": False, "worker": {"phase": "ready", "queue_depth": 0}},
                        {"ready": True, "worker": {"phase": "generating", "queue_depth": 0}},
                        {"ready": True, "worker": {"phase": "ready", "queue_depth": 1}}):
            self.assertFalse(capacity_ready(payload))

    def fixture(self, path, entity="astrid", shape=(1, 192)):
        np.savez(path, **{k: np.zeros(shape, dtype=np.float32) for k in ("h1", "h2", "h3")},
                 entity=entity, backend="numpy", timestamp=100., last_live_wall_time=99.,
                 tick_count=10, snapshot_version=2, config_fingerprint="fixture")

    def test_capture_preserves_bytes_and_uses_private_immutable_copy(self):
        with tempfile.TemporaryDirectory() as raw:
            source, output = Path(raw) / "source.npz", Path(raw) / "study" / "a.npz"
            self.fixture(source)
            capture(source, output)
            self.assertEqual(source.read_bytes(), output.read_bytes())
            self.assertEqual(output.stat().st_mode & 0o777, 0o400)
            self.assertEqual(output.parent.stat().st_mode & 0o777, 0o700)
            receipt = json.loads(output.with_suffix(".capture.json").read_text())
            self.assertEqual(receipt["sha256"], digest_file(source)["sha256"])
            self.assertFalse(receipt["live_handle_operation"])
            with self.assertRaises(FileExistsError):
                capture(source, output)

    def test_wrong_entity_or_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw) / "source.npz"
            self.fixture(source, entity="another-handle")
            with self.assertRaises(ValueError):
                read_snapshot(source)
            self.fixture(source, shape=(1, 8))
            with self.assertRaises(ValueError):
                read_snapshot(source)

    def test_private_writer_never_overwrites(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "receipt.json"
            private_write(output, b"original")
            with self.assertRaises(FileExistsError):
                private_write(output, b"replacement")
            self.assertEqual(output.read_bytes(), b"original")
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
