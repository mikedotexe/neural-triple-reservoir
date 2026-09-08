"""Owner-only, local-asset study driver. Never instantiate the coupled server.

CPU inference avoids sharing the live Metal generation queue. Optional GPU calls
require a healthy idle worker before admission, not an atomic reservation. Input
snapshots are copied from persistence, never requested through live handles.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import http.client
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import resource
import socket
import time

import numpy as np


def digest_file(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"asset changed while hashing: {path.name}")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns}


def private_write(path, content, *, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        if immutable:
            os.fchmod(handle.fileno(), 0o400)


def write_json(path, value):
    private_write(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def read_snapshot(path):
    raw = path.read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        layers = tuple(np.array(data[key], dtype=np.float32, copy=True)
                       for key in ("h1", "h2", "h3"))
        metadata = {key: data[key].item() for key in
                    ("entity", "backend", "timestamp", "last_live_wall_time",
                     "tick_count", "snapshot_version", "config_fingerprint")}
    if any(layer.shape != (1, 192) or not np.isfinite(layer).all() for layer in layers):
        raise ValueError("expected three finite 192-node layers")
    if metadata["entity"] != "astrid" or metadata["backend"] != "numpy":
        raise ValueError("study requires Astrid's persisted numpy handle")
    return raw, layers, metadata


def capture(source, output):
    before = source.stat()
    raw, _, metadata = read_snapshot(source)
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("snapshot changed during capture; retry at a stable point")
    private_write(output, raw, immutable=True)
    receipt = {"source": str(source.resolve()), "copy": str(output.resolve()),
               "sha256": hashlib.sha256(raw).hexdigest(), "metadata": metadata,
               "captured_unix_s": time.time(), "live_handle_operation": False}
    write_json(output.with_suffix(".capture.json"), receipt)
    print(json.dumps(receipt), flush=True)


def capacity_ready(payload):
    return (payload.get("ready") is True and payload.get("worker", {}).get("phase") == "ready"
            and payload.get("worker", {}).get("queue_depth") == 0)


def wait_for_capacity():
    """Read-only admission check; not an atomic reservation or scheduler edit."""
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        connection = http.client.HTTPConnection("127.0.0.1", 8090, timeout=5)
        try:
            connection.request("GET", "/readyz")
            response = connection.getresponse()
            payload = json.loads(response.read(16_384))
            if response.status != 200 or payload.get("ready") is not True:
                raise RuntimeError("live service not healthy; study stopped")
            if capacity_ready(payload):
                return
        finally:
            connection.close()
        time.sleep(1)
    raise RuntimeError("no idle live-service window within 120 seconds")


def study(args):
    # Fail offline even if a future loader tries to resolve a missing asset.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    original_connect = socket.socket.connect
    def confined_connect(sock, address):
        if args.device == "gpu" and address == ("127.0.0.1", 8090):
            return original_connect(sock, address)
        raise RuntimeError("only loopback readiness probes permitted in study process")
    socket.socket.connect = confined_connect
    import mlx.core as mx
    from coupled_astrid_server import _load_mlx_runtime, _build_generation_token_policy
    from mlx_reservoir import MLXTripleReservoir, EmbeddingProjection
    from offline_coupling_replay import (
        ReplaySettings, run_replay, compare_common_prefix, object_digest,
    )
    from triple_reservoir_coreml import build_canonical_reservoir
    from reservoir_service import reservoir_config_fingerprint

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    mx.set_cache_limit(256 * 1024**2)
    root = args.output.resolve()
    if root.exists():
        raise ValueError("study output must be a new directory")
    root.mkdir(parents=True, mode=0o700)
    started = time.time()
    spec = json.loads(args.spec.read_text())
    if set(spec["contexts"]) != {"original", "scoped"}:
        raise ValueError("two preregistered contexts required")
    model_path = args.model.resolve(strict=True)
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()) |
                   {"config.json", "generation_config.json", "tokenizer.json",
                    "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"})
    assets = {name: digest_file(model_path / name) for name in names}
    model_hash = object_digest({name: row["sha256"] for name, row in assets.items()})
    token_hash = object_digest({name: assets[name]["sha256"] for name in
                               ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")})
    snapshots = [read_snapshot(path) for path in (args.state_a, args.state_b)]
    if snapshots[0][2]["timestamp"] >= snapshots[1][2]["timestamp"]:
        raise ValueError("states must be distinct chronological captures")
    if all(np.array_equal(a, b) for a, b in zip(snapshots[0][1], snapshots[1][1])):
        raise ValueError("state contrast is identical")
    np_model, cfg = build_canonical_reservoir()
    fingerprint = reservoir_config_fingerprint(cfg)
    if any(row[2]["config_fingerprint"] != fingerprint for row in snapshots):
        raise ValueError("snapshot config differs from canonical reservoir")
    reservoir = MLXTripleReservoir(np_model)
    reservoir.init_multi_readout(np_model)
    source_root = Path(__file__).resolve().parent
    source_assets = {name: digest_file(source_root / name) for name in
                     ("real_model_coupling_study.py", "offline_coupling_replay.py",
                      "mlx_reservoir.py", "coupled_astrid_server.py",
                      "triple_reservoir_coreml.py", "reservoir_service.py")}
    manifest = {"schema": "real_model_coupling_study_v1", "model_path": str(model_path),
                "assets": assets, "model_sha256": model_hash,
                "source_assets": source_assets,
                "versions": {"python": platform.python_version(),
                             **{name: importlib.metadata.version(name) for name in
                                ("mlx", "mlx-lm", "numpy", "transformers")}},
                "spec_sha256": digest_file(args.spec)["sha256"],
                "snapshots": [{"sha256": hashlib.sha256(raw).hexdigest(), "metadata": meta}
                              for raw, _, meta in snapshots],
                "reservoir_config": asdict(cfg), "config_fingerprint": fingerprint,
                "device": args.device, "network_scope": "loopback_readyz_GET_only" if args.device == "gpu" else "none",
                "report_turn_reconstruction": False, "live_state_changed": False,
                "authority_effect": False}
    write_json(root / "manifest.json", manifest)
    print(json.dumps({"stage": "assets_verified", "model_sha256": model_hash}), flush=True)
    if args.device == "gpu":
        wait_for_capacity()
    model, tokenizer, load_seconds, runtime = _load_mlx_runtime(
        str(model_path), memory_map_requested=True)
    language = model.language_model if hasattr(model, "language_model") else model
    embed = language.model.embed_tokens
    hidden = language.args.hidden_size
    vocab = language.args.vocab_size
    projection = EmbeddingProjection(hidden, cfg.input_dim)
    stop, skip = _build_generation_token_policy(tokenizer)
    prompts = {name: tuple(tokenizer.apply_chat_template(
        [{"role": "system", "content": spec["system"]}, {"role": "user", "content": text}],
        tokenize=True, add_generation_prompt=True)) for name, text in spec["contexts"].items()}
    if any(len(prompt) > 512 for prompt in prompts.values()):
        raise ValueError("bounded study permits at most 512 prompt tokens")
    teacher = tuple(tokenizer.encode(spec["teacher"], add_special_tokens=False))
    if not 3 <= len(teacher) <= 24:
        raise ValueError("teacher continuation must have 3..24 tokens")
    settings = ReplaySettings(seed=spec["seed"], max_tokens=24,
                              temperature=spec["temperature"],
                              coupling_strength=spec["coupling_strength"],
                              stop_tokens=tuple(sorted(stop)), skip_tick_tokens=tuple(sorted(skip)))
    # Base LM logits depend only on the supplied prefix, not reservoir state.
    # Memoizing exact prefixes saves CPU work without sharing mutable KV caches.
    cached = {}
    calls = []
    def logits(prefix):
        if prefix not in cached:
            if time.time() - started > 1800 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 18 * 1024**3:
                raise RuntimeError("study exceeded wall-time or peak RSS bound")
            if args.device == "gpu":
                wait_for_capacity()
            t0 = time.time()
            value = model(mx.array([prefix], dtype=mx.int32), cache=None)[0, -1, :]
            mx.eval(value)
            cached[prefix] = np.array(value.astype(mx.float32))
            calls.append({"prefix_tokens": len(prefix), "seconds": time.time() - t0})
            print(json.dumps({"stage": "inference", "call": len(calls), **calls[-1]}), flush=True)
        return cached[prefix].copy()
    def embedding(token):
        return np.array(embed(mx.array([token], dtype=mx.int32)).astype(mx.float32))
    def trial(context, state_index, *, forced):
        return run_replay(logits_for_prefix=logits, embed_token=embedding,
                          vocab_size=vocab, reservoir=reservoir, projection=projection,
                          initial_state=snapshots[state_index][1], prompt_tokens=prompts[context],
                          model_sha256=model_hash, tokenizer_sha256=token_hash,
                          settings=settings, teacher_tokens=teacher if forced else None)
    cells = [(context, state) for context in prompts for state in (0, 1)]
    paired = {cell: trial(*cell, forced=True) for cell in cells}
    reverse = {cell: trial(*cell, forced=True) for cell in reversed(cells)}
    repeatable = all(np.array_equal(paired[cell].probabilities, reverse[cell].probabilities)
                     for cell in cells)
    if not repeatable:
        raise RuntimeError("reversed-order replay differs")
    contrasts = {
        "state_in_original": compare_common_prefix(paired["original", 0], paired["original", 1]),
        "state_in_scoped": compare_common_prefix(paired["scoped", 0], paired["scoped", 1]),
        "context_in_state_a": compare_common_prefix(paired["original", 0], paired["scoped", 0]),
        "context_in_state_b": compare_common_prefix(paired["original", 1], paired["scoped", 1]),
    }
    free = {cell: trial(*cell, forced=False) for cell in cells}
    for name, original in assets.items():
        current = (model_path / name).stat()
        if (current.st_size, current.st_mtime_ns) != (original["bytes"], original["mtime_ns"]):
            raise RuntimeError("model asset changed during trial")
    if any(digest_file(source_root / name)["sha256"] != original["sha256"]
           for name, original in source_assets.items()):
        raise RuntimeError("study source changed during trial")
    result = {"schema": "real_model_coupling_result_v1", "model_sha256": model_hash,
              "prompt_token_counts": {key: len(value) for key, value in prompts.items()},
              "teacher_tokens": len(teacher), "settings": asdict(settings),
              "contrasts": contrasts, "reversed_order_identical": repeatable,
              "common_prefix_receipts": {str(cell): paired[cell].receipt for cell in cells},
              "free_receipts": {str(cell): free[cell].receipt for cell in cells},
              "load_seconds": load_seconds, "runtime": runtime, "inference_calls": calls,
              "elapsed_seconds": time.time() - started,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "raw_text_included": False, "live_state_changed": False,
              "felt_resolution_claimed": False, "authority_effect": False}
    write_json(root / "result.json", result)
    # Condition key is separate from the bounded texts for a condition-blind read.
    shuffled = np.random.default_rng(spec["seed"]).permutation(len(cells))
    texts, keys = {}, {}
    for index, cell_index in enumerate(shuffled):
        cell = cells[int(cell_index)]
        label = f"sample_{index + 1}"
        texts[label] = tokenizer.decode(list(free[cell].tokens))
        keys[label] = list(cell)
    write_json(root / "continuations.private.json", texts)
    write_json(root / "condition_key.private.json", keys)
    print(json.dumps({"stage": "complete", "result": str(root / "result.json"),
                      "contrasts": contrasts, "reversed_order_identical": repeatable}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture_p = sub.add_parser("capture")
    capture_p.add_argument("--source", type=Path, required=True)
    capture_p.add_argument("--output", type=Path, required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    for name in ("model", "state-a", "state-b", "spec", "output"):
        run_p.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.source, args.output)
    else:
        study(args)


if __name__ == "__main__":
    main()
