"""Bounded, non-preemptive QoS scheduling primitives for coupled generation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
QOS_CLASSES = frozenset({"interactive", "reflective", "background", "normal"})
QOS_MODES = frozenset({"shadow", "active"})
MAX_PENDING_CAPACITY = 32
PROMOTION_INTERVAL_S = 30.0
CLASS_RANK = {
    "interactive": 0,
    "reflective": 1,
    "normal": 1,
    "background": 2,
}
QUEUE_WAIT_CAP_S = {
    "interactive": 120.0,
    "reflective": 300.0,
    "normal": 300.0,
    "background": 600.0,
}


class InvalidModelQos(ValueError):
    """Raised when an additive QoS envelope is malformed."""


@dataclass(frozen=True)
class ModelQosV1:
    """Validated request metadata. It never contains prompt or response text."""

    request_id: str
    idempotency_key: str
    qos_class: str
    queue_timeout_s: float

    @classmethod
    def from_payload(cls, payload: Any) -> ModelQosV1 | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise InvalidModelQos("model_qos_v1 must be an object")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise InvalidModelQos("model_qos_v1.schema_version must be 1")

        request_id = _bounded_identifier(payload.get("request_id"), "request_id")
        idempotency_key = _bounded_identifier(
            payload.get("idempotency_key"),
            "idempotency_key",
        )
        qos_class = payload.get("class")
        if qos_class not in QOS_CLASSES:
            raise InvalidModelQos(
                "model_qos_v1.class must be interactive, reflective, background, or normal"
            )
        timeout_ms = payload.get("queue_timeout_ms")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int):
            raise InvalidModelQos("model_qos_v1.queue_timeout_ms must be an integer")
        if timeout_ms < 1:
            raise InvalidModelQos("model_qos_v1.queue_timeout_ms must be positive")
        timeout_s = min(timeout_ms / 1000.0, QUEUE_WAIT_CAP_S[qos_class])
        return cls(request_id, idempotency_key, qos_class, timeout_s)

    @property
    def rank(self) -> int:
        return CLASS_RANK[self.qos_class]


def _bounded_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidModelQos(f"model_qos_v1.{field} must be a non-empty string")
    value = value.strip()
    if len(value) > 256:
        raise InvalidModelQos(f"model_qos_v1.{field} exceeds 256 characters")
    return value


def effective_rank(qos: ModelQosV1 | None, waited_s: float) -> int:
    """Return an aged rank; legacy requests use compatibility class normal."""
    base = CLASS_RANK["normal"] if qos is None else qos.rank
    promotions = int(max(0.0, waited_s) // PROMOTION_INTERVAL_S)
    return max(0, base - promotions)


def select_job_index(jobs: list[Any], now_monotonic: float, mode: str) -> int:
    """Select a pending job by FIFO or active QoS without mutating the list."""
    if mode not in QOS_MODES:
        raise ValueError(f"unsupported QoS mode: {mode}")
    if not jobs:
        raise ValueError("cannot select from an empty pending queue")
    if mode == "shadow":
        return min(range(len(jobs)), key=lambda index: jobs[index].arrival_sequence)
    return min(
        range(len(jobs)),
        key=lambda index: (
            effective_rank(
                jobs[index].qos,
                now_monotonic - jobs[index].enqueued_monotonic,
            ),
            jobs[index].arrival_sequence,
        ),
    )


class ModelQosReceiptWriter:
    """Owner-only append writer for bounded scheduling metadata."""

    def __init__(self, path: str | os.PathLike[str] | None):
        self.path = None if path is None else Path(path).expanduser().resolve()
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)

    def append(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        bounded = {
            "schema_version": 1,
            "stream": "model_qos",
            "recorded_at": datetime.now(UTC).isoformat(),
            **event,
        }
        canonical = json.dumps(
            bounded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        bounded["receipt_id"] = "model-qos-" + hashlib.sha256(
            canonical.encode("ascii")
        ).hexdigest()
        line = json.dumps(
            bounded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        with self._lock:
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            try:
                os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
                os.write(descriptor, line.encode("ascii") + b"\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def receipt_identity(qos: ModelQosV1 | None) -> dict[str, Any]:
    """Return non-secret identity fields suitable for a metadata receipt."""
    if qos is None:
        return {
            "request_id_hash": None,
            "idempotency_key_hash": None,
            "class": "normal",
            "versioned": False,
        }
    return {
        "request_id_hash": hashlib.sha256(qos.request_id.encode()).hexdigest(),
        "idempotency_key_hash": hashlib.sha256(
            qos.idempotency_key.encode()
        ).hexdigest(),
        "class": qos.qos_class,
        "versioned": True,
    }
