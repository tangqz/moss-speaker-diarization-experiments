"""Configuration loading and fail-closed validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class RecoveryConfig:
    protocol_version: str
    arm: str
    seed: int
    k: int
    optimizer_updates: int
    scheduler_horizon: int
    source_slots_per_update: int
    max_length: int
    sequence_parallel_size: int
    data_parallel_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lambda_max: float
    lambda_warmup_updates: int
    probe_controls: bool
    model_history_probability: float
    sampling_probability: float
    sampling_temperature: float
    repeat_probability: float
    allow_high_perturbation_override: bool
    calibration_file: str
    train_manifest: str
    dev_manifest: str
    model_path: str
    processor_path: str
    official_source: str
    swift_source: str
    baseline_plugin: str
    train_sha256: str
    dev_sha256: str
    upstream_swift_commit: str
    official_moss_commit: str
    gate_file: str
    output_root: str

    def validate(self) -> None:
        if self.protocol_version not in {"local-ul30-v1"}:
            raise ValueError("unsupported protocol_version")
        if self.arm not in {"recovery", "ul"}:
            raise ValueError("arm must be recovery or ul")
        expected_accum = 8
        checks = {
            "k": self.k == 16,
            "optimizer_updates": self.optimizer_updates == 30,
            "scheduler_horizon": self.scheduler_horizon == 402,
            "source_slots_per_update": self.source_slots_per_update == 4,
            "max_length": self.max_length == 131072,
            "sequence_parallel_size": self.sequence_parallel_size == 4,
            "data_parallel_size": self.data_parallel_size == 1,
            "gradient_accumulation_steps": self.gradient_accumulation_steps == expected_accum,
            "learning_rate": self.learning_rate == 1e-6,
            "lambda_max": self.lambda_max == 0.1,
            "lambda_warmup_updates": self.lambda_warmup_updates == 0,
            "model_history_probability": 0 <= self.model_history_probability <= 1,
            "sampling_probability": 0 <= self.sampling_probability <= 1,
            "sampling_temperature": self.sampling_temperature > 0,
            "repeat_probability": 0 <= self.repeat_probability <= 1,
        }
        bad = [key for key, ok in checks.items() if not ok]
        if bad:
            raise ValueError("frozen v2 config mismatch: " + ", ".join(bad))
        if self.repeat_probability != 1.0:
            raise ValueError("local-ul30 uses deterministic insertion and requires repeat_probability=1")
        if self.model_path != "/work/qt28/moss/models/MOSS-Transcribe-Diarize":
            raise ValueError("local-ul30 must initialize from the fresh Base model tree")
        if self.processor_path != self.model_path:
            raise ValueError("processor_path must equal the fresh Base model tree")
        for value, name in [(self.train_sha256, "train_sha256"), (self.dev_sha256, "dev_sha256")]:
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path) -> RecoveryConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    config = RecoveryConfig(**data)
    config.validate()
    return config
