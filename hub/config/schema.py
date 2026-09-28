from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    path: str = ""
    token_type: str = "phaedra"
    variable_order: list[str] = field(default_factory=lambda: ["rho", "u", "v", "p"])
    input_variables: list[str] = field(default_factory=lambda: ["rho", "u", "v", "p"])
    output_variables: list[str] = field(default_factory=lambda: ["rho", "u", "v", "p"])
    split_train: str = "train"
    split_val: str = "val"
    pair_mode_train: str = "all_even_forward"
    pair_mode_val: str = "fixed"
    time_start: int = 0
    time_end: int = 14
    time_step: int = 2
    fixed_input_time: int = 0
    fixed_output_time: int = 14
    max_train_members: int | None = None
    max_val_members: int | None = None
    num_workers: int = 4
    batch_size_train: int = 32
    batch_size_val: int = 8
    amp_vocab_size: int = 1024
    morph_vocab_size: int = 8640
    fsq_vocab_size: int = 8640


@dataclass
class BudgetConfig:
    target_params: int = 38_000_000
    tolerance: int = 1_000_000


@dataclass
class TrainingConfig:
    epochs: int = 20
    resume_from: str | None = None
    resume_exact_batch: bool = True
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    focal_gamma: float = 2.0
    focal_alpha: float | None = None
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"
    log_every: int = 20
    val_every: int = 20
    checkpoint_every: int = 1000
    hybrid_morph_input: str = "gt"
    hybrid_lr_amp: float | None = None
    hybrid_lr_morph: float | None = None
    hybrid_weight_decay_amp: float | None = None
    hybrid_weight_decay_morph: float | None = None


@dataclass
class WandBConfig:
    enabled: bool = True
    project: str = "operator-learning-hub"
    entity: str | None = None
    run_name: str | None = None
    mode: str = "online"


@dataclass
class ValidationConfig:
    max_batches: int = 8
    max_plots: int = 6
    output_dir: str = "${oc.env:PHAEDRA_OUTPUT_ROOT}/operators"
    decoder_enabled: bool = False
    phaedra_root: str | None = None
    model_name: str | None = None
    model_path: str | None = None
    use_ema: bool = True


@dataclass
class RuntimeConfig:
    seed: int = 42
    ddp: bool = False
    tf32: bool = True


@dataclass
class TestingConfig:
    checkpoint: str | None = None
    split: str = "val"
    max_plots: int = 6
    max_members: int | None = None
    input_time_idx: int = 0
    final_time_idx: int = 14
    source_fields_path: str | None = None
    denormalization_config: str | None = None
    dataset_name: str = "ceu_kh"
    output_dir: str | None = None
    rollouts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HubConfig:
    model_type: str = "seq2seq"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    denormalization: dict[str, Any] = field(default_factory=dict)
