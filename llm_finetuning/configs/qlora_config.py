"""Pydantic v2 configuration schemas for QLoRA fine-tuning and GGUF export."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = MODULE_ROOT / "dataset" / "processed" / "cardio_instructions.jsonl"
DEFAULT_OUTPUT_DIR = MODULE_ROOT / "artifacts" / "qwen2.5-7b-cardio-qlora"
DEFAULT_GGUF_DIR = MODULE_ROOT / "export" / "gguf"


class QLoRAConfig(BaseModel):
    """Configuration for 4-bit QLoRA fine-tuning of Qwen-2.5-7B-Instruct via Unsloth/TRL."""

    base_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct")
    load_in_4bit: bool = Field(default=True)
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = Field(default="nf4")
    bnb_4bit_use_double_quant: bool = Field(default=True)
    max_seq_length: int = Field(default=2048, ge=256, le=32768)

    # LoRA adapters
    lora_r: int = Field(default=16, ge=4, le=128)
    lora_alpha: int = Field(default=32, ge=4, le=256)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training hyperparameters
    dataset_path: Path = Field(default=DEFAULT_DATASET_PATH)
    output_dir: Path = Field(default=DEFAULT_OUTPUT_DIR)
    num_train_epochs: int = Field(default=3, ge=1, le=50)
    per_device_train_batch_size: int = Field(default=2, ge=1, le=64)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=2e-4, gt=0.0, le=1e-2)
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    logging_steps: int = Field(default=5, ge=1)
    save_steps: int = Field(default=50, ge=1)
    seed: int = Field(default=42)

    # MLflow tracking
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="llm_qlora_finetuning")

    @field_validator("target_modules")
    @classmethod
    def check_non_empty_modules(cls, modules: list[str]) -> list[str]:
        if not modules:
            raise ValueError("target_modules must not be empty")
        return modules


class ExportConfig(BaseModel):
    """Configuration for exporting fine-tuned QLoRA weights to GGUF for Ollama."""

    adapter_dir: Path = Field(default=DEFAULT_OUTPUT_DIR)
    gguf_output_dir: Path = Field(default=DEFAULT_GGUF_DIR)
    gguf_filename: str = Field(default="cardio-qwen2.5-7b-q4_k_m.gguf")
    quantization_method: Literal["q4_k_m", "q5_k_m", "q8_0", "f16"] = Field(default="q4_k_m")
    max_seq_length: int = Field(default=2048, ge=256, le=32768)
    ollama_model_name: str = Field(default="cardio-qwen2.5-7b")
