"""Unit tests for LLM dataset preparation, QLoRA config, GGUF export, and Ollama Modelfile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_finetuning.configs.qlora_config import ExportConfig, QLoRAConfig
from llm_finetuning.dataset.prepare_dataset import (
    SEED_CLINICAL_PAIRS,
    build_instruction_sample,
    generate_dataset,
)
from llm_finetuning.export.export_gguf import export_to_gguf, render_modelfile
from llm_finetuning.training.finetune_qlora import format_chat_text, run_qlora_training


def test_qlora_config_defaults_and_validation():
    cfg = QLoRAConfig()
    assert cfg.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.load_in_4bit is True
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert "q_proj" in cfg.target_modules

    with pytest.raises(ValidationError):
        QLoRAConfig(lora_r=1)

    with pytest.raises(ValidationError):
        QLoRAConfig(target_modules=[])


def test_build_instruction_sample_formats_chatml_messages():
    sample = build_instruction_sample(SEED_CLINICAL_PAIRS[0])
    assert len(sample.messages) == 3
    assert [m.role for m in sample.messages] == ["system", "user", "assistant"]
    assert "systolic" in sample.messages[1].content.lower()
    chat_text = format_chat_text(sample.model_dump())
    assert "<|im_start|>system" in chat_text
    assert "<|im_end|>" in chat_text


def test_generate_dataset_writes_valid_jsonl(tmp_path: Path):
    out_file = tmp_path / "cardio_instructions.jsonl"
    samples = generate_dataset(output_path=out_file, corpus_dir=tmp_path)
    assert out_file.is_file()
    lines = [line for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == len(samples) >= 10
    first = json.loads(lines[0])
    assert {"id", "instruction", "input", "output", "messages"}.issubset(first.keys())


def test_finetune_qlora_dry_run(tmp_path: Path):
    dataset_path = tmp_path / "cardio_instructions.jsonl"
    generate_dataset(output_path=dataset_path, corpus_dir=tmp_path)
    cfg = QLoRAConfig(dataset_path=dataset_path, output_dir=tmp_path / "out")
    result = run_qlora_training(cfg, dry_run=True)
    assert result["status"] == "dry_run_ok"
    assert result["base_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert result["num_samples"] >= 10


def test_export_gguf_dry_run_and_modelfile(tmp_path: Path):
    cfg = ExportConfig(adapter_dir=tmp_path / "adapter", gguf_output_dir=tmp_path / "gguf")
    result = export_to_gguf(cfg, dry_run=True)
    assert result["status"] == "dry_run_ok"
    assert result["quantization_method"] == "q4_k_m"

    modelfile_path = Path(__file__).resolve().parents[3] / "llm_finetuning" / "export" / "ollama" / "Modelfile"
    assert modelfile_path.is_file()
    content = modelfile_path.read_text(encoding="utf-8")
    assert "FROM " in content
    assert "TEMPLATE " in content
    assert "SYSTEM " in content
    assert "PARAMETER temperature" in content
    assert render_modelfile().strip() == content.strip()
