"""Unsloth + TRL 4-bit QLoRA fine-tuning script for Qwen-2.5-7B-Instruct with MLflow tracking.

Designed for GPU environments (Google Colab T4/A100 or Kaggle P100/T4).
Supports `--dry-run` locally to validate dataset, QLoRA config, and MLflow setup without GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_finetuning.configs.qlora_config import QLoRAConfig
from llm_finetuning.dataset.prepare_dataset import generate_dataset


def load_jsonl_dataset(dataset_path: Path) -> list[dict]:
    """Read instruction JSONL records and ensure they exist."""
    if not dataset_path.is_file():
        generate_dataset(output_path=dataset_path)
    records = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"Dataset at {dataset_path} is empty.")
    return records


def format_chat_text(sample: dict) -> str:
    """Format a ChatML-like prompt string when tokenizer.apply_chat_template is unavailable."""
    parts = []
    for msg in sample.get("messages", []):
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    return "\n".join(parts)


def run_qlora_training(config: QLoRAConfig, *, dry_run: bool = False) -> dict:
    """Execute 4-bit QLoRA training with Unsloth + TRL SFTTrainer and MLflow logging."""
    records = load_jsonl_dataset(config.dataset_path)

    if dry_run:
        return {
            "status": "dry_run_ok",
            "base_model": config.base_model,
            "num_samples": len(records),
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "load_in_4bit": config.load_in_4bit,
        }

    import mlflow
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    def _apply_template(example: dict) -> dict:
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        else:
            text = format_chat_text(example)
        return {"text": text}

    hf_dataset = Dataset.from_list(records).map(_apply_template)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        seed=config.seed,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        fp16=True,
        optim="adamw_8bit",
        report_to="mlflow",
    )

    with mlflow.start_run(run_name="qwen2.5_7b_cardio_qlora") as run:
        mlflow.log_params(
            {
                "base_model": config.base_model,
                "load_in_4bit": config.load_in_4bit,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "learning_rate": config.learning_rate,
                "num_train_epochs": config.num_train_epochs,
                "num_samples": len(records),
            }
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            args=training_args,
        )
        train_output = trainer.train()
        final_loss = float(train_output.training_loss)
        mlflow.log_metric("final_training_loss", final_loss)

        model.save_pretrained(str(config.output_dir))
        tokenizer.save_pretrained(str(config.output_dir))
        mlflow.log_artifacts(str(config.output_dir), artifact_path="qlora_adapter")

        return {
            "status": "trained",
            "mlflow_run_id": run.info.run_id,
            "final_training_loss": final_loss,
            "output_dir": str(config.output_dir),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="4-bit QLoRA fine-tuning for Qwen-2.5-7B-Instruct.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dataset without GPU training.")
    args = parser.parse_args()

    cfg_kwargs = {
        "base_model": args.base_model,
        "num_train_epochs": args.epochs,
        "mlflow_tracking_uri": args.mlflow_uri,
    }
    if args.dataset is not None:
        cfg_kwargs["dataset_path"] = args.dataset
    if args.output_dir is not None:
        cfg_kwargs["output_dir"] = args.output_dir

    cfg = QLoRAConfig(**cfg_kwargs)
    result = run_qlora_training(cfg, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
