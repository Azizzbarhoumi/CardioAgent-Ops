"""Export fine-tuned QLoRA adapter / merged model to quantized GGUF for Ollama serving."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from llm_finetuning.configs.qlora_config import ExportConfig


def render_modelfile(gguf_relative_path: str = "../gguf/cardio-qwen2.5-7b-q4_k_m.gguf") -> str:
    """Render an Ollama Modelfile targeting the exported Qwen-2.5-7B GGUF checkpoint."""
    return (
        f"FROM {gguf_relative_path}\n\n"
        'TEMPLATE """{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{- end }}'
        '{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{- end }}'
        '<|im_start|>assistant\n"""\n\n'
        'SYSTEM """You are a cardiology clinical reasoning assistant specialized in phonocardiogram (PCG) '
        'auscultation and ESC Valvular Heart Disease guidelines. Always ground answers in provided audio '
        'predictions and guideline excerpts, output valid JSON when requested, and include an educational disclaimer."""\n\n'
        "PARAMETER temperature 0.2\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER num_ctx 4096\n"
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|im_start|>"\n'
    )


def export_to_gguf(config: ExportConfig, *, dry_run: bool = False) -> dict:
    """Export LoRA adapter to quantized GGUF using Unsloth (or llama.cpp fallback)."""
    config.gguf_output_dir.mkdir(parents=True, exist_ok=True)
    target_gguf = config.gguf_output_dir / config.gguf_filename

    if dry_run:
        return {
            "status": "dry_run_ok",
            "adapter_dir": str(config.adapter_dir),
            "target_gguf": str(target_gguf),
            "quantization_method": config.quantization_method,
        }

    if not config.adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {config.adapter_dir}")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(config.adapter_dir),
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        model.save_pretrained_gguf(
            str(config.gguf_output_dir),
            tokenizer,
            quantization_method=config.quantization_method,
        )
    except ImportError:
        subprocess.run(
            [
                "python",
                "convert_hf_to_gguf.py",
                str(config.adapter_dir),
                "--outfile",
                str(target_gguf),
                "--outtype",
                config.quantization_method,
            ],
            check=True,
        )

    return {
        "status": "exported",
        "target_gguf": str(target_gguf),
        "quantization_method": config.quantization_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fine-tuned QLoRA model to GGUF.")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quantization", default="q4_k_m", choices=["q4_k_m", "q5_k_m", "q8_0", "f16"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    kwargs = {"quantization_method": args.quantization}
    if args.adapter_dir is not None:
        kwargs["adapter_dir"] = args.adapter_dir
    if args.output_dir is not None:
        kwargs["gguf_output_dir"] = args.output_dir

    cfg = ExportConfig(**kwargs)
    print(json.dumps(export_to_gguf(cfg, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
