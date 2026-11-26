"""Prepare cardiology PCG instruction-tuning JSONL dataset for Qwen-2.5-7B-Instruct."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parent
DEFAULT_RAW_CORPUS_DIR = PROJECT_ROOT / "Data" / "external" / "medical_corpus"
DEFAULT_OUTPUT_FILE = MODULE_ROOT / "dataset" / "processed" / "cardio_instructions.jsonl"

CARDIO_SYSTEM_PROMPT = (
    "You are a cardiology clinical reasoning assistant specialized in phonocardiogram (PCG) "
    "auscultation and European Society of Cardiology (ESC) Valvular Heart Disease guidelines. "
    "Provide grounded, concise clinical reasoning without inventing unverified measurements."
)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class InstructionSample(BaseModel):
    id: str
    category: str
    instruction: str = Field(min_length=5)
    input: str = ""
    output: str = Field(min_length=5)
    messages: list[ChatMessage] = Field(min_length=3)


SEED_CLINICAL_PAIRS: list[dict[str, str]] = [
    {
        "id": "pcg_systolic_as_01",
        "category": "systolic_murmur",
        "instruction": "Interpret the PCG classification and recommend ESC-aligned next steps.",
        "input": "PCG prediction: abnormal | confidence: 0.91 | suspected_timing: systolic | patient: 68yo with exertional dyspnea.",
        "output": json.dumps(
            {
                "summary": "Abnormal systolic PCG finding (confidence 0.91) with exertional dyspnea raises suspicion for aortic stenosis or mitral regurgitation.",
                "risk_level": "high",
                "suspected_conditions": ["Aortic stenosis (midsystolic ejection murmur)", "Mitral regurgitation (holosystolic murmur)"],
                "guideline_references": ["ESC 2021 VHD Guidelines: Transthoracic echocardiography (TTE) Class I, Level B"],
                "recommended_next_steps": [
                    "Perform transthoracic echocardiography (TTE) to assess valve gradients and LVEF.",
                    "Obtain a 12-lead ECG and targeted cardiac auscultation for carotid radiation.",
                ],
                "disclaimer": "Educational decision-support only; confirm via clinical exam and TTE.",
                "grounded": True,
            },
            ensure_ascii=False,
        ),
    },
    {
        "id": "pcg_systolic_mr_02",
        "category": "systolic_murmur",
        "instruction": "Differentiate an apical holosystolic murmur from an aortic ejection murmur on PCG.",
        "input": "PCG prediction: abnormal | confidence: 0.87 | suspected_timing: systolic | clinical note: murmur radiates to left axilla.",
        "output": (
            "An abnormal systolic PCG signal radiating to the left axilla is characteristic of mitral regurgitation (MR), "
            "producing a plateau holosystolic murmur across S1-S2. In contrast, aortic stenosis produces a crescendo-decrescendo "
            "midsystolic murmur at the right upper sternal border radiating to the carotids. ESC guidelines recommend TTE to "
            "quantify regurgitant volume, effective regurgitant orifice area, and LV dimensions."
        ),
    },
    {
        "id": "pcg_diastolic_ar_03",
        "category": "diastolic_murmur",
        "instruction": "Assess the clinical urgency of a diastolic murmur flagged by the audio engine.",
        "input": "PCG prediction: abnormal | confidence: 0.89 | suspected_timing: diastolic | clinical note: wide pulse pressure.",
        "output": json.dumps(
            {
                "summary": "Abnormal diastolic PCG signal (confidence 0.89) with widened pulse pressure is strongly suggestive of aortic regurgitation.",
                "risk_level": "high",
                "suspected_conditions": ["Aortic regurgitation (early diastolic decrescendo murmur)", "Mitral stenosis"],
                "guideline_references": ["ESC 2021 VHD Guidelines: Diastolic murmurs require prompt TTE evaluation"],
                "recommended_next_steps": [
                    "Arrange urgent transthoracic echocardiography (TTE) to evaluate aortic root and LV end-systolic diameter.",
                    "Evaluate for hemodynamic instability or acute heart failure signs.",
                ],
                "disclaimer": "Educational decision-support only; confirm via clinical exam and TTE.",
                "grounded": True,
            },
            ensure_ascii=False,
        ),
    },
    {
        "id": "pcg_diastolic_ms_04",
        "category": "diastolic_murmur",
        "instruction": "Summarize ESC guideline workup for a suspected mid-diastolic rumble.",
        "input": "PCG prediction: abnormal | confidence: 0.84 | suspected_timing: diastolic | note: opening snap at apex.",
        "output": (
            "A low-pitched mid-diastolic rumble preceded by an opening snap at the cardiac apex strongly indicates mitral stenosis (MS). "
            "Per ESC Valvular Heart Disease guidelines, transthoracic echocardiography is indicated to measure mean transmitral gradient, "
            "Wilkins valve score, mitral valve area by planimetry, and pulmonary artery systolic pressure."
        ),
    },
    {
        "id": "pcg_normal_s1s2_05",
        "category": "normal_pcg",
        "instruction": "Interpret a normal PCG screening output and state appropriate safety guardrails.",
        "input": "PCG prediction: normal | confidence: 0.94 | suspected_timing: normal | patient: asymptomatic adult routine check.",
        "output": json.dumps(
            {
                "summary": "Normal PCG screening (confidence 0.94) indicating regular S1 and S2 heart sounds without detected pathological murmurs.",
                "risk_level": "low",
                "suspected_conditions": ["Normal S1/S2 phonocardiogram"],
                "guideline_references": ["ESC PCG Screening Protocol: Automated screening is an adjunct tool"],
                "recommended_next_steps": [
                    "Continue routine clinical follow-up.",
                    "Perform formal auscultation or TTE if new cardiopulmonary symptoms develop.",
                ],
                "disclaimer": "Educational decision-support only; automated screening does not rule out all cardiac conditions.",
                "grounded": True,
            },
            ensure_ascii=False,
        ),
    },
    {
        "id": "pcg_continuous_pda_06",
        "category": "continuous_murmur",
        "instruction": "Explain the acoustic pattern and likely etiology of a continuous murmur on PCG.",
        "input": "PCG prediction: abnormal | confidence: 0.88 | suspected_timing: continuous | site: left infraclavicular area.",
        "output": (
            "A continuous machinery-like murmur enveloping S2 in the left infraclavicular region is classic for patent ductus arteriosus (PDA). "
            "Per ESC Adult Congenital Heart Disease guidelines, TTE is indicated to visualize the ductal shunt, assess left heart volume overload, "
            "and estimate pulmonary artery pressures before considering transcatheter device closure."
        ),
    },
    {
        "id": "pcg_innocent_vs_pathologic_07",
        "category": "differential_diagnosis",
        "instruction": "Compare innocent systolic flow murmurs against pathological systolic murmurs.",
        "input": "PCG prediction: abnormal | confidence: 0.72 | suspected_timing: systolic | patient: 19yo febrile athlete.",
        "output": (
            "In a febrile young adult, an early-to-mid systolic soft murmur (Grade I-II/VI) without radiation and with normal S2 splitting "
            "may represent an innocent hyperdynamic flow murmur. However, because automated PCG flagged an abnormal systolic pattern, "
            "repeat auscultation after fever resolution and TTE if the murmur persists >= Grade III/VI or has abnormal S2 are recommended."
        ),
    },
    {
        "id": "pcg_snr_artifact_guardrail_08",
        "category": "quality_guardrail",
        "instruction": "How should low Signal-to-Noise Ratio (SNR) affect clinical interpretation of PCG predictions?",
        "input": "PCG prediction: abnormal | confidence: 0.56 | suspected_timing: unknown | SNR: 2.1 dB (noisy recording).",
        "output": (
            "A low SNR recording (2.1 dB) with borderline model confidence (0.56) indicates substantial ambient or friction noise. "
            "Before acting on the abnormal flag, repeat the PCG recording in a quiet room with firm stethoscope skin contact, "
            "and rely on direct clinical auscultation."
        ),
    },
    {
        "id": "pcg_gallop_s3_s4_09",
        "category": "extra_heart_sounds",
        "instruction": "Explain the clinical significance of an S3 gallop detected during early diastole.",
        "input": "PCG prediction: abnormal | confidence: 0.86 | suspected_timing: diastolic | age: 72yo with ankle edema.",
        "output": (
            "In a 72-year-old patient with peripheral edema, a low-frequency early diastolic S3 gallop suggests elevated left ventricular "
            "filling pressures and potential systolic heart failure (HFrEF). ESC Heart Failure guidelines recommend natriuretic peptides (BNP/NT-proBNP), "
            "12-lead ECG, and prompt transthoracic echocardiography."
        ),
    },
    {
        "id": "pcg_tricuspid_regurg_10",
        "category": "systolic_murmur",
        "instruction": "What auscultatory maneuver helps distinguish tricuspid regurgitation from mitral regurgitation?",
        "input": "PCG prediction: abnormal | confidence: 0.83 | suspected_timing: systolic | site: lower left sternal border.",
        "output": (
            "Carvallo's sign: tricuspid regurgitation murmurs at the lower left sternal border increase in intensity during inspiration "
            "due to augmented venous return to the right heart, whereas mitral regurgitation murmurs at the apex do not accentuate with inspiration."
        ),
    },
]


def build_instruction_sample(raw_item: dict[str, str]) -> InstructionSample:
    """Format a raw instruction/input/output dictionary into ChatML + Alpaca fields."""
    user_content = raw_item["instruction"].strip()
    if raw_item.get("input", "").strip():
        user_content = f"{user_content}\n\nContext: {raw_item['input'].strip()}"

    messages = [
        ChatMessage(role="system", content=CARDIO_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
        ChatMessage(role="assistant", content=raw_item["output"].strip()),
    ]
    return InstructionSample(
        id=raw_item["id"],
        category=raw_item.get("category", "cardiology_pcg"),
        instruction=raw_item["instruction"],
        input=raw_item.get("input", ""),
        output=raw_item["output"],
        messages=messages,
    )


def _load_external_corpus(corpus_dir: Path | None) -> list[dict[str, str]]:
    """Optionally ingest extra .jsonl or .txt medical corpus entries from Data/external/medical_corpus."""
    extra: list[dict[str, str]] = []
    if not corpus_dir or not corpus_dir.is_dir():
        return extra
    for path in sorted(corpus_dir.glob("*.jsonl")):
        for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "instruction" in obj and "output" in obj:
                    extra.append(
                        {
                            "id": str(obj.get("id", f"{path.stem}_{idx}")),
                            "category": str(obj.get("category", "medical_corpus")),
                            "instruction": str(obj["instruction"]),
                            "input": str(obj.get("input", "")),
                            "output": str(obj["output"]),
                        }
                    )
            except json.JSONDecodeError:
                continue
    return extra


def generate_dataset(
    *,
    output_path: Path = DEFAULT_OUTPUT_FILE,
    corpus_dir: Path | None = DEFAULT_RAW_CORPUS_DIR,
) -> list[InstructionSample]:
    """Validate and write cardiology PCG instruction pairs to JSONL."""
    raw_items = list(SEED_CLINICAL_PAIRS) + _load_external_corpus(corpus_dir)
    samples = [build_instruction_sample(item) for item in raw_items]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cardiology PCG instruction JSONL dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="Output JSONL path.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_RAW_CORPUS_DIR, help="Optional extra corpus dir.")
    args = parser.parse_args()
    samples = generate_dataset(output_path=args.output, corpus_dir=args.corpus_dir)
    print(f"Generated {len(samples)} instruction samples at {args.output}")


if __name__ == "__main__":
    main()
