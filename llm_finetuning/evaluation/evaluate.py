"""Evaluation helpers for schema compliance of fine-tuned LLM outputs."""

from __future__ import annotations

import json

from agents.schemas.models import ClinicalReasoningReport


def check_json_schema_compliance(raw_outputs: list[str]) -> dict[str, float]:
    """Compute fraction of LLM outputs conforming strictly to ClinicalReasoningReport."""
    if not raw_outputs:
        return {"compliance_rate": 0.0, "total": 0}
    valid = 0
    for text in raw_outputs:
        try:
            ClinicalReasoningReport.model_validate(json.loads(text))
            valid += 1
        except Exception:
            continue
    return {"compliance_rate": valid / len(raw_outputs), "total": len(raw_outputs)}
